"""Merge any number of Krea 2 LoRA safetensors without loading the base model."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


DTYPE_TO_NUMPY = {
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "BF16": np.dtype("<u2"),
}
DTYPE_SIZE = {key: dtype.itemsize for key, dtype in DTYPE_TO_NUMPY.items()}
OUTPUT_DTYPES = {"f16": "F16", "bf16": "BF16", "f32": "F32"}
LORA_TAG = re.compile(
    r"<lora\s*:\s*([^:<>]+?)\s*:\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*>",
    re.IGNORECASE,
)
NAME_METADATA_KEYS = (
    "modelspec.title",
    "title",
    "model_name",
    "lora_name",
    "ss_output_name",
    "name",
    "repoId",
)


class MergeError(Exception):
    """A user-facing validation error."""


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


class SafeTensorFile:
    """Minimal, mmap-backed safetensors reader with BF16 support."""

    def __init__(self, path: Path):
        self.path = path
        try:
            with path.open("rb") as handle:
                raw_length = handle.read(8)
                if len(raw_length) != 8:
                    raise MergeError(f"{path}: truncated safetensors header")
                header_length = struct.unpack("<Q", raw_length)[0]
                header = json.loads(handle.read(header_length))
        except (OSError, json.JSONDecodeError, struct.error) as exc:
            raise MergeError(f"Could not read {path}: {exc}") from exc

        self.metadata = header.pop("__metadata__", {})
        self.data_start = 8 + header_length
        self.tensors: dict[str, TensorInfo] = {}
        for name, item in header.items():
            dtype = item["dtype"]
            if dtype not in DTYPE_TO_NUMPY:
                raise MergeError(
                    f"{path}: tensor {name!r} uses unsupported dtype {dtype}; "
                    "supported dtypes are F16, BF16, and F32"
                )
            start, end = item["data_offsets"]
            shape = tuple(item["shape"])
            expected = math.prod(shape) * DTYPE_SIZE[dtype]
            if end - start != expected:
                raise MergeError(f"{path}: invalid byte size for tensor {name!r}")
            self.tensors[name] = TensorInfo(dtype, shape, start, end)
        self._mmap: np.memmap | None = None

    def __enter__(self) -> "SafeTensorFile":
        self._mmap = np.memmap(self.path, mode="r", dtype=np.uint8)
        return self

    def __exit__(self, *_: object) -> None:
        if self._mmap is not None:
            self._mmap._mmap.close()
            self._mmap = None

    def array(self, name: str) -> np.ndarray:
        info = self.tensors[name]
        if self._mmap is None:
            raise RuntimeError("SafeTensorFile must be used as a context manager")
        raw = self._mmap[self.data_start + info.start : self.data_start + info.end]
        values = np.frombuffer(raw, dtype=DTYPE_TO_NUMPY[info.dtype]).reshape(info.shape)
        if info.dtype == "BF16":
            return (values.astype(np.uint32) << 16).view(np.float32)
        return values


@dataclass
class ModuleWeights:
    a_name: str | None = None
    b_name: str | None = None
    diff_name: str | None = None

    @property
    def is_diff(self) -> bool:
        return self.diff_name is not None


@dataclass
class Adapter:
    file: SafeTensorFile
    weight: float
    alpha: float
    rank_hint: int | None
    modules: dict[str, ModuleWeights]
    source_format: str


def _canonicalize(base: str) -> tuple[str, str]:
    """Return (native Krea 2 base name, detected source format)."""
    if base.startswith("diffusion_model."):
        return base, "comfy"
    if not base.startswith("transformer."):
        raise MergeError(f"Unrecognized Krea 2 LoRA module name: {base}")

    name = base
    exact = {
        "transformer.img_in": "diffusion_model.first",
        "transformer.time_embed.linear_1": "diffusion_model.tmlp.0",
        "transformer.time_embed.linear_2": "diffusion_model.tmlp.2",
        "transformer.time_mod_proj": "diffusion_model.tproj.1",
        "transformer.text_fusion.projector": "diffusion_model.txtfusion.projector",
        "transformer.txt_in.linear_1": "diffusion_model.txtmlp.1",
        "transformer.txt_in.linear_2": "diffusion_model.txtmlp.3",
        "transformer.final_layer.linear": "diffusion_model.last.linear",
    }
    if name in exact:
        return exact[name], "diffusers"

    name = name.replace("transformer.transformer_blocks.", "diffusion_model.blocks.", 1)
    name = name.replace("transformer.text_fusion.", "diffusion_model.txtfusion.", 1)
    replacements = (
        (".attn.to_gate", ".attn.gate"),
        (".attn.to_k", ".attn.wk"),
        (".attn.to_out.0", ".attn.wo"),
        (".attn.to_q", ".attn.wq"),
        (".attn.to_v", ".attn.wv"),
        (".ff.", ".mlp."),
    )
    for old, new in replacements:
        name = name.replace(old, new)
    if not name.startswith("diffusion_model."):
        raise MergeError(f"Unsupported Diffusers Krea 2 module name: {base}")
    return name, "diffusers"


def _to_diffusers(base: str) -> str:
    exact = {
        "diffusion_model.first": "transformer.img_in",
        "diffusion_model.tmlp.0": "transformer.time_embed.linear_1",
        "diffusion_model.tmlp.2": "transformer.time_embed.linear_2",
        "diffusion_model.tproj.1": "transformer.time_mod_proj",
        "diffusion_model.txtfusion.projector": "transformer.text_fusion.projector",
        "diffusion_model.txtmlp.1": "transformer.txt_in.linear_1",
        "diffusion_model.txtmlp.3": "transformer.txt_in.linear_2",
        "diffusion_model.last.linear": "transformer.final_layer.linear",
    }
    if base in exact:
        return exact[base]
    name = base.replace("diffusion_model.blocks.", "transformer.transformer_blocks.", 1)
    name = name.replace("diffusion_model.txtfusion.", "transformer.text_fusion.", 1)
    replacements = (
        (".attn.gate", ".attn.to_gate"),
        (".attn.wk", ".attn.to_k"),
        (".attn.wo", ".attn.to_out.0"),
        (".attn.wq", ".attn.to_q"),
        (".attn.wv", ".attn.to_v"),
        (".mlp.", ".ff."),
    )
    for old, new in replacements:
        name = name.replace(old, new)
    if not name.startswith("transformer."):
        raise MergeError(f"Cannot convert module to Diffusers format: {base}")
    return name


def _metadata_number(metadata: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in metadata:
            try:
                return float(metadata[key])
            except (TypeError, ValueError):
                raise MergeError(f"Metadata field {key!r} is not numeric: {metadata[key]!r}")
    return None


def _normalized_alias(value: str) -> str:
    return value.strip().casefold()


def lora_aliases(file: SafeTensorFile) -> set[str]:
    """Extract WebUI lookup names from metadata, plus filename fallbacks."""
    aliases: set[str] = set()
    for key in NAME_METADATA_KEYS:
        value = file.metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        aliases.add(_normalized_alias(value))
        if "/" in value:
            aliases.add(_normalized_alias(value.rsplit("/", 1)[-1]))

    stem = file.path.stem
    aliases.add(_normalized_alias(stem))
    if stem.casefold().endswith(".metadata"):
        aliases.add(_normalized_alias(stem[: -len(".metadata")]))
    return {alias for alias in aliases if alias}


def parse_lora_prompt(prompt: str) -> list[tuple[str, float]]:
    tags = [(match.group(1).strip(), float(match.group(2))) for match in LORA_TAG.finditer(prompt)]
    if not tags:
        raise MergeError("No valid <lora:name:weight> tags were found in --prompt")
    remainder = LORA_TAG.sub("", prompt)
    if re.search(r"<\s*lora\s*:", remainder, re.IGNORECASE):
        raise MergeError("At least one malformed <lora:name:weight> tag was found in --prompt")
    return tags


def resolve_prompt_inputs(prompt: str, directories: list[Path]) -> list[tuple[Path, float]]:
    tags = parse_lora_prompt(prompt)
    candidates: dict[str, set[Path]] = {}
    scanned = 0
    for directory in directories:
        if not directory.is_dir():
            raise MergeError(f"LoRA directory does not exist or is not a directory: {directory}")
        for path in directory.rglob("*.safetensors"):
            try:
                file = SafeTensorFile(path)
                # Ignore full checkpoints and unrelated safetensors after a header-only check.
                inspect_adapter(file, 1.0)
            except MergeError:
                continue
            scanned += 1
            resolved = path.resolve()
            for alias in lora_aliases(file):
                candidates.setdefault(alias, set()).add(resolved)

    resolved_weights: dict[Path, float] = {}
    for requested_name, weight in tags:
        alias = _normalized_alias(requested_name)
        matches = candidates.get(alias, set())
        if not matches:
            available = sorted(candidates)
            preview = ", ".join(available[:20]) or "none"
            suffix = " ..." if len(available) > 20 else ""
            raise MergeError(
                f"Could not resolve LoRA name {requested_name!r} after scanning {scanned} Krea 2 LoRAs. "
                f"Available aliases: {preview}{suffix}"
            )
        if len(matches) > 1:
            paths = ", ".join(str(path) for path in sorted(matches))
            raise MergeError(f"LoRA name {requested_name!r} is ambiguous; it matches: {paths}")
        path = next(iter(matches))
        resolved_weights[path] = resolved_weights.get(path, 0.0) + weight
    return list(resolved_weights.items())


def inspect_adapter(file: SafeTensorFile, weight: float) -> Adapter:
    modules: dict[str, ModuleWeights] = {}
    formats: set[str] = set()
    for name in file.tensors:
        if not name.endswith(".lora_A.weight"):
            continue
        base = name[: -len(".lora_A.weight")]
        b_name = base + ".lora_B.weight"
        if b_name not in file.tensors:
            raise MergeError(f"{file.path}: missing pair {b_name!r}")
        canonical, source_format = _canonicalize(base)
        if canonical in modules:
            raise MergeError(f"{file.path}: duplicate canonical module {canonical!r}")
        a_info, b_info = file.tensors[name], file.tensors[b_name]
        if len(a_info.shape) != 2 or len(b_info.shape) != 2:
            raise MergeError(f"{file.path}: {base} is not a linear 2D LoRA")
        rank = a_info.shape[0]
        if b_info.shape[1] != rank:
            raise MergeError(f"{file.path}: rank mismatch in {base}")
        modules[canonical] = ModuleWeights(a_name=name, b_name=b_name)
        formats.add(source_format)

    for name, info in file.tensors.items():
        if not name.endswith(".diff"):
            continue
        base = name[: -len(".diff")]
        canonical, source_format = _canonicalize(base)
        if canonical in modules:
            raise MergeError(f"{file.path}: duplicate canonical module {canonical!r}")
        if len(info.shape) != 2:
            raise MergeError(f"{file.path}: raw delta {name!r} is not a 2D weight matrix")
        modules[canonical] = ModuleWeights(diff_name=name)
        formats.add(source_format)

    if not modules:
        raise MergeError(f"{file.path}: no Krea 2 lora_A/lora_B pairs or 2D .diff tensors found")
    if len(formats) != 1:
        raise MergeError(f"{file.path}: mixed naming formats are not supported within one file")

    ranks = {
        min(file.tensors[pair.diff_name].shape) if pair.is_diff else file.tensors[pair.a_name].shape[0]
        for pair in modules.values()
    }
    rank_hint_value = _metadata_number(file.metadata, ("ss_network_dim", "lora_rank", "rank"))
    rank_hint = int(rank_hint_value) if rank_hint_value is not None else None
    alpha = _metadata_number(file.metadata, ("ss_network_alpha", "lora_alpha", "network_alpha"))
    # PEFT commonly defaults alpha to rank. For rank-pattern adapters the per-module
    # rank is used below when no explicit alpha exists.
    if alpha is None:
        alpha = float(rank_hint) if rank_hint is not None else float(next(iter(ranks)))
    return Adapter(file, weight, alpha, rank_hint, modules, next(iter(formats)))


def module_shape(adapter: Adapter, module: str) -> tuple[int, int, int]:
    """Return (rank, input features, output features)."""
    weights = adapter.modules[module]
    if weights.is_diff:
        out_features, in_features = adapter.file.tensors[weights.diff_name].shape
        return min(out_features, in_features), in_features, out_features
    a_shape = adapter.file.tensors[weights.a_name].shape
    b_shape = adapter.file.tensors[weights.b_name].shape
    return a_shape[0], a_shape[1], b_shape[0]


def module_arrays(adapter: Adapter, module: str) -> tuple[np.ndarray, np.ndarray]:
    """Return exact float32 (A, B) factors for LoRA or raw-delta input."""
    weights = adapter.modules[module]
    if not weights.is_diff:
        return (
            adapter.file.array(weights.a_name).astype(np.float32),
            adapter.file.array(weights.b_name).astype(np.float32),
        )
    delta = adapter.file.array(weights.diff_name).astype(np.float32)
    out_features, in_features = delta.shape
    if out_features <= in_features:
        return delta, np.eye(out_features, dtype=np.float32)
    return np.eye(in_features, dtype=np.float32), delta


def float_to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    # Round-to-nearest-even before truncating the low 16 bits.
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2")


def encode(values: np.ndarray, dtype: str) -> bytes:
    if dtype == "BF16":
        return float_to_bf16(values).tobytes(order="C")
    return np.asarray(values, dtype=DTYPE_TO_NUMPY[dtype]).tobytes(order="C")


def choose_dtype(adapters: list[Adapter], requested: str) -> str:
    if requested != "auto":
        return OUTPUT_DTYPES[requested]
    seen: set[str] = set()
    for adapter in adapters:
        for weights in adapter.modules.values():
            names = [weights.diff_name] if weights.is_diff else [weights.a_name, weights.b_name]
            seen.update(adapter.file.tensors[name].dtype for name in names)
    if "BF16" in seen:
        return "BF16"
    if "F16" in seen:
        return "F16"
    return "F32"


@dataclass(frozen=True)
class OutputTensor:
    name: str
    shape: tuple[int, ...]
    module: str
    kind: str


def output_tensors(adapters: list[Adapter], output_format: str) -> list[OutputTensor]:
    result: list[OutputTensor] = []
    for module in sorted(set().union(*(adapter.modules for adapter in adapters))):
        contributors = [a for a in adapters if module in a.modules and a.weight != 0]
        if not contributors:
            continue
        shapes = [module_shape(adapter, module) for adapter in contributors]
        if len({shape[1] for shape in shapes}) != 1 or len({shape[2] for shape in shapes}) != 1:
            details = ", ".join(
                f"{adapter.file.path.name}: rank={rank}, in={in_features}, out={out_features}"
                for adapter, (rank, in_features, out_features) in zip(contributors, shapes)
            )
            raise MergeError(f"Incompatible dimensions for {module}: {details}")
        rank = sum(shape[0] for shape in shapes)
        base = module if output_format == "comfy" else _to_diffusers(module)
        result.append(OutputTensor(base + ".lora_A.weight", (rank, shapes[0][1]), module, "A"))
        result.append(OutputTensor(base + ".lora_B.weight", (shapes[0][2], rank), module, "B"))
    return result


def write_safetensors(
    path: Path,
    tensors: list[OutputTensor],
    adapters: list[Adapter],
    dtype: str,
    output_format: str,
) -> None:
    metadata = {
        "modelspec.architecture": "krea2/lora",
        "modelspec.implementation": "diffusers" if output_format == "diffusers" else "comfy",
        "merge_method": "weighted_rank_concatenation",
        "merge_inputs": json.dumps(
            [{"path": str(a.file.path), "weight": a.weight, "alpha": a.alpha} for a in adapters]
        ),
        "merge_note": (
            "Input LoRA alpha/rank scales, raw deltas, and user weights are baked into lora_B; runtime scale is 1"
        ),
    }
    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    for tensor in tensors:
        size = math.prod(tensor.shape) * DTYPE_SIZE[dtype]
        header[tensor.name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw_header += b" " * ((-len(raw_header)) % 8)

    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(struct.pack("<Q", len(raw_header)))
            handle.write(raw_header)
            for tensor in tensors:
                contributors = [a for a in adapters if tensor.module in a.modules and a.weight != 0]
                if tensor.kind == "A":
                    merged = np.concatenate(
                        [module_arrays(adapter, tensor.module)[0] for adapter in contributors],
                        axis=0,
                    )
                else:
                    parts = []
                    for adapter in contributors:
                        weights = adapter.modules[tensor.module]
                        _, b_matrix = module_arrays(adapter, tensor.module)
                        rank, _, _ = module_shape(adapter, tensor.module)
                        # If alpha was absent and ranks vary per module, PEFT's default is alpha=rank.
                        effective_alpha = adapter.alpha
                        if weights.is_diff:
                            # A raw delta already is the full weight update; it has no alpha/rank scale.
                            scale = adapter.weight
                        elif adapter.rank_hint is None and not any(
                            key in adapter.file.metadata
                            for key in ("ss_network_alpha", "lora_alpha", "network_alpha")
                        ):
                            effective_alpha = float(rank)
                            scale = adapter.weight * effective_alpha / rank
                        else:
                            scale = adapter.weight * effective_alpha / rank
                        parts.append(b_matrix * scale)
                    merged = np.concatenate(parts, axis=1)
                handle.write(encode(merged, dtype))
        os.replace(temp, path)
    except BaseException:
        if temp.exists():
            temp.unlink()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge weighted Krea 2 LoRAs of equal or different ranks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-i",
        "--input",
        action="append",
        nargs=2,
        metavar=("LORA", "WEIGHT"),
        help="input .safetensors and its weight; repeat for every LoRA",
    )
    source.add_argument(
        "-p",
        "--prompt",
        help='WebUI text containing tags such as "<lora:style:0.7> <lora:lines:-0.2>"',
    )
    parser.add_argument(
        "--lora-dir",
        action="append",
        type=Path,
        help="directory recursively searched by --prompt; repeat for multiple directories (default: current directory)",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="output .safetensors path")
    parser.add_argument(
        "--format", choices=("comfy", "diffusers"), default="comfy", help="output tensor naming convention"
    )
    parser.add_argument(
        "--dtype", choices=("auto", "bf16", "f16", "f32"), default="auto", help="output tensor dtype"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    return parser


def _open_files(paths: list[Path]) -> Iterator[list[SafeTensorFile]]:
    files: list[SafeTensorFile] = []
    try:
        for path in paths:
            file = SafeTensorFile(path)
            file.__enter__()
            files.append(file)
        yield files
    finally:
        for file in reversed(files):
            file.__exit__(None, None, None)


def run(args: argparse.Namespace) -> None:
    inputs: list[tuple[Path, float]] = []
    raw_inputs = getattr(args, "input", None)
    prompt = getattr(args, "prompt", None)
    if prompt is not None:
        directories = getattr(args, "lora_dir", None) or [Path.cwd()]
        resolved = resolve_prompt_inputs(prompt, directories)
        raw_inputs = [[str(path), str(weight)] for path, weight in resolved]
        print(f"Resolved {len(raw_inputs)} LoRA file(s) from WebUI prompt tags:")
        for path, weight in resolved:
            print(f"  {path.name}: weight={weight:g}")
    for raw_path, raw_weight in raw_inputs:
        path = Path(raw_path)
        if not path.is_file():
            raise MergeError(f"Input does not exist or is not a file: {path}")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise MergeError(f"Invalid weight for {path}: {raw_weight!r}") from exc
        if not math.isfinite(weight):
            raise MergeError(f"Weight must be finite for {path}")
        inputs.append((path, weight))
    if all(weight == 0 for _, weight in inputs):
        raise MergeError("At least one input weight must be non-zero")
    if args.output.exists() and not args.overwrite:
        raise MergeError(f"Output already exists: {args.output} (use --overwrite to replace it)")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    context = _open_files([path for path, _ in inputs])
    files = next(context)
    try:
        adapters = [inspect_adapter(file, weight) for file, (_, weight) in zip(files, inputs)]
        dtype = choose_dtype(adapters, args.dtype)
        tensors = output_tensors(adapters, args.format)
        print(f"Merging {len(adapters)} Krea 2 LoRAs into {args.output}")
        for adapter in adapters:
            ranks = sorted({module_shape(adapter, module)[0] for module in adapter.modules})
            kinds = sorted({"diff" if weights.is_diff else "lora" for weights in adapter.modules.values()})
            print(
                f"  {adapter.file.path.name}: weight={adapter.weight:g}, alpha={adapter.alpha:g}, "
                f"rank(s)={ranks}, modules={len(adapter.modules)}, kind={'+'.join(kinds)}, "
                f"format={adapter.source_format}"
            )
        print(f"  output: format={args.format}, dtype={dtype}, modules={len(tensors) // 2}")
        write_safetensors(args.output, tensors, adapters, dtype, args.format)
    finally:
        try:
            next(context)
        except StopIteration:
            pass
    print(f"Wrote {args.output} ({args.output.stat().st_size / (1024**2):.1f} MiB)")


def main() -> None:
    parser = build_parser()
    try:
        run(parser.parse_args())
    except MergeError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        print("Interrupted; no partial output was kept.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
