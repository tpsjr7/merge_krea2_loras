import argparse
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from merge_krea2_loras import SafeTensorFile, encode, parse_lora_prompt, resolve_prompt_inputs, run


def write_fixture(path: Path, tensors: dict[str, tuple[np.ndarray, str]], metadata: dict[str, str]) -> None:
    header = {"__metadata__": metadata}
    payloads = []
    offset = 0
    for name, (array, dtype) in tensors.items():
        payload = encode(array, dtype)
        payloads.append(payload)
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        for payload in payloads:
            handle.write(payload)


class MergeTests(unittest.TestCase):
    def test_raw_diff_is_merged_exactly_with_lora(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora, raw_diff, output = root / "style.safetensors", root / "bypass.safetensors", root / "out.safetensors"
            a = np.array([[1, 2, 3]], dtype=np.float32)
            b = np.array([[2]], dtype=np.float32)
            delta = np.array([[4, -1, 0.5]], dtype=np.float32)
            write_fixture(
                lora,
                {
                    "diffusion_model.txtfusion.projector.lora_A.weight": (a, "F32"),
                    "diffusion_model.txtfusion.projector.lora_B.weight": (b, "F32"),
                },
                {},
            )
            write_fixture(
                raw_diff,
                {"diffusion_model.txtfusion.projector.diff": (delta, "F32")},
                {"name": "fedor_bypass"},
            )
            run(
                argparse.Namespace(
                    input=[[str(lora), "0.25"], [str(raw_diff), "3"]],
                    output=output,
                    format="comfy",
                    dtype="f32",
                    overwrite=False,
                )
            )
            with SafeTensorFile(output) as merged:
                base = "diffusion_model.txtfusion.projector"
                actual = merged.array(base + ".lora_B.weight") @ merged.array(base + ".lora_A.weight")
                expected = 0.25 * (b @ a) + 3.0 * delta
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

    def test_webui_prompt_metadata_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cool = root / "unhelpful-download-name.safetensors"
            sketch = root / "ugly_sketch.metadata.safetensors"
            tensors = {
                "diffusion_model.blocks.0.attn.wq.lora_A.weight": (
                    np.ones((1, 2), dtype=np.float32), "F32"
                ),
                "diffusion_model.blocks.0.attn.wq.lora_B.weight": (
                    np.ones((2, 1), dtype=np.float32), "F32"
                ),
            }
            write_fixture(cool, tensors, {"repoId": "creator/coolstyle"})
            write_fixture(sketch, tensors, {})
            prompt = "portrait <lora:COOLSTYLE:3> <lora:ugly_sketch:-2> <lora:coolstyle:.5>"
            resolved = dict(resolve_prompt_inputs(prompt, [root]))
            self.assertEqual(resolved[cool.resolve()], 3.5)
            self.assertEqual(resolved[sketch.resolve()], -2.0)

    def test_rejects_malformed_webui_tag(self) -> None:
        with self.assertRaisesRegex(Exception, "malformed"):
            parse_lora_prompt("<lora:good:1> <lora:bad:not-a-number>")

    def test_mixed_format_rank_dtype_and_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, output = root / "first.safetensors", root / "second.safetensors", root / "out.safetensors"
            a1 = np.array([[1, 2, 0], [0, 1, 1]], dtype=np.float32)
            b1 = np.array([[1, 0], [0, 2]], dtype=np.float32)
            a2 = np.array([[2, 0, 1]], dtype=np.float32)
            b2 = np.array([[1], [3]], dtype=np.float32)
            write_fixture(
                first,
                {
                    "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": (a1, "F16"),
                    "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": (b1, "F16"),
                },
                {"lora_rank": "2"},
            )
            write_fixture(
                second,
                {
                    "diffusion_model.blocks.0.attn.wq.lora_A.weight": (a2, "BF16"),
                    "diffusion_model.blocks.0.attn.wq.lora_B.weight": (b2, "BF16"),
                },
                {"ss_network_dim": "1", "ss_network_alpha": "0.5"},
            )
            args = argparse.Namespace(
                input=[[str(first), "0.5"], [str(second), "-2"]],
                output=output,
                format="comfy",
                dtype="f32",
                overwrite=False,
            )
            run(args)
            with SafeTensorFile(output) as merged:
                base = "diffusion_model.blocks.0.attn.wq"
                actual = merged.array(base + ".lora_B.weight") @ merged.array(base + ".lora_A.weight")
                expected = 0.5 * (b1 @ a1) - 1.0 * (b2 @ a2)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
                self.assertEqual(merged.tensors[base + ".lora_A.weight"].shape, (3, 3))
                self.assertEqual(merged.metadata["modelspec.architecture"], "krea2/lora")

    def test_diffusers_output_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "in.safetensors", root / "out.safetensors"
            write_fixture(
                source,
                {
                    "diffusion_model.txtfusion.refiner_blocks.1.attn.wo.lora_A.weight": (
                        np.ones((1, 2), dtype=np.float32),
                        "F32",
                    ),
                    "diffusion_model.txtfusion.refiner_blocks.1.attn.wo.lora_B.weight": (
                        np.ones((2, 1), dtype=np.float32),
                        "F32",
                    ),
                },
                {},
            )
            run(
                argparse.Namespace(
                    input=[[str(source), "1"]],
                    output=output,
                    format="diffusers",
                    dtype="auto",
                    overwrite=False,
                )
            )
            with SafeTensorFile(output) as merged:
                self.assertIn(
                    "transformer.text_fusion.refiner_blocks.1.attn.to_out.0.lora_A.weight",
                    merged.tensors,
                )


if __name__ == "__main__":
    unittest.main()
