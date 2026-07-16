# Krea 2 LoRA merger

Merge any number of Krea 2 LoRAs with a separate weight for each input. Mixed
rank and mixed native/Comfy + Diffusers naming are supported.

## Setup and usage

```powershell
uv sync

uv run python merge_krea2_loras.py `
  --input "first.safetensors" 0.7 `
  --input "second.safetensors" 0.35 `
  --output "merged.safetensors"
```

The default output uses native/Comfy Krea 2 names. For a Diffusers-format file:

```powershell
uv run python merge_krea2_loras.py `
  -i "first.safetensors" 1.0 `
  -i "second.safetensors" -0.2 `
  -o "merged-diffusers.safetensors" `
  --format diffusers
```

Run `uv run python merge_krea2_loras.py --help` for dtype and overwrite options.

## Stable Diffusion WebUI prompt tags

The script can resolve LoRAs from WebUI-style prompt text:

```powershell
uv run python merge_krea2_loras.py `
  --prompt "<lora:coolstyle:3> <lora:ugly_sketch:-2> <lora:line_art:2>" `
  --lora-dir "D:\path\to\loras" `
  --output "merged.safetensors"
```

`--lora-dir` is searched recursively and may be repeated. If omitted, the
current directory is searched. Names are matched case-insensitively against
`modelspec.title`, `title`, `model_name`, `lora_name`, `ss_output_name`, `name`,
and `repoId` metadata. The filename stem is also indexed as a fallback. An
ambiguous or missing name is reported instead of silently selecting a file.
Repeated tags resolving to the same file have their weights added together.
Krea 2 raw `.diff` delta adapters (including projector-only bypass adapters) are
converted into exact LoRA factors automatically, so they can be mixed with
ordinary LoRAs in the same prompt.

## How it merges

For each adapted layer, LoRA represents a change `alpha/rank * B @ A`. The
script concatenates all A matrices vertically and all B matrices horizontally.
Each requested weight and each input's `alpha/rank` are baked into its B block,
so the merged file exactly represents the weighted sum without SVD loss. A layer
that exists in only some inputs is retained. Different source and per-layer ranks
are valid; the output rank for a layer is their sum.

Only tensor headers and one output tensor at a time are held in memory. The base
Krea 2 checkpoint is not needed.
