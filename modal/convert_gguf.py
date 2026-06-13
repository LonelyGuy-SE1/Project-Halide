"""Convert the merged MiniCPM-V checkpoint to GGUF on Modal.

This is for the llama.cpp milestone path. It performs conversion in Modal, not
on the local machine, and stores artifacts in the checkpoint volume.
"""

from __future__ import annotations

import modal

app = modal.App("halide-gguf-convert")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "build-essential")
    .run_commands(
        "cd /root && git clone --depth 1 https://github.com/ggerganov/llama.cpp",
        "cd /root/llama.cpp && pip install -r requirements/requirements-convert_hf_to_gguf.txt",
        "cd /root/llama.cpp && cmake -B build && cmake --build build --config Release --target llama-quantize -j4",
    )
    .pip_install("transformers[torch]==5.7.0", "huggingface_hub>=0.20.0")
)

checkpoint_volume = modal.Volume.from_name("halide-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=2 * 3600,
)
def convert_to_gguf(
    model_dir: str = "/checkpoints/minicpm-v-4.6-merged-v4-stage1",
    outfile: str = "/checkpoints/minicpm-v-4.6-merged-v4-stage1-f16.gguf",
    quantized_outfile: str = "/checkpoints/minicpm-v-4.6-merged-v4-stage1-q4_k_m.gguf",
) -> dict:
    import json
    import shutil
    import subprocess
    from pathlib import Path

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

    work_model_path = Path("/tmp/halide-gguf-model")
    if work_model_path.exists():
        shutil.rmtree(work_model_path)
    shutil.copytree(model_path, work_model_path)

    tokenizer_config = work_model_path / "tokenizer_config.json"
    if tokenizer_config.exists():
        config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        config.pop("tokenizer_class", None)
        config.pop("auto_map", None)
        tokenizer_config.write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )

    convert_cmd = [
        "python",
        "/root/llama.cpp/convert_hf_to_gguf.py",
        str(work_model_path),
        "--outfile",
        outfile,
        "--outtype",
        "f16",
    ]
    convert = subprocess.run(
        convert_cmd,
        cwd="/root/llama.cpp",
        text=True,
        capture_output=True,
    )
    if convert.returncode != 0:
        return {
            "status": "convert_failed",
            "returncode": convert.returncode,
            "stdout": convert.stdout[-4000:],
            "stderr": convert.stderr[-4000:],
        }

    quantize_cmd = [
        "/root/llama.cpp/build/bin/llama-quantize",
        outfile,
        quantized_outfile,
        "Q4_K_M",
    ]
    quantize = subprocess.run(
        quantize_cmd,
        cwd="/root/llama.cpp",
        text=True,
        capture_output=True,
    )
    checkpoint_volume.commit()
    if quantize.returncode != 0:
        return {
            "status": "quantize_failed",
            "returncode": quantize.returncode,
            "outfile": outfile,
            "stdout": quantize.stdout[-4000:],
            "stderr": quantize.stderr[-4000:],
        }

    return {
        "status": "ok",
        "outfile": outfile,
        "quantized_outfile": quantized_outfile,
        "stdout": quantize.stdout[-1000:],
    }


@app.local_entrypoint()
def main(
    model_dir: str = "/checkpoints/minicpm-v-4.6-merged-v4-stage1",
    outfile: str = "/checkpoints/minicpm-v-4.6-merged-v4-stage1-f16.gguf",
    quantized_outfile: str = "/checkpoints/minicpm-v-4.6-merged-v4-stage1-q4_k_m.gguf",
):
    result = convert_to_gguf.remote(
        model_dir=model_dir,
        outfile=outfile,
        quantized_outfile=quantized_outfile,
    )
    print(result)
