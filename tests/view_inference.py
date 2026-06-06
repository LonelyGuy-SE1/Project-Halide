"""Local Gradio viewer for the Halide inference pipeline.

No models loaded on local CPU. The viewer uploads the chosen image to a
Modal volume, then invokes a Modal T4 function via `modal run`. Open
http://127.0.0.1:7860 in your browser.

Layout:
  [ Image picker (3 preloaded + upload) ]
  [ Run both ]  [ Run base only ]  [ Run finetuned only ]
  [ Input image ]  [ Base panel  ]  [ Finetuned panel ]
  [ Expected (GT, if in-dist)        ]
  [ Status / timing                  ]

Run:
  python tests/view_inference.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# CRITICAL: matplotlib and Gradio queueing need $HOME. Background processes
# (nohup, detached) don't inherit it on Windows. Set it BEFORE importing
# anything that might trigger matplotlib import.
if not os.environ.get("HOME"):
    os.environ["HOME"] = str(Path.home() if Path.home().exists() else Path("C:/Users/Admin"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("USERPROFILE", "C:/Users/Admin")

import gradio as gr

REPO = Path(__file__).resolve().parents[1]
TEST_IMAGES_DIR = REPO / "data" / "test_images"
TRAINING_DATA = REPO / "data" / "training_data.jsonl"
MODAL_SCRIPT = "modal/view_inference.py"
LOG_FILE = REPO / "data" / "test_results" / "viewer_debug.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    """Append a timestamped debug line to viewer_debug.log."""
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass
VOLUME_NAME = "halide-viewer-uploads"

PRELOADED = [
    ("01_indist_scan1 (in training)", "01_indist_scan1.png"),
    ("02_ood_synthetic_scratch", "02_ood_synthetic_scratch.png"),
    ("03_ood_blueneg", "03_ood_blueneg.png"),
]


def _load_ground_truth() -> dict[str, dict]:
    gt: dict[str, dict] = {}
    if not TRAINING_DATA.exists():
        return gt
    with TRAINING_DATA.open() as f:
        for line in f:
            row = json.loads(line)
            counts: dict[str, int] = {}
            for a in row.get("annotations", []):
                counts[a["label"]] = counts.get(a["label"], 0) + 1
            gt[row["image"]] = {
                "label_counts": counts,
                "total": len(row.get("annotations", [])),
                "width": row.get("width"),
                "height": row.get("height"),
            }
    return gt


def _subprocess_env() -> dict:
    """Build a subprocess env with HOME and UTF-8 set, so the modal CLI doesn't blow up."""
    e = os.environ.copy()
    e["PYTHONIOENCODING"] = "utf-8"
    e["PYTHONUTF8"] = "1"
    e["HOME"] = e.get("HOME") or e.get("USERPROFILE") or "C:/Users/Admin"
    e["USERPROFILE"] = e.get("USERPROFILE") or e["HOME"]
    e["MPLBACKEND"] = "Agg"
    return e


def _upload_to_modal_volume(local_pil, tag: str) -> tuple[str, str]:
    """Write the image to a temp file, push to Modal volume, return (remote_path, status)."""
    from PIL import Image
    import io

    if isinstance(local_pil, str):
        pil = Image.open(local_pil)
    else:
        pil = local_pil
    remote_name = f"viewer_{tag}_{int(time.time())}.png"
    tmp = REPO / "data" / "test_results" / "_upload_tmp.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    pil.convert("RGB").save(tmp, "PNG")
    # modal volume put src dst
    _log(f"upload: tmp={tmp} ({tmp.stat().st_size}B), remote={remote_name}")
    proc = subprocess.run(
        ["modal", "volume", "put", VOLUME_NAME, str(tmp), remote_name],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_subprocess_env(),
        cwd=str(REPO),
    )
    _log(f"upload: rc={proc.returncode}, stderr_tail={proc.stderr[-200:]!r}")
    if proc.returncode != 0:
        return ("", f"upload failed (rc={proc.returncode}): {proc.stderr[-500:]}")
    return (remote_name, f"uploaded as {remote_name} ({tmp.stat().st_size} bytes)")


def _call_modal(which: str, image_path: str) -> dict:
    """Invoke the Modal local_entrypoint `main` with --which base|finetuned|both.

    Returns dict with `result` or `error`.
    """
    cmd = ["modal", "run", f"{MODAL_SCRIPT}::main", "--image-path", image_path, "--which", which]
    _log(f"call_modal: which={which}, image_path={image_path}, cmd={cmd}")
    t0 = time.time()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_subprocess_env(),
        cwd=str(REPO),
    )
    elapsed = round(time.time() - t0, 2)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    _log(f"call_modal: rc={proc.returncode}, elapsed={elapsed}s, stdout_len={len(stdout)}, stderr_len={len(stderr)}")
    _log(f"call_modal: stdout_tail={stdout[-400:]!r}")
    _log(f"call_modal: stderr_tail={stderr[-400:]!r}")
    if proc.returncode != 0:
        return {"error": f"modal run failed (rc={proc.returncode})", "stderr": stderr[-1500:], "modal_elapsed": elapsed}
    start = stdout.find("===RESULT_START===")
    end = stdout.find("===RESULT_END===")
    if start < 0 or end < 0:
        return {"error": "no result markers in modal stdout", "stdout_tail": stdout[-500:], "modal_elapsed": elapsed}
    payload = stdout[start + len("===RESULT_START==="):end].strip()
    try:
        return {"result": json.loads(payload), "modal_elapsed": elapsed, "stdout_tail": stdout[-200:]}
    except json.JSONDecodeError as exc:
        return {"error": f"JSON parse: {exc}", "stdout": payload[:500], "modal_elapsed": elapsed}


def _format_status(label: str, model_id: str | None, info: dict) -> str:
    if not info:
        return f"### {label}\n_(not run)_"
    if "error" in info:
        return f"### {label}\n**ERROR:** {info['error']}\n\n```\n{info.get('stderr', '')[-800:]}\n```"
    return (
        f"### {label}\n"
        f"- model: `{model_id or info.get('model_id', '?')}`\n"
        f"- device: `{info.get('device', '?')}`\n"
        f"- load: {info.get('load_seconds', '?')}s\n"
        f"- inference: {info.get('inference_seconds', '?')}s"
    )


def _format_gt(filename: str, gt_map: dict[str, dict]) -> str:
    if not filename or filename not in gt_map:
        return "_(no ground truth available for this image)_"
    g = gt_map[filename]
    counts = g.get("label_counts", {})
    sorted_counts = dict(sorted(counts.items(), key=lambda x: -x[1]))
    return (
        f"**Image:** `{filename}`  \n"
        f"**Dimensions:** {g.get('width')} x {g.get('height')}  \n"
        f"**Total defects:** {g.get('total')}  \n"
        f"**Label breakdown:**\n```json\n{json.dumps(sorted_counts, indent=2)}\n```"
    )


def _resolve(picker_value, uploaded):
    """Return (pil, filename_for_gt_lookup, display_label)."""
    if uploaded is not None:
        return uploaded, None, "(uploaded)"
    if picker_value:
        for label, fname in PRELOADED:
            if label == picker_value:
                from PIL import Image
                return Image.open(TEST_IMAGES_DIR / fname), fname, label
    return None, None, "(none)"


def _make_runner(gt_map):
    """Build the 8-output list in one place. The output order is:
    [overall, base_status, base_raw, base_parsed, ft_status, ft_raw, ft_parsed, gt_panel]
    and is shared with the click() handler and the error builders below.
    """
    # Single source of truth for the output schema
    OUT = ["overall", "base_status", "base_raw", "base_parsed", "ft_status", "ft_raw", "ft_parsed", "gt_panel"]

    def _result(d: dict) -> list:
        """Build an 8-element list from a dict keyed by OUT names. Missing keys -> ''."""
        return [d.get(k, "") for k in OUT]

    def run(which: str, picker, uploaded, progress=gr.Progress(track_tqdm=False)):
        _log(f"=== run start: which={which}, picker={picker}, uploaded={'yes' if uploaded is not None else 'no'}")
        try:
            pil, fname, display = _resolve(picker, uploaded)
            if pil is None:
                _log("run: no image resolved")
                return _result({
                    "overall": "_(no image)_",
                    "base_status": "_(no image)_",
                    "gt_panel": "_(no image)_",
                })

            progress(0.1, desc="Uploading image to Modal volume...")
            remote, upload_status = _upload_to_modal_volume(pil, which)
            if not remote:
                _log(f"run: upload failed: {upload_status}")
                return _result({
                    "overall": f"**Upload failed:** {upload_status}",
                    "base_status": f"**Upload failed:** {upload_status}",
                })

            progress(0.3, desc=f"Calling Modal (T4 cold start ~30-60s, which={which})...")
            t0 = time.time()
            data = _call_modal(which, remote)
            wall = round(time.time() - t0, 2)
            _log(f"run: modal call done, wall={wall}s, has_result={'result' in data}, error={data.get('error')}")

            if "result" not in data:
                err = data.get("error", "?")
                stderr = data.get("stderr", "")[-400:] if data.get("stderr") else ""
                return _result({
                    "overall": f"**Modal call failed** (wall {wall}s): {err}",
                    "base_status": f"**Modal call failed:** {err}\n\n```\n{stderr}\n```",
                })

            r = data["result"]
            if which == "both":
                base_info = r.get("results", {}).get("base", {})
                ft_info = r.get("results", {}).get("finetuned", {})
            elif which == "base":
                base_info = r
                ft_info = {}
            else:
                base_info = {}
                ft_info = r

            base_raw = base_info.get("raw_output", "") if base_info else ""
            ft_raw = ft_info.get("raw_output", "") if ft_info else ""
            _log(f"run: success, base_chars={len(base_raw)}, ft_chars={len(ft_raw)}")
            return _result({
                "overall": f"### Run complete\n- input: {display}\n- remote_path: `{remote}`\n- {upload_status}\n- modal wall: {wall}s\n- modal_elapsed: {data.get('modal_elapsed')}s",
                "base_status": _format_status("Base (`openbmb/MiniCPM-V-4_6`)", None, base_info),
                "base_raw": base_raw,
                "base_parsed": json.dumps(base_info.get("parsed_json", {}), indent=2) if base_info else "{}",
                "ft_status": _format_status("Finetuned (`Lonelyguyse1/halide-vision`)", None, ft_info),
                "ft_raw": ft_raw,
                "ft_parsed": json.dumps(ft_info.get("parsed_json", {}), indent=2) if ft_info else "{}",
                "gt_panel": _format_gt(fname, gt_map),
            })
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            _log(f"run: EXCEPTION: {tb}")
            return _result({
                "overall": f"**Exception:** {type(exc).__name__}: {exc}",
                "base_status": f"**Exception:** {type(exc).__name__}: {exc}\n\n```\n{tb}\n```",
            })

    return run


def main():
    gt_map = _load_ground_truth()
    preloaded_choices = [label for label, _ in PRELOADED]
    runner = _make_runner(gt_map)

    with gr.Blocks(title="Halide Inference Viewer") as demo:
        gr.Markdown(
            "# Halide Inference Viewer\n"
            "Local CPU does no inference. All models run on Modal T4. "
            "First call may take ~60s (cold start); subsequent calls ~10s. "
            "Cost ~$0.01 per test."
        )

        with gr.Row():
            picker = gr.Dropdown(choices=preloaded_choices, value=preloaded_choices[0], label="Preloaded image")
            upload = gr.Image(type="pil", label="...or upload your own", height=200)

        with gr.Row():
            btn_both = gr.Button("Run BOTH (cheaper, one T4 call)", variant="primary")
            btn_base = gr.Button("Run BASE only")
            btn_ft = gr.Button("Run FINETUNED only")

        overall = gr.Markdown(value="_results will appear here_")

        with gr.Row():
            base_status = gr.Markdown(value="_base not run_")
            ft_status = gr.Markdown(value="_finetuned not run_")
        with gr.Row():
            base_raw = gr.Textbox(label="Base raw output", lines=10)
            ft_raw = gr.Textbox(label="Finetuned raw output", lines=10)
        with gr.Row():
            base_parsed = gr.Code(label="Base parsed JSON", language="json")
            ft_parsed = gr.Code(label="Finetuned parsed JSON", language="json")

        gt_panel = gr.Markdown(value="_ground truth will appear here_")

        outputs = [overall, base_status, base_raw, base_parsed, ft_status, ft_raw, ft_parsed, gt_panel]
        btn_both.click(lambda p, u: runner("both", p, u), inputs=[picker, upload], outputs=outputs)
        btn_base.click(lambda p, u: runner("base", p, u), inputs=[picker, upload], outputs=outputs)
        btn_ft.click(lambda p, u: runner("finetuned", p, u), inputs=[picker, upload], outputs=outputs)

    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
