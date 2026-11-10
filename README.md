# GlyphType

GlyphType captures a selected rectangle of a KDE Plasma Wayland desktop, recognizes its text locally with Tesseract or an optional ONNX line model, and can type the result into an explicitly selected window. The `type` command handles one static passage; `watch` tracks repeated captures of the same configured region. Native X11 remains available as a fallback.

Use the typing feature only in local or explicitly authorized environments. Public competitive leaderboard automation is outside this project.

## Install

Requirements: Linux with KDE Plasma on Wayland, Python 3.11+, Tesseract with language data, Spectacle for drag selection, and the KDE RemoteDesktop portal for live typing. On Arch Linux, install `tesseract`, `tesseract-data-eng`, `spectacle`, `xdg-desktop-portal`, and `xdg-desktop-portal-kde`. Native X11 instead needs `slop` and `xdotool` for selection and input.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
cp config.example.toml config.toml
.venv/bin/glyphtype doctor
```

If pip is absent, run `.venv/bin/python -m ensurepip --upgrade` first. The program itself runs offline. Check OCR languages with `tesseract --list-langs`. If English data is absent from the system, GlyphType automatically uses `models/tessdata/eng.traineddata` when that local file exists. You can instead set `[ocr] tessdata_dir` to a directory containing the requested `.traineddata` files. A path relative to `config.toml` is supported.

## Configure

`[display] backend = "auto"` selects native KDE Wayland or X11 from the active session. Set `[capture]` to the exact rectangle for repeated capture. Set `[ocr] language` to an installed Tesseract language. `[preprocessing]` defaults to the unmodified image; compare changes with a labeled screenshot before relying on them. `[typing] dry_run` defaults to `true`; set `[typing] character_delay_ms` to control speed. Live typing checks the identity of the selected target window before each character.

`[tracking]` controls the interval, minimum Tesseract word confidence (0–100), and number of identical observations needed before sending new text. Watch mode runs Tesseract twice per frame to obtain text and confidence, so its OCR latency is higher than one-shot recognition.

The `[capture]` coordinates specify a fixed rectangle in virtual desktop pixels, including negative coordinates on some multimonitor layouts. KWin's exact-area screenshot interface can be restricted on some KDE installations; `doctor` checks a single pixel and reports a denial. The program does not fall back to whole-screen capture.

## Select, read, and type

Drag a rectangle around the displayed text and preview the OCR output without keyboard input:

```bash
.venv/bin/glyphtype type --select --config config.toml --start
```

In Spectacle's rectangle selector, press **Enter** or double-click to finish the selection. You do not need to press **Save**: GlyphType passes Spectacle a temporary output path and reads the cropped image automatically.

You can also save the crop for inspection or use configured coordinates:

```bash
.venv/bin/glyphtype capture --select --output captures/text.png
.venv/bin/glyphtype type --capture --config config.toml --start
.venv/bin/glyphtype ocr --image captures/text.png --config config.toml
```

Create `captures/` first. `capture` refuses to overwrite an existing image. To measure OCR accuracy, use `ocr --expected-file expected.txt`; `--save-mismatch path.png` optionally copies an image when OCR differs from the reference.

For live KDE Wayland typing, set `typing.dry_run = false` in `config.toml`. Start the command, approve the desktop's keyboard permission request, then click the intended application when the crosshair appears. Refocus its input during the countdown:

```bash
.venv/bin/glyphtype type --select --config config.toml --start --live
```

`--image path.png` can replace `--select` to type a saved OCR image. `--expected-file expected.txt` cancels live typing if OCR differs from its checked reference. Live input requires `--start`, `--live`, and `dry_run = false`. On Wayland it also requires portal approval and a clicked target window. KWin reports the active window's identity; typing stops if it loses focus. Press **Meta+Shift+F12** to stop globally. `--focus-wait 0` skips the refocus delay. The portal sends keysyms to the focused window, so inspect its contents if a delivery error occurs.

Typing treats OCR line wraps as spaces: `type` and `watch` join consecutive line breaks into one space and trim surrounding whitespace. The `ocr` command still prints the raw recognized layout. Preview the `type` dry run before live input to check the resulting words and spacing.

On native X11, live typing still uses `xdotool`; pass `--target-window 12345678` after obtaining the ID with `xdotool getactivewindow`. Escape is its emergency stop key. Generic Wayland compositors are not supported; the current native implementation uses KWin, Spectacle, and the RemoteDesktop portal. Live portal key delivery has not yet been manually verified in this session.

## Track changing text

For KDE Wayland watch mode, set fixed `[capture]` coordinates and start with a short dry run. Spectacle can select a one-shot crop, but does not provide reusable coordinates to `watch`:

```bash
.venv/bin/glyphtype watch --config config.toml --start --max-frames 5
```

Omit `--max-frames` to watch until stopped. The dry run prints each confirmed suffix it *would* type. For live KDE Wayland use, set `typing.dry_run = false`, then add `--live`; approve the portal permission and click the authorized target window. On X11, `watch --select` and `--target-window` remain available.

Repeated identical screenshots do not resend characters. A suffix is sent only after the configured number of identical, high-confidence observations. A changed or uncertain passage pauses the watcher. In live Wayland mode, review the logged OCR text and press **Meta+Shift+F8** to retry or resume, or **Meta+Shift+F9** to explicitly accept a new high-confidence passage. On X11 use F8/F9. Accepting a new passage resets progress. Dry-run mode exits on pause so you can inspect the output and restart deliberately.

Progress is kept in memory for this run. It counts successful input-backend calls; neither display backend confirms that the application inserted each character. If delivery is uncertain, inspect the target before resuming. Restarting the command starts a new passage and does not restore the previous character count.

## Manual validation

1. Run the dry preview and inspect the text, including capitalization, spaces, punctuation, and spaces where the image wraps lines.
2. On KDE Wayland, drag a region with Spectacle and check the saved crop contains only that region. Check a configured capture separately.
3. In an authorized input window, verify live typing, Meta+Shift+F12 stop, capitalization, punctuation, and focus-loss stop. Confirm no input reaches a newly focused window.
4. Run `watch` on a stable passage, then show an exact appended suffix and a different passage. Confirm no duplicates, and confirm the changed passage pauses until manually accepted.

This development session is KDE Wayland. Exact-area capture, KWin focus reporting, shortcut registration, and the portal session and keyboard-device request were verified here. The portal approval dialog, actual key delivery, Spectacle drag selection, and target-window handoff still require manual validation.

## Generate synthetic OCR data (Phase 5)

Generate 10,000 labeled **single-line** images and 500 optional two-line images for line-spacing experiments. The two-line images have separate manifests and are not loaded by the future line-model dataset loader:

```bash
.venv/bin/glyphtype-generate-dataset --output datasets/synthetic --count 10000 --multiline-count 500 --seed 42
```

Use repeated `--font /path/to/font.ttf` options to pin fonts for a reproducible run, or let the generator find several local font families through fontconfig. `--corpus words.txt` uses a local UTF-8 word source instead of the built-in vocabulary. The generator varies font size, character spacing, capitalization, punctuation, digits, contrast, blur, and noise. Two-line samples also vary line spacing. It refuses to overwrite an existing output directory.

The output contains `train.jsonl`, `validation.jsonl`, `test.jsonl`, corresponding PNG folders, and `metadata.json` with the seed, font paths, Pillow version, and split rule. Each JSONL record contains the exact rendered text and image path. Identical labels are assigned to one split by a stable hash, so the same text cannot appear in both training and test manifests. Images are opened only when indexed:

```python
from pathlib import Path
from glyphtype.dataset import LineDataset

dataset = LineDataset(Path("datasets/synthetic"), "train")
image, label = dataset[0]
```

Keep real screenshots and their manually checked labels in a separate validation set. Synthetic accuracy does not establish accuracy on the target desktop. `datasets/` and `models/` are ignored by Git; do not commit generated images or checkpoints. The commands below train and compare a compact line-recognition model against Tesseract.

## Train and compare the optional line model (Phase 6)

The custom recognizer uses a small CNN, a bidirectional LSTM, and CTC. Training defaults to CPU, batch size 8, and a 32-pixel image height. For a CPU-only PyTorch install, install its CPU wheel first, then the optional training dependencies:

```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[train]'
.venv/bin/glyphtype-train --dataset datasets/synthetic --output models/line_ocr --epochs 50 --batch-size 8 --device cpu
```

Training writes `latest.pt`, `best.pt`, and `history.jsonl`. Validation loss is logged each epoch; training stops after five epochs without improvement by default. Resume with the same data and model settings using `--resume models/line_ocr/latest.pt --epochs 60`. Use `--device cuda` only if CUDA is available; mixed precision is enabled only for CUDA. Check the history for process RAM and optional GPU memory peaks.

Export the best checkpoint and compare ONNX Runtime with Tesseract on **the same** labeled test manifest:

```bash
.venv/bin/glyphtype-export-onnx --checkpoint models/line_ocr/best.pt --output models/line_ocr/model.onnx
.venv/bin/glyphtype-evaluate --dataset datasets/synthetic --model models/line_ocr/model.onnx --language eng
```

Export checks ONNX structure and PyTorch/ONNX numerical agreement at two image widths. The evaluation report contains CER, WER, exact-line accuracy, mean and 95th-percentile latency, and peak process/child memory. Run the same evaluation against a separately labeled real-screenshot `test.jsonl` before trusting the model on desktop text. Keep Tesseract as the default while its measured accuracy is better.

ONNX inference needs `onnxruntime`, NumPy, Pillow, the `.onnx` file, and its adjacent `.json` metadata; it does **not** import PyTorch. To opt into an exported model for one-shot `ocr` or `type`, set `[ocr] backend = "onnx"` and `model_path = "models/line_ocr/model.onnx"` in `config.toml`. `watch` currently requires Tesseract because the ONNX model does not provide calibrated confidence for safe automatic tracking.

The current local pilot used 1,000 generated line images (800 train, 100 validation, 100 test), CPU batch size 8, and 50 epochs. On that **synthetic** test split, its ONNX CER was 0.070 and exact-line accuracy 18%; English Tesseract scored CER 0.127 and exact-line accuracy 13%. Mean inference latency was 2.5 ms for ONNX versus 129 ms for Tesseract. The pilot files are ignored under `datasets/phase6_pilot/` and `models/phase6_pilot/`. These figures do not establish accuracy on real screenshots, so Tesseract remains the default backend.
