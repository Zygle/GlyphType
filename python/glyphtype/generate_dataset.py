"""Reproducible synthetic text images for a future line OCR model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, __version__ as pillow_version


DEFAULT_WORDS = (
    "about above after again along always answer around because before better bright browser"
    " capture change clear code color comes common could create custom different display"
    " during engine every example focus frame friend future image input inside keyboard"
    " language later letter line little local memory model moment moving number object"
    " output people phrase pixel place point practice present reader region repeat result"
    " screen second simple since small space speed stable system target testing their"
    " there these thing three through typing under until value watch where which window"
    " words world write yellow zebra"
).split()
SPLITS = ("train", "validation", "test")


def split_for_label(text: str) -> str:
    """Assign identical labels to the same split on every run."""
    bucket = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def discover_fonts(explicit: list[Path] | None) -> list[Path]:
    if explicit:
        fonts = sorted(set(path.expanduser().resolve() for path in explicit))
    else:
        fonts = []
        for family in ("Noto Sans", "Noto Serif", "Liberation Sans", "Liberation Mono", "DejaVu Sans"):
            try:
                result = subprocess.run(
                    ["fc-match", "-f", "%{file}", family], capture_output=True,
                    text=True, timeout=5, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                break
            if result.returncode == 0 and result.stdout.strip():
                fonts.append(Path(result.stdout.strip()).resolve())
        fonts = sorted(set(fonts))
    usable = []
    for path in fonts:
        try:
            ImageFont.truetype(str(path), 24)
        except OSError as exc:
            raise ValueError(f"Cannot load font {path}: {exc}") from exc
        usable.append(path)
    if not usable:
        raise ValueError("No usable fonts found; pass one or more --font paths")
    return usable


def load_words(corpus: Path | None) -> list[str]:
    if corpus is None:
        return list(DEFAULT_WORDS)
    text = corpus.read_text(encoding="utf-8")
    words = sorted(set(re.findall(r"[A-Za-z]+", text)))
    if len(words) < 10:
        raise ValueError("Corpus must contain at least 10 distinct alphabetic words")
    return words


def make_label(rng: random.Random, words: list[str], kind: str) -> str:
    if kind == "character":
        return rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:'-()")
    if kind == "word":
        word = rng.choice(words)
        return word.capitalize() if rng.random() < 0.35 else word
    count = rng.randint(4, 11)
    parts = [rng.choice(words) for _ in range(count)]
    parts[0] = parts[0].capitalize()
    if rng.random() < 0.4:
        parts[rng.randrange(1, count)] = str(rng.randint(0, 9999))
    if rng.random() < 0.35:
        parts[rng.randrange(1, count)] += rng.choice((",", ";", ":"))
    return " ".join(parts) + rng.choice((".", "!", "?"))


def choose_kind(rng: random.Random) -> str:
    value = rng.random()
    return "character" if value < 0.05 else "word" if value < 0.20 else "sentence"


def render_label(text: str, font_path: Path, rng: random.Random) -> tuple[Image.Image, dict]:
    """Render exact label text with controlled appearance variations."""
    font_size = rng.randint(18, 42)
    font = ImageFont.truetype(str(font_path), font_size)
    character_spacing = rng.randint(0, 3)
    line_spacing = rng.randint(4, 18)
    pad = rng.randint(8, 20)
    lines = text.split("\n")
    widths = [sum(font.getlength(char) for char in line) + max(0, len(line) - 1) * character_spacing
              for line in lines]
    line_height = font_size + 12
    size = (max(32, int(max(widths, default=1)) + 2 * pad + 4),
            len(lines) * line_height + (len(lines) - 1) * line_spacing + 2 * pad)
    background = rng.randint(235, 255)
    ink = rng.randint(0, 45)
    image = Image.new("L", size, background)
    draw = ImageDraw.Draw(image)
    for line_index, line in enumerate(lines):
        x = float(pad)
        y = pad + line_index * (line_height + line_spacing)
        for char in line:
            draw.text((x, y), char, font=font, fill=ink, anchor="lt")
            x += font.getlength(char) + character_spacing
    contrast = round(rng.uniform(0.75, 1.45), 3)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    blur = round(rng.uniform(0, 0.7), 3) if rng.random() < 0.45 else 0.0
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    noise_alpha = round(rng.uniform(0.005, 0.045), 3) if rng.random() < 0.55 else 0.0
    if noise_alpha:
        noise = Image.frombytes("L", size, rng.randbytes(size[0] * size[1]))
        image = Image.blend(image, noise, noise_alpha)
    return image, {
        "font_size": font_size, "character_spacing": character_spacing,
        "line_spacing": line_spacing, "contrast": contrast, "blur": blur,
        "noise_alpha": noise_alpha,
    }


def split_counts(total: int) -> dict[str, int]:
    validation = total // 10
    test = total // 10
    return {"train": total - validation - test, "validation": validation, "test": test}


def generate(output: Path, count: int, multiline_count: int, seed: int,
             fonts: list[Path], words: list[str]) -> dict:
    if count < 10 or multiline_count < 0:
        raise ValueError("--count must be at least 10 and --multiline-count nonnegative")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    rng = random.Random(seed)
    seen_lines: set[str] = set()
    counts = split_counts(count)
    extra_counts = split_counts(multiline_count)
    for group, targets in (("line", counts), ("multiline", extra_counts)):
        for split in SPLITS:
            target = targets[split]
            folder = output / group / split
            folder.mkdir(parents=True)
            manifest = output / (f"{split}.jsonl" if group == "line" else f"multiline_{split}.jsonl")
            with manifest.open("w", encoding="utf-8") as stream:
                for index in range(target):
                    for attempt in range(10000):
                        if group == "line":
                            kind = choose_kind(rng)
                            label = make_label(rng, words, kind)
                            parts = [label]
                        else:
                            kind = "multiline"
                            parts = [make_label(rng, words, "sentence") for _ in range(2)]
                            label = "\n".join(parts)
                        if split_for_label(label) == split and (
                            group == "line" or all(part not in seen_lines for part in parts)
                        ):
                            break
                    else:
                        raise RuntimeError(f"Could not generate a {group} label for {split}")
                    seen_lines.update(parts)
                    font = rng.choice(fonts)
                    image, appearance = render_label(label, font, rng)
                    relative = Path(group) / split / f"{index:06d}.png"
                    image.save(output / relative, format="PNG")
                    image.close()
                    record = {"image": relative.as_posix(), "text": label,
                              "kind": kind, "font": str(font), **appearance}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        "seed": seed, "line_counts": counts, "multiline_counts": extra_counts,
        "fonts": [str(font) for font in fonts], "pillow_version": pillow_version,
        "split_rule": "SHA256(label) mod 10: 0-7 train, 8 validation, 9 test",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/synthetic"))
    parser.add_argument("--count", type=int, default=10000, help="Single-line image count")
    parser.add_argument("--multiline-count", type=int, default=500,
                        help="Separate two-line images for spacing experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font", type=Path, action="append", help="Font path; repeat for variety")
    parser.add_argument("--corpus", type=Path, help="Optional local UTF-8 word source")
    args = parser.parse_args(argv)
    try:
        metadata = generate(args.output, args.count, args.multiline_count,
                            args.seed, discover_fonts(args.font), load_words(args.corpus))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
