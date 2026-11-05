#!/usr/bin/env python3
"""Generate portable SVG headings for GitHub's CSS-restricted Markdown viewer.

Run from any directory. No font downloads or third-party packages are required.
Heading labels stay in image alt text; explicit anchors retain existing links.
"""

import html
import os
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets/docs/headings"
DOCUMENTS = (
    "README.md", "SECURITY.md", "THIRD_PARTY_NOTICES.md",
    "SCOPE/README.md", "ssh-keys/README.md", "chat-templates/README.md",
    "docs/installation.md", "docs/models.md", "docs/usage.md", "docs/sources.md",
)

# Original, simple 5-by-7 block lettering. Each hex value describes one row.
ROWS = {
    "A": "0e 11 11 1f 11 11 11", "B": "1e 11 11 1e 11 11 1e",
    "C": "0f 10 10 10 10 10 0f", "D": "1e 11 11 11 11 11 1e",
    "E": "1f 10 10 1e 10 10 1f", "F": "1f 10 10 1e 10 10 10",
    "G": "0f 10 10 13 11 11 0f", "H": "11 11 11 1f 11 11 11",
    "I": "1f 04 04 04 04 04 1f", "J": "07 02 02 02 12 12 0c",
    "K": "11 12 14 18 14 12 11", "L": "10 10 10 10 10 10 1f",
    "M": "11 1b 15 15 11 11 11", "N": "11 19 15 13 11 11 11",
    "O": "0e 11 11 11 11 11 0e", "P": "1e 11 11 1e 10 10 10",
    "Q": "0e 11 11 11 15 12 0d", "R": "1e 11 11 1e 14 12 11",
    "S": "0f 10 10 0e 01 01 1e", "T": "1f 04 04 04 04 04 04",
    "U": "11 11 11 11 11 11 0e", "V": "11 11 11 11 11 0a 04",
    "W": "11 11 11 15 15 1b 11", "X": "11 11 0a 04 0a 11 11",
    "Y": "11 11 0a 04 04 04 04", "Z": "1f 01 02 04 08 10 1f",
    "0": "0e 11 13 15 19 11 0e", "1": "04 0c 04 04 04 04 0e",
    "2": "0e 11 01 02 04 08 1f", "3": "1e 01 01 0e 01 01 1e",
    "4": "02 06 0a 12 1f 02 02", "5": "1f 10 10 1e 01 01 1e",
    "6": "0e 10 10 1e 11 11 0e", "7": "1f 01 02 04 08 08 08",
    "8": "0e 11 11 0e 11 11 0e", "9": "0e 11 11 0f 01 01 0e",
    " ": "00 00 00 00 00 00 00", "-": "00 00 00 1f 00 00 00",
    ".": "00 00 00 00 00 06 06", ",": "00 00 00 00 06 06 04",
    "'": "04 04 02 00 00 00 00", "/": "01 01 02 04 08 10 10",
    ":": "00 06 06 00 06 06 00", "&": "0c 12 14 08 15 12 0d",
    "(": "02 04 08 08 08 04 02", ")": "08 04 02 02 02 04 08",
    "+": "00 04 04 1f 04 04 00", "?": "0e 11 01 02 04 00 04",
}
GLYPHS = {char: [int(row, 16) for row in rows.split()] for char, rows in ROWS.items()}
HEADING = re.compile(r"^(#{1,3}) (.+?)\s*$")
IMAGE = re.compile(r"!\[([^\]]+)\]\(([^)]+)\)")


def slug(label):
    return re.sub(r"[^\w\- ]", "", label.lower()).replace(" ", "-")


def svg(label, level):
    # Draw the letters as paths so every viewer gets the same block font.
    scale = {1: 3, 2: 2.5, 3: 2}[level]
    visible = label.upper().translate(str.maketrans({"—": "-", "–": "-", "’": "'"}))
    unknown = set(visible) - GLYPHS.keys()
    if unknown:
        raise ValueError(f"Add glyphs for {sorted(unknown)} in heading {label!r}")
    lines = textwrap.wrap(visible, width=int((720 - 32) / (6 * scale))) or [""]
    width = (max(map(len, lines)) * 6 - 1) * scale + 32
    height = (len(lines) * 10 - 3) * scale + 24
    commands = []
    for line_no, line in enumerate(lines):
        for char_no, char in enumerate(line):
            for row_no, row in enumerate(GLYPHS[char]):
                bits = f"{row:05b}"
                for run in re.finditer("1+", bits):
                    x = 16 + (char_no * 6 + run.start()) * scale
                    y = 12 + (line_no * 10 + row_no) * scale
                    w = len(run.group()) * scale
                    commands.append(f"M{x:g},{y:g}h{w:g}v{scale:g}h{-w:g}z")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" height="{height:g}" '
        f'viewBox="0 0 {width:g} {height:g}" role="img" aria-labelledby="title">\n'
        f'  <title id="title">{html.escape(label)}</title>\n'
        f'  <rect width="{width:g}" height="{height:g}" rx="4" fill="#080d09"/>\n'
        f'  <path fill="#39ff14" d="{"".join(commands)}"/>\n'
        '</svg>\n'
    )


def render_document(path):
    output, images, counts = [], {}, {}
    fence = None
    for line in path.read_text().splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            current = marker.group(1)
            if fence is None:
                fence = current
            elif current[0] == fence[0] and len(current) >= len(fence):
                fence = None
            output.append(line)
            continue
        match = HEADING.fullmatch(line) if fence is None else None
        if not match:
            output.append(line)
            continue
        marks, label = match.groups()
        if image := IMAGE.fullmatch(label):
            if "assets/docs/headings/" not in image.group(2):
                output.append(line)
                continue
            label = image.group(1)
        base = slug(label)
        count = counts.get(base, 0)
        counts[base] = count + 1
        anchor = base if count == 0 else f"{base}-{count}"
        anchor_tag = f'<a name="{html.escape(anchor, quote=True)}"></a>'
        if output[-2:] != [anchor_tag, ""]:
            output.extend([anchor_tag, ""])
        asset = ASSETS / f"h{len(marks)}-{base}.svg"
        images[asset] = svg(label, len(marks))
        target = Path(os.path.relpath(asset, path.parent)).as_posix()
        output.append(f"{marks} ![{label}]({target})")
    return "\n".join(output) + "\n", images


def main():
    # Prepare everything before writing, so an unsupported glyph fails cleanly.
    documents, images = {}, {}
    for name in DOCUMENTS:
        path = ROOT / name
        documents[path], assets = render_document(path)
        images.update(assets)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for path, body in {**documents, **images}.items():
        if not path.exists() or path.read_text() != body:
            path.write_text(body)
    print(f"Styled {len(documents)} documents; generated {len(images)} SVG headings.")


if __name__ == "__main__":
    main()
