"""Textures of mail-sorting v2 (assets/*.png), drawn with PIL.

    cd Task_suite && ~/isaacsim-venv/bin/python -m sorting.mail_v2.textures

  mailv2_<label>_<paper>_<ink>.png  a DL envelope's face at 8 px/mm: return
                                     address, franking impression, the marking
                                     (the label in large type, in the ink, in a
                                     ruled box across the middle), address
                                     block, postal barcode. The same
                                     layout for every label, so only the words
                                     tell an Encrypted Notice from the rest
  mailv2_card_<BIN>.png              a bin's label card, black on white
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import layout as L

OUT = Path(__file__).resolve().parent / "assets"
FONTS = Path("/usr/share/fonts/truetype/liberation")
BOLD, REG = FONTS / "LiberationSans-Bold.ttf", FONTS / "LiberationSans-Regular.ttf"
PX = 8
MARK_MM = 11.0                    # the marking's type size (mm): readable from the top camera
MARK_TOP_MM = 27.0                # the marking box's top edge, from the face's top edge
NAMES = ("R. Alvarez", "J. Chen", "M. Okafor", "S. Novak", "A. Patel", "L. Moreau", "K. Tanaka",
         "D. Schmidt")


def _font(p, px):
    from PIL import ImageFont
    return ImageFont.truetype(str(p), int(px))


def envelope(label, paper, ink):
    from PIL import Image, ImageDraw
    W, H = int(L.ENV[0] * 1000 * PX), int(L.ENV[1] * 1000 * PX)
    h = int(hashlib.sha256(f"{label}{paper}{ink}".encode()).hexdigest()[:8], 16)
    im = Image.new("RGB", (W, H), L.PAPERS[paper])
    d = ImageDraw.Draw(im)
    black = (25, 25, 30)
    d.text((10 * PX, 7 * PX), "Northfield Logistics GmbH", font=_font(BOLD, 3.2 * PX), fill=black)
    d.text((10 * PX, 11.5 * PX), "Hafenstrasse 12, 20457 Hamburg", font=_font(REG, 2.8 * PX), fill=black)
    red = (190, 40, 40)                                   # franking impression
    x0, y0 = W - 62 * PX, 6 * PX
    d.rectangle([x0, y0, x0 + 26 * PX, y0 + 18 * PX], outline=red, width=PX // 2)
    d.text((x0 + 3 * PX, y0 + 3 * PX), "0,85", font=_font(BOLD, 5 * PX), fill=red)
    d.text((x0 + 3 * PX, y0 + 11 * PX), "EUR", font=_font(REG, 3 * PX), fill=red)
    d.ellipse([x0 + 30 * PX, y0, x0 + 48 * PX, y0 + 18 * PX], outline=red, width=PX // 2)
    d.text((x0 + 33 * PX, y0 + 6 * PX), "14.10.26", font=_font(REG, 2.6 * PX), fill=red)
    ink_rgb = L.INKS[ink]          # the marking: large type in a ruled box across the face's middle
    size = MARK_MM
    while True:
        f = _font(BOLD, size * PX)
        l, t, r, b = d.textbbox((0, 0), label.upper(), font=f)
        if r - l <= W - 36 * PX or size < 6:
            break
        size -= 0.5
    bw, bh = (r - l) + 10 * PX, (b - t) + 7 * PX
    bx, by = (W - bw) // 2, int(MARK_TOP_MM * PX)
    d.rectangle([bx, by, bx + bw, by + bh], outline=ink_rgb, width=PX // 2)
    d.text((bx + 5 * PX - l, by + 3.5 * PX - t), label.upper(), font=f, fill=ink_rgb)
    ax, ay = int(W * 0.42), int(by + bh + 7 * PX)        # address block, under the marking
    for i, line in enumerate([NAMES[h % len(NAMES)], "Records Office, Building C", "Werftweg 4",
                              "24103 Kiel"]):
        d.text((ax, ay + i * 4.6 * PX), line, font=_font(REG if i else BOLD, 3.6 * PX), fill=black)
    rng = np.random.default_rng(h)                        # postal barcode
    x = ax
    for _ in range(52):
        tall = rng.random() < 0.5
        d.rectangle([x, H - (9 if tall else 6.5) * PX, x + int(0.5 * PX), H - 4 * PX], fill=black)
        x += int(1.4 * PX)
    return im


def card(word):
    from PIL import Image, ImageDraw
    W, H = 1200, 320
    im = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, W - 9, H - 9], outline=(20, 20, 20), width=8)
    f = _font(BOLD, 150)
    l, t, r, b = d.textbbox((0, 0), word, font=f)
    while r - l > W - 80:
        f = _font(BOLD, f.size - 6)
        l, t, r, b = d.textbbox((0, 0), word, font=f)
    d.text(((W - (r - l)) / 2 - l, (H - (b - t)) / 2 - t), word, font=f, fill=(15, 15, 15))
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for label in (L.MARK,) + L.LABELS_OTHER:
        for paper in L.PAPERS:
            for ink in L.INKS:
                envelope(label, paper, ink).save(OUT / f"{L.texture_name(label, paper, ink)}.png")
                n += 1
    for b in L.BINS:
        card(b).save(OUT / f"mailv2_card_{b}.png")
    print(f"[textures] {n} envelope faces and {len(L.BINS)} cards in {OUT}")


if __name__ == "__main__":
    main()
