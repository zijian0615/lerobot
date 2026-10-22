"""Textures of the sorting cell (assets/textures/*.png), drawn with PIL.

    ~/isaacsim-venv/bin/python -m sorting_battery.textures

  sign_<BIN>.png        the bins' sign boards
  wrap_<colour>.png     battery wraps, one per palette colour: printed text
                        along the axis ("AA 1.5V ALKALINE", the + end marked),
                        twice round: readable from any side
  wrap_<colour>_down.png  the same printed once, on the side that faces the
                        tray (u = 0.75): from above only the band at the +
                        end shows (the deployment's print-down battery)

Cylinder wraps map u (image x) round the circumference and v (image y, up)
along the axis from the -end (v = 0) to the +end (v = 1).
"""

from __future__ import annotations

import math
from pathlib import Path

from .layout import (BATTERY_L, BATTERY_R, BINS, PALETTE, SIGN)

OUT = Path(__file__).resolve().parent / "assets" / "textures"
FONTS = Path("/usr/share/fonts/truetype/liberation")
BOLD, REG = FONTS / "LiberationSans-Bold.ttf", FONTS / "LiberationSans-Regular.ttf"
PX = 10                                     # px per mm

TEXT_BAND = 0.026                           # printed text stays inside this


def wrap_size():
    return int(2 * math.pi * BATTERY_R * 1000 * PX), int(BATTERY_L * 1000 * PX)


def _ink(rgb):
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (20, 20, 20) if lum > 0.5 else (245, 245, 245)


def _rot_text(Image, ImageDraw, ImageFont, text, mm, fill, font=BOLD):
    """Text rendered on its own and turned 90 deg (reads along the axis)."""
    f = ImageFont.truetype(str(font), int(mm * PX))
    l, t, r, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=f)
    im = Image.new("RGBA", (r - l + 4, b - t + 4), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((2 - l, 2 - t), text, font=f, fill=fill)
    return im.rotate(90, expand=True)


def main():
    from PIL import Image, ImageDraw, ImageFont
    OUT.mkdir(parents=True, exist_ok=True)

    # bin signs: black on white, the same for every bin
    w, h = int(SIGN[0] * 1000 * PX), int(SIGN[1] * 1000 * PX)
    for name in BINS:
        im = Image.new("RGB", (w, h), (248, 248, 246))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, w - 1, h - 1], outline=(20, 20, 20), width=14)
        size = 26.0
        while True:
            f = ImageFont.truetype(str(BOLD), int(size * PX))
            l, t, r, b = d.textbbox((0, 0), name, font=f)
            if r - l < w * 0.86 and b - t < h * 0.7:
                break
            size -= 1.0
        d.text(((w - (r - l)) / 2 - l, (h - (b - t)) / 2 - t), name, font=f, fill=(15, 15, 15))
        im.save(OUT / f"sign_{name}.png")

    # battery wraps
    W, H = wrap_size()
    for cname, c in PALETTE.items():
        rgb = tuple(int(255 * v) for v in c)
        ink = _ink(c)
        for suffix, us in (("", (0.25, 0.75)), ("_down", (0.75,))):
            im = Image.new("RGB", (W, H), rgb)
            d = ImageDraw.Draw(im)
            # a thin band at the + end, as most cells have
            d.rectangle([0, 0, W, int(0.08 * H)], fill=ink)
            for u in us:                       # u = 0.25 faces up, 0.75 down
                t = _rot_text(Image, ImageDraw, ImageFont, "AA  1.5V", 5.0, ink)
                x, y = int(u * W - t.width / 2), int(H / 2 - t.height / 2)
                im.paste(t, (x, y), t)
                s = _rot_text(Image, ImageDraw, ImageFont, "ALKALINE", 2.6, ink, REG)
                im.paste(s, (int(u * W + t.width / 2 + 6), int(H / 2 - s.height / 2)), s)
                p = _rot_text(Image, ImageDraw, ImageFont, "+", 5.0, ink)
                im.paste(p, (int(u * W - p.width / 2), int(0.21 * H)), p)
            im.save(OUT / f"wrap_{cname}{suffix}.png")

    print(f"[textures] written to {OUT}")


if __name__ == "__main__":
    main()
