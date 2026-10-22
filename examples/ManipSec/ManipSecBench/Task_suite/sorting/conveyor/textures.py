"""Textures of the dynamic-sorting cell (sorting/assets/textures/conv_*.png).

    ~/isaacsim-venv/bin/python -m sorting.conveyor.textures

  conv_parcel_<A|B|C>.png   a parcel's top face: kraft board, a white label
                            with the class letter
  conv_sign_<A|B|C>.png     the bins' sign boards
  conv_kraft.png            the parcels' other faces
"""

from __future__ import annotations

from pathlib import Path

from .layout import CLASSES, PARCEL, SIGN

OUT = Path(__file__).resolve().parents[1] / "assets" / "textures"
BOLD = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")
PX = 10
KRAFT = (176, 132, 86)


def main():
    from PIL import Image, ImageDraw, ImageFont
    import random
    OUT.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(0)

    def kraft(w, h):
        im = Image.new("RGB", (w, h), KRAFT)
        px = im.load()
        for _ in range(w * h // 6):                       # fibre speckle
            x, y = rnd.randrange(w), rnd.randrange(h)
            d = rnd.randint(-14, 14)
            px[x, y] = tuple(max(0, min(255, c + d)) for c in KRAFT)
        return im

    L, W = int(PARCEL[0] * 1000 * PX), int(PARCEL[1] * 1000 * PX)   # top face: 44 x 28 mm
    kraft(L, W).save(OUT / "conv_kraft.png")
    for c in CLASSES:
        im = kraft(L, W)
        d = ImageDraw.Draw(im)
        m = 18
        d.rounded_rectangle([L / 2 - W / 2 + m, m, L / 2 + W / 2 - m, W - m], radius=12,
                            fill=(248, 248, 244), outline=(30, 30, 30), width=4)
        f = ImageFont.truetype(str(BOLD), int(W * 0.62))
        l, t, r, b = d.textbbox((0, 0), c, font=f)
        d.text(((L - (r - l)) / 2 - l, (W - (b - t)) / 2 - t), c, font=f, fill=(15, 15, 15))
        im.save(OUT / f"conv_parcel_{c}.png")

    w, h = int(SIGN[0] * 1000 * PX), int(SIGN[1] * 1000 * PX)
    for c in CLASSES:
        im = Image.new("RGB", (w, h), (248, 248, 246))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, w - 1, h - 1], outline=(20, 20, 20), width=14)
        f = ImageFont.truetype(str(BOLD), int(h * 0.72))
        l, t, r, b = d.textbbox((0, 0), c, font=f)
        d.text(((w - (r - l)) / 2 - l, (h - (b - t)) / 2 - t), c, font=f, fill=(15, 15, 15))
        im.save(OUT / f"conv_sign_{c}.png")
    print(f"[textures] conveyor textures in {OUT}")


if __name__ == "__main__":
    main()
