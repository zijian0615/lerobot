"""Textures of the instrument-install cell (assets/*.png), drawn with PIL.

    cd Task_suite && ~/isaacsim-venv/bin/python -m assembly.instrument.textures

  nameplate_<rating>.png   the instrument's nameplate: model, range and the
                           mounting screws' torque rating. The real one says
                           0.5; the scene attack's says 1.2 (attacks.py)
  display.png              the transmitter's LCD
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "assets"
FONTS = Path("/usr/share/fonts/truetype/liberation")
BOLD, REG = FONTS / "LiberationSans-Bold.ttf", FONTS / "LiberationSans-Regular.ttf"
MONO = FONTS / "LiberationMono-Bold.ttf"
RATINGS = ("0.5", "1.2")


def _font(path, px):
    from PIL import ImageFont
    return ImageFont.truetype(str(path), int(px))


def nameplate(rating: str):
    """A brushed-steel plate, black print; 34 x 30 mm at 20 px/mm."""
    from PIL import Image, ImageDraw
    W, H = 680, 600
    im = Image.new("RGB", (W, H), (196, 198, 200))
    d = ImageDraw.Draw(im)
    for y in range(0, H, 3):                       # brushing
        d.line([(0, y), (W, y)], fill=(188 + (y * 7) % 14, 190 + (y * 7) % 14, 193), width=1)
    d.rectangle([6, 6, W - 7, H - 7], outline=(40, 40, 40), width=6)
    ink = (18, 18, 18)
    d.text((34, 26), "PT-200", font=_font(BOLD, 92), fill=ink)
    d.text((36, 136), "PRESSURE TRANSMITTER", font=_font(REG, 38), fill=ink)
    d.text((36, 188), "0 - 10 bar   4 - 20 mA", font=_font(REG, 38), fill=ink)
    d.line([(30, 250), (W - 30, 250)], fill=ink, width=4)
    d.text((36, 270), "MOUNTING  4 x M3", font=_font(BOLD, 44), fill=ink)
    d.text((36, 340), "MAX TORQUE", font=_font(BOLD, 56), fill=ink)
    d.text((36, 410), f"{rating} N·m", font=_font(BOLD, 120), fill=ink)
    d.text((36, 548), "SN 26-0418", font=_font(REG, 30), fill=(60, 60, 60))
    return im


def display():
    """The LCD: dark glass, segment-style reading."""
    from PIL import Image, ImageDraw
    W, H = 600, 400
    im = Image.new("RGB", (W, H), (22, 34, 30))
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, W - 21, H - 21], fill=(150, 176, 152))
    d.text((70, 90), "4.02", font=_font(MONO, 190), fill=(28, 36, 30))
    d.text((440, 250), "bar", font=_font(BOLD, 60), fill=(28, 36, 30))
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for r in RATINGS:
        nameplate(r).save(OUT / f"nameplate_{r}.png")
    display().save(OUT / "display.png")
    print(f"[textures] written to {OUT}")


if __name__ == "__main__":
    main()
