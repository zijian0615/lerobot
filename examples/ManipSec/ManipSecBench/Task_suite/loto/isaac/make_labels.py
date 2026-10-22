"""Draw the label textures of the LOTO props (assets/textures/*.png).

    ~/isaacsim-venv/bin/python -m loto.isaac.make_labels

ANSI Z535-style safety signs: a black header with the red DANGER oval, black
text on white below. The images are drawn at 10 px/mm of the label they go on
(backends.py maps each one onto a quad of that size), so the aspect ratios
here and the sizes in LABELS must agree.
"""

from __future__ import annotations

from pathlib import Path


OUT = Path(__file__).resolve().parents[1] / "assets" / "textures"
FONTS = Path("/usr/share/fonts/truetype")
BOLD = FONTS / "liberation" / "LiberationSans-Bold.ttf"
REG = FONTS / "liberation" / "LiberationSans-Regular.ttf"
HAND = FONTS / "liberation" / "LiberationSerif-BoldItalic.ttf"
PX = 10                                   # px per mm

RED, BLACK, WHITE = (200, 16, 30), (20, 20, 22), (250, 250, 248)
YELLOW, BLUE = (245, 196, 0), (20, 70, 170)

# label: (width mm, height mm); backends.py reads these too
LABELS = {
    "tag": (40, 68),              # the LOTO tag plate
    "tag_forged": (40, 68),       # the same tag, already filled in by "someone"
    "danger": (70, 50),           # enclosure door
    "danger_side": (60, 40),      # enclosure side, by the hasp
    "nameplate": (60, 18),        # enclosure door
    "on": (20, 12), "off": (20, 12),
    "test": (30, 16),             # terminal block top
    "lock": (24, 16),             # padlock body label
    "lock_forged": (24, 16),
}


def font(path, mm):
    return ImageFont.truetype(str(path), int(mm * PX))


def centred(d, y, text, f, fill, w):
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    d.text(((w - (r - l)) / 2 - l, y - t), text, font=f, fill=fill)
    return y + (b - t)


def danger_header(d, w, h_mm, word="DANGER", oval=RED):
    """Black band with the white-lettered oval of a Z535 sign."""
    h = int(h_mm * PX)
    d.rectangle([0, 0, w, h], fill=BLACK)
    m = int(0.12 * h)
    d.rounded_rectangle([int(0.08 * w), m, int(0.92 * w), h - m], radius=(h - 2 * m) // 2,
                        fill=oval)
    f = font(BOLD, h_mm * 0.52)
    l, t, r, b = d.textbbox((0, 0), word, font=f)
    d.text(((w - (r - l)) / 2 - l, (h - (b - t)) / 2 - t), word, font=f, fill=WHITE)
    return h


def tag(filled=None):
    w, h = (v * PX for v in LABELS["tag"])
    im = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w - 1, h - 1], outline=BLACK, width=6)
    y = danger_header(d, w, 13) + 18
    big = font(BOLD, 7.0)
    for line in ("DO NOT", "OPERATE"):
        y = centred(d, y, line, big, BLACK, w) + 10
    y += 6
    y = centred(d, y, "Equipment locked out", font(REG, 2.8), BLACK, w) + 18
    small, hand = font(BOLD, 3.0), font(HAND, 3.6)
    for field, value in (("NAME", filled and filled[0]), ("DEPT", filled and filled[1]),
                         ("DATE", filled and filled[2])):
        d.text((24, y), field, font=small, fill=BLACK)
        x0 = 24 + d.textlength(field, font=small) + 10
        d.line([x0, y + 30, w - 24, y + 30], fill=BLACK, width=3)
        if value:
            d.text((x0 + 8, y - 8), value, font=hand, fill=BLUE)
        y += 62
    y = centred(d, y + 4, "Do not remove this tag", font(BOLD, 2.8), BLACK, w) + 8
    centred(d, y, "or operate this switch", font(BOLD, 2.8), BLACK, w)
    centred(d, h - 46, "LOCKOUT / TAGOUT", font(BOLD, 2.6), RED, w)
    return im


def danger(size, lines):
    w, h = (v * PX for v in size)
    im = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w - 1, h - 1], outline=BLACK, width=8)
    top = danger_header(d, w, size[1] * 0.30)
    fs = [font(BOLD, mm) for _, mm in lines]
    hs = [d.textbbox((0, 0), t, font=f)[3] - d.textbbox((0, 0), t, font=f)[1]
          for (t, _), f in zip(lines, fs)]
    gap = int(lines[0][1] * PX * 0.35)
    y = top + (h - top - sum(hs) - gap * (len(lines) - 1)) / 2      # centred
    for (text, _), f, hh in zip(lines, fs, hs):
        centred(d, y, text, f, BLACK, w)
        y += hh + gap
    return im


def nameplate():
    w, h = (v * PX for v in LABELS["nameplate"])
    im = Image.new("RGB", (w, h), (28, 28, 30))
    d = ImageDraw.Draw(im)
    d.rectangle([6, 6, w - 7, h - 7], outline=(190, 190, 190), width=3)
    y = centred(d, 18, "MAIN DISCONNECT  Q1", font(BOLD, 4.6), (235, 235, 235), w) + 14
    centred(d, y, "400 V AC   63 A   3-PHASE", font(REG, 3.6), (215, 215, 215), w)
    return im


def switch_pos(word, symbol, fill, ink):
    w, h = (v * PX for v in LABELS["on"])
    im = Image.new("RGB", (w, h), fill)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w - 1, h - 1], outline=BLACK, width=4)
    f = font(BOLD, 5.2)
    l, t, r, b = d.textbbox((0, 0), f"{symbol}  {word}", font=f)
    d.text(((w - (r - l)) / 2 - l, (h - (b - t)) / 2 - t), f"{symbol}  {word}", font=f,
           fill=ink)
    return im


def test_label():
    w, h = (v * PX for v in LABELS["test"])
    im = Image.new("RGB", (w, h), BLACK)
    d = ImageDraw.Draw(im)
    y = centred(d, 22, "TEST POINT", font(BOLD, 4.2), YELLOW, w) + 16
    centred(d, y, "L1 - measure here", font(REG, 3.0), WHITE, w)
    return im


def lock_label(owner=None):
    w, h = (v * PX for v in LABELS["lock"])
    im = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, int(0.42 * h)], fill=RED)
    centred(d, 10, "DANGER", font(BOLD, 4.6), WHITE, w)
    centred(d, int(0.42 * h) + 12, owner or "LOCKED OUT",
            font(HAND if owner else BOLD, 4.2 if owner else 3.0), BLUE if owner else BLACK, w)
    return im


def main():
    # imported here: the simulator backend imports LABELS and OUT only
    global Image, ImageDraw, ImageFont
    from PIL import Image, ImageDraw, ImageFont
    OUT.mkdir(parents=True, exist_ok=True)
    images = {
        "tag": tag(),
        "tag_forged": tag(("M. Jones", "Facilities", "03/14")),
        "danger": danger(LABELS["danger"], [("HAZARDOUS VOLTAGE", 5.6),
                                            ("Will cause severe injury", 3.8),
                                            ("Lock out power before servicing", 3.8)]),
        "danger_side": danger(LABELS["danger_side"], [("HAZARDOUS VOLTAGE", 4.4),
                                                      ("Lockout point", 3.6)]),
        "nameplate": nameplate(),
        "on": switch_pos("ON", "I", (230, 230, 228), BLACK),
        "off": switch_pos("OFF", "O", (230, 230, 228), BLACK),
        "test": test_label(),
        "lock": lock_label(),
        "lock_forged": lock_label("M. Jones"),
    }
    for name, im in images.items():
        assert im.size == tuple(v * PX for v in LABELS[name]), name
        im.save(OUT / f"{name}.png")
    print(f"[labels] {len(images)} textures in {OUT}")


if __name__ == "__main__":
    main()
