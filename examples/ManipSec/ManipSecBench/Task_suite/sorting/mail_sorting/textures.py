"""Textures of the mail-sorting cell (sorting/assets/textures/mail_*.png),
drawn with PIL.

    ~/isaacsim-venv/bin/python -m sorting.mail_sorting.textures

  mail_<label>_<paper>_<ink>.png  an envelope's face: the paper, a stamp, a
                                  return address, the marking (the label, in
                                  the ink, in a ruled box) and an address
                                  block. One per label x paper x ink: the same
                                  layout for every label, so only the words
                                  tell an Encrypted Notice from the rest
  mail_back_<paper>.png           the back: the flap's folds
  mail_sign_<BIN>.png             the bins' sign boards, black on white

Image x runs along the envelope's length (+x), image up along +y.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .layout import BINS, ENVELOPE, INKS, LABELS_OTHER, MARK, PAPERS, SIGN, texture_name

OUT = Path(__file__).resolve().parents[1] / "assets" / "textures"
FONTS = Path("/usr/share/fonts/truetype/liberation")
BOLD, REG = FONTS / "LiberationSans-Bold.ttf", FONTS / "LiberationSans-Regular.ttf"
PX = 8                                    # px per mm
STAMPS = ((176, 40, 40), (40, 110, 60), (40, 70, 150), (190, 120, 30))


def _fit(d, text, font, max_w, size_mm, ImageFont):
    while True:
        f = ImageFont.truetype(str(font), int(size_mm * PX))
        l, t, r, b = d.textbbox((0, 0), text, font=f)
        if r - l <= max_w or size_mm < 3:
            return f, (l, t, r, b)
        size_mm -= 0.25


def face(label, paper, ink):
    from PIL import Image, ImageDraw, ImageFont
    import random
    W, H = int(ENVELOPE[0] * 1000 * PX), int(ENVELOPE[1] * 1000 * PX)
    seed = int(hashlib.sha256(f"{label}|{paper}|{ink}".encode()).hexdigest()[:8], 16)
    rnd = random.Random(seed)
    base = PAPERS[paper]
    im = Image.new("RGB", (W, H), base)
    px = im.load()
    for _ in range(W * H // 10):                           # paper grain
        x, y = rnd.randrange(W), rnd.randrange(H)
        e = rnd.randint(-6, 6)
        px[x, y] = tuple(max(0, min(255, c + e)) for c in base)
    d = ImageDraw.Draw(im)
    mm = lambda v: int(v * PX)
    grey = (120, 120, 118)
    # stamp, top right; return address, top left
    sc = STAMPS[seed % len(STAMPS)]
    d.rectangle([W - mm(19), mm(3), W - mm(4), mm(20)], fill=(250, 250, 248))
    d.rectangle([W - mm(18), mm(4), W - mm(5), mm(19)], fill=sc)
    for k in range(3):
        d.rectangle([mm(4), mm(4 + 2.6 * k), mm(4 + (26 - 5 * k)), mm(5 + 2.6 * k)], fill=grey)
    # the marking: bold, in a ruled box, centred
    f, (l, t, r, b) = _fit(d, label, BOLD, W - mm(16), 6.0, ImageFont)
    tw, th = r - l, b - t
    cx, cy = W / 2, mm(25)
    pad = mm(2.2)
    d.rectangle([cx - tw / 2 - pad, cy - th / 2 - pad, cx + tw / 2 + pad, cy + th / 2 + pad],
                outline=INKS[ink], width=mm(0.6))
    d.text((cx - tw / 2 - l, cy - th / 2 - t), label, font=f, fill=INKS[ink])
    # address block, lower middle
    for k in range(3):
        d.rectangle([mm(30), mm(35 + 3.2 * k), mm(30 + (36 - 6 * (k % 2))), mm(36.2 + 3.2 * k)],
                    fill=grey)
    return im


def back(paper):
    from PIL import Image, ImageDraw
    W, H = int(ENVELOPE[0] * 1000 * PX), int(ENVELOPE[1] * 1000 * PX)
    shade = tuple(int(c * 0.93) for c in PAPERS[paper])
    im = Image.new("RGB", (W, H), PAPERS[paper])
    d = ImageDraw.Draw(im)
    d.line([(0, 0), (W / 2, H * 0.55), (W, 0)], fill=shade, width=6)
    d.line([(0, H), (W * 0.42, H * 0.45)], fill=shade, width=4)
    d.line([(W, H), (W * 0.58, H * 0.45)], fill=shade, width=4)
    return im


def sign(name):
    from PIL import Image, ImageDraw, ImageFont
    w, h = int(SIGN[0] * 1000 * 10), int(SIGN[1] * 1000 * 10)
    im = Image.new("RGB", (w, h), (248, 248, 246))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w - 1, h - 1], outline=(20, 20, 20), width=14)
    size = 30.0
    while True:
        f = ImageFont.truetype(str(BOLD), int(size * 10))
        l, t, r, b = d.textbbox((0, 0), name, font=f)
        if r - l < w * 0.88 and b - t < h * 0.62:
            break
        size -= 0.5
    d.text(((w - (r - l)) / 2 - l, (h - (b - t)) / 2 - t), name, font=f, fill=(15, 15, 15))
    return im


def path(name: str) -> Path:
    """A texture's file, drawn on first use."""
    p = OUT / f"{name}.png"
    if not p.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        if name.startswith("mail_sign_"):
            im = sign(name[len("mail_sign_"):])
        elif name.startswith("mail_back_"):
            im = back(name[len("mail_back_"):])
        else:
            im = next(face(lb, pa, ik) for lb in (MARK,) + LABELS_OTHER for pa in PAPERS
                      for ik in INKS if texture_name(lb, pa, ik) == name)
        im.save(p, optimize=True)
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = ([f"mail_sign_{b}" for b in BINS] + [f"mail_back_{p}" for p in PAPERS]
             + [texture_name(lb, pa, ik) for lb in (MARK,) + LABELS_OTHER for pa in PAPERS
                for ik in INKS])
    for n in names:
        (OUT / f"{n}.png").unlink(missing_ok=True)
        path(n)
    print(f"[textures] {len(names)} mail-sorting textures in {OUT}")


if __name__ == "__main__":
    main()
