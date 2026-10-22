"""The real xArm6's finish on Isaac Sim's xarm6.usd, which has one flat white
material per link: glossy pearl-white shells, brushed-aluminium rings at the
joints, the embossed hexagonal "UF" badge on the joint caps (from a photo of
the lab's arm).

    from assembly.instrument import xarm_look
    xarm_look.apply(stage, "/World/xarm6")      # before world.reset()

Rings are face subsets of each link's mesh, chosen in the link's own frame
(the band of faces around a joint axis near a link-to-link interface); badges
are textured discs on the caps' outer faces. Purely visual: nothing here has
a collider or mass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
BADGE = ASSETS / "xarm_badge.png"
SHELL = (0.93, 0.92, 0.88)                   # pearl white

# rings: (link, axis, lo, hi, radius below which a face is on that joint's housing,
# axis origin in the link frame), all in the link's frame, metres
RINGS = (
    ("link1", 2, -0.112, -0.090, 0.060, (0.0, 0.0)),            # J1, the base column
    ("link2", 2, 0.077, 0.097, 0.052, (0.0, 0.0)),              # J2, the shoulder
    ("link4", 2, -0.173, -0.150, 0.050, (0.0, 0.0)),            # J4, the forearm
)
# badges on the caps' outer faces: (link, centre, normal, diameter)
BADGES = (
    ("link2", (0.0, 0.0, 0.1195), (0, 0, 1), 0.040),            # shoulder
    ("link3", (0.0, 0.0, 0.1075), (0, 0, 1), 0.036),            # elbow
    ("link5", (0.0, 0.0, 0.0585), (0, 0, 1), 0.034),            # wrist
)


def badge_texture(path=BADGE, px=512):
    """The badge: a hexagon around 'UF', raised from the shell (light edge up
    and left, shadow down and right), on the shell's colour."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    # the texture is sRGB, the shell's colour linear: match them
    base = tuple(int(round(255 * (1.055 * c ** (1 / 2.4) - 0.055))) for c in SHELL)
    im = Image.new("RGB", (px, px), base)

    def layer(dx, dy, fill):
        m = Image.new("L", (px, px), 0)
        d = ImageDraw.Draw(m)
        c, r = px / 2, px * 0.42
        hexagon = [(c + dx + r * np.cos(np.radians(a)), c + dy + r * np.sin(np.radians(a)))
                   for a in range(30, 390, 60)]
        d.line(hexagon, fill=255, width=int(px * 0.045), joint="curve")
        r2 = px * 0.33
        inner = [(c + dx + r2 * np.cos(np.radians(a)), c + dy + r2 * np.sin(np.radians(a)))
                 for a in range(30, 390, 60)]
        d.line(inner, fill=255, width=int(px * 0.02), joint="curve")
        f = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", int(px * 0.26))
        l, t, rr, b = d.textbbox((0, 0), "UF", font=f)
        d.text((c + dx - (rr - l) / 2 - l, c + dy - (b - t) / 2 - t), "UF", font=f, fill=255)
        return m.filter(ImageFilter.GaussianBlur(px * 0.004)), fill

    for dx, dy, shade in ((-3, -3, 1.06), (3, 3, 0.80), (0, 0, 0.97)):
        m, _ = layer(dx, dy, None)
        tone = tuple(min(255, int(v * shade)) for v in base)
        im.paste(Image.new("RGB", (px, px), tone), (0, 0), m)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path


def _material(stage, key, color=None, texture=None, metal=0.0, rough=0.5):
    from pxr import Sdf
    from isaacsim.core.api.materials import OmniPBR
    p = OmniPBR(prim_path=f"/World/Looks/{key}", name=key, color=np.array(color or (1.0, 1.0, 1.0)))
    p.set_metallic_constant(metal)
    p.set_reflection_roughness(rough)
    if texture:
        p.set_project_uvw(False)
        p.shaders_list[0].CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(texture)))
    return p.material


def _bind(prim, mat, strong=True):
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants if strong
        else UsdShade.Tokens.weakerThanDescendants)


def _deinstance(stage, root):
    """Make a link's visuals editable; return the prims the asset binds a
    material on (strongerThanDescendants: rebound weaker by apply(), or they
    would hide the ring subsets)."""
    from pxr import Usd, UsdGeom, UsdShade
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.IsInstance():
            p.SetInstanceable(False)
    return [p for p in Usd.PrimRange(stage.GetPrimAtPath(root))
            if not p.IsA(UsdGeom.Subset) and not p.IsA(UsdShade.Material)]


def _link_mesh(stage, robot, link):
    from pxr import Usd, UsdGeom
    for p in Usd.PrimRange(stage.GetPrimAtPath(f"{robot}/{link}/visuals")):
        if p.IsA(UsdGeom.Mesh):
            return UsdGeom.Mesh(p)
    return None


def apply(stage, robot="/World/xarm6"):
    from pxr import Gf, Sdf, UsdGeom
    if not BADGE.exists():
        badge_texture()
    shell = _material(stage, "xarm_shell", SHELL, metal=0.0, rough=0.18)
    alu = _material(stage, "xarm_alu", (0.74, 0.75, 0.77), metal=0.9, rough=0.32)
    badge = _material(stage, "xarm_badge", texture=BADGE, metal=0.0, rough=0.2)
    for link in ("world", "link1", "link2", "link3", "link4", "link5"):
        for p in _deinstance(stage, f"{robot}/{link}/visuals"):
            _bind(p, shell, strong=False)   # weaker: the ring subsets below override it
        m = _link_mesh(stage, robot, link)
        if m is not None:
            _bind(m.GetPrim(), shell, strong=False)
    link_xf = {}
    for link, ax, lo, hi, rmax, o in RINGS:
        m = _link_mesh(stage, robot, link)
        if m is None:
            continue
        P = np.array(m.GetPointsAttr().Get(), float)
        T = (np.array(UsdGeom.Xformable(m.GetPrim()).ComputeLocalToWorldTransform(0))
             @ np.linalg.inv(np.array(UsdGeom.Xformable(stage.GetPrimAtPath(f"{robot}/{link}"))
                                      .ComputeLocalToWorldTransform(0))))
        P = np.c_[P, np.ones(len(P))] @ T
        counts = np.array(m.GetFaceVertexCountsAttr().Get(), int)
        idx = np.array(m.GetFaceVertexIndicesAttr().Get(), int)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        cen = np.array([P[idx[s:s + c], :3].mean(0) for s, c in zip(starts, counts)])
        other = [i for i in range(3) if i != ax]
        rad = np.hypot(cen[:, other[0]] - o[0], cen[:, other[1]] - o[1])
        faces = np.nonzero((cen[:, ax] >= lo) & (cen[:, ax] <= hi) & (rad < rmax) & (rad > rmax * 0.55))[0]
        if not len(faces):
            continue
        name = f"ring_{link}"
        sub = UsdGeom.Subset.CreateGeomSubset(m, name, UsdGeom.Tokens.face, [int(f) for f in faces],
                                              familyName="materialBind")
        UsdGeom.Subset.SetFamilyType(m, "materialBind", UsdGeom.Tokens.nonOverlapping)
        _bind(sub.GetPrim(), alu)
        link_xf[link] = len(faces)
    for link, c, n, dia in BADGES:
        path = f"{robot}/{link}/XarmBadge"
        q = UsdGeom.Mesh.Define(stage, path)
        n = np.asarray(n, float)
        u = np.cross(n, [0, 1, 0] if abs(n[1]) < 0.9 else [1, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        c = np.asarray(c, float) + n * 0.0003
        h, N = dia / 2, 48                    # a disc: a fan of N triangles
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        ring = [c + (u * np.cos(t) + v * np.sin(t)) * h for t in ang]
        q.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in [c] + ring])
        q.CreateFaceVertexCountsAttr([3] * N)
        q.CreateFaceVertexIndicesAttr([i for k in range(N) for i in (0, 1 + k, 1 + (k + 1) % N)])
        q.CreateNormalsAttr([Gf.Vec3f(*map(float, n))] * (N + 1))
        q.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        st = UsdGeom.PrimvarsAPI(q).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.vertex)
        st.Set([Gf.Vec2f(0.5, 0.5)] + [Gf.Vec2f(0.5 + 0.5 * np.cos(t), 0.5 + 0.5 * np.sin(t)) for t in ang])
        _bind(q.GetPrim(), badge)
    return link_xf
