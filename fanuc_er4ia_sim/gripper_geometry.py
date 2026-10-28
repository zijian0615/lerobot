"""Machined AA jaw geometry, independent of Isaac/USD. Dimensions in metres."""

import math
import scene_config as C


def clip_profile(poly, boundary, distal):
    """Clip a profile at a longitudinal material boundary, retaining its shape."""
    out = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        a_in = a[0] >= boundary if distal else a[0] <= boundary
        b_in = b[0] >= boundary if distal else b[0] <= boundary
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = (boundary - a[0]) / (b[0] - a[0])
            out.append((boundary, a[1] + t * (b[1] - a[1])))
    return out


def finger_profile(arc_segments=64):
    """Inner half-circle and beveled outer tip of one aluminum blade."""
    L, W, R = C.FINGER_LENGTH, C.FINGER_WIDTH, C.SCALLOP_R
    center = L * C.SCALLOP_AT
    pts = [(0.0, 0.0), (center - R, 0.0)]
    # Clip the socket at the physical end of the blade. The scaled closed
    # photo shows open hooks, not a full circle followed by touching tips.
    theta_end = (
        math.acos(max(-1.0, min(1.0, (center - L) / R))) if center + R > L else math.pi
    )
    pts += [
        (
            center - R * math.cos(theta_end * k / arc_segments),
            R * math.sin(theta_end * k / arc_segments),
        )
        for k in range(1, arc_segments + 1)
    ]
    if center + R <= L:
        pts += [(L - C.FINGER_EDGE_BREAK, 0.0), (L, C.FINGER_EDGE_BREAK)]
    pts += [(L, C.FINGER_TIP_WIDTH), (L - C.FINGER_TIP_BEVEL, W), (0.0, W)]
    return pts


def triangulate(poly):
    """Triangulate a simple polygon. Needed because the finger profile is
    concave (the scallop), and a naive fan triangulation would fill the pocket
    back in."""
    idx = list(range(len(poly)))
    area = sum(
        poly[i][0] * poly[(i + 1) % len(poly)][1]
        - poly[(i + 1) % len(poly)][0] * poly[i][1]
        for i in range(len(poly))
    )
    if area < 0:
        idx.reverse()

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])) - ((a[1] - o[1]) * (b[0] - o[0]))

    def inside(p, a, b, c):
        d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
        neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (neg and pos)

    tris, guard = [], 0
    while len(idx) > 3 and guard < 5000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 0:
                continue  # reflex vertex, not an ear
            if any(inside(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue  # another vertex inside the ear
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    if len(tris) != len(poly) - 2:
        raise ValueError("Finger polygon could not be fully triangulated")
    return tris
