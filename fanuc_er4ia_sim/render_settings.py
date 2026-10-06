"""
Renderer settings shared by the render scripts.

The RTX real-time path gives almost no specular response on UsdPreviewSurface,
which flattens machined aluminium, black oxide and moulded plastic into the same
matte look. That is exactly the material separation a zero-shot localiser needs,
so the offline renders use path tracing.
"""
import carb


def enable_path_tracing(spp=64, max_bounces=6):
    s = carb.settings.get_settings()
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/spp", spp)
    s.set("/rtx/pathtracing/totalSpp", spp)
    s.set("/rtx/pathtracing/maxBounces", max_bounces)
    s.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", max_bounces)
    s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    s.set("/rtx/ambientOcclusion/enabled", True)
    print(f"[render] path tracing enabled ({spp} spp, {max_bounces} bounces)")


def realtime():
    carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")
