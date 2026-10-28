"""Depth from one of the cell's Orbbec cameras, in its own process: robot_server.py starts one per camera, reads stdout.

    uv run --no-project --python 3.12 --with pyorbbecsdk2 python fanuc_er4ia_sim/challenge/depth_reader.py --camera table
    uv run --no-project --python 3.12 --with pyorbbecsdk2 python fanuc_er4ia_sim/challenge/depth_reader.py --camera wrist --params

table: Orbbec Gemini 2 (USB product 0x0670), 640x400. wrist: Gemini 305 (0x0840), 848x480, the wrist colour stream's size.

The depth has to come through Orbbec's SDK: the camera's raw V4L2 16-bit stream is not depth (the SDK converts it). The
SDK runs here, not in the cell server, so a crash in it costs the depth stream (the server restarts this process) and
never the arm. Only the chosen camera is opened; the SDK lists the other but never opens it.

Stream: for each frame, a 24-byte header struct.pack("<4sIdHH", b"DPTH", seq, time.time(), width, height) and then
width * height uint16 depth values in millimetres, row-major (0 = no depth).
--params: print the depth stream's factory intrinsics and its extrinsics to the colour camera, as JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import time

OUT = os.fdopen(os.dup(1), "wb")  # the frames go here; the SDK prints to stdout (fd 1), so that now goes to stderr
os.dup2(2, 1)

import numpy as np  # noqa: E402
import pyorbbecsdk as ob  # noqa: E402

# camera: (USB product, depth size, the colour stream the cell serves: size and format)
CAMERAS = {"table": (0x0670, (640, 400), (1920, 1080)),   # Orbbec Gemini 2
           "wrist": (0x0840, (848, 480), (848, 480))}     # Orbbec Gemini 305: 848x480 is the most it does at 15 fps over USB 2
HEADER = struct.Struct("<4sIdHH")


def open_camera(camera: str):
    ctx = ob.Context()
    ctx.set_logger_level(ob.OBLogLevel.NONE if hasattr(ob.OBLogLevel, "NONE") else ob.OBLogLevel.ERROR)
    ctx.set_uvc_backend_type(ob.OBUvcBackendType.V4L2)  # the video group can open V4L2 nodes; libusb would need root
    devs = ctx.query_devices()
    pid = CAMERAS[camera][0]
    idx = [i for i in range(devs.get_count()) if devs.get_device_pid_by_index(i) == pid]
    if len(idx) != 1:
        raise SystemExit(f"expected one {camera} camera (USB product {pid:#06x}), found {len(idx)}")
    return ctx, devs.get_device_by_index(idx[0])


def depth_profile(pipe, w: int, h: int, fps: int):
    return pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR).get_video_stream_profile(w, h, ob.OBFormat.Y16, fps)


def params(dev, camera: str, w: int, h: int, fps: int) -> dict:
    pipe = ob.Pipeline(dev)
    d = depth_profile(pipe, w, h, fps)
    cw, ch = CAMERAS[camera][2]
    c = pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR).get_video_stream_profile(cw, ch, ob.OBFormat.MJPG, 0)
    k, e = d.get_intrinsic(), d.get_extrinsic_to(c)
    return {"size": [k.width, k.height], "fx": k.fx, "fy": k.fy, "cx": k.cx, "cy": k.cy,
            "depth_to_colour": {"R": np.asarray(e.rot, float).reshape(3, 3).tolist(),
                                "t_m": (np.asarray(e.transform, float) / 1000.0).tolist()},
            "firmware": dev.get_device_info().get_firmware_version()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", choices=sorted(CAMERAS), default="table")
    p.add_argument("--width", type=int, help="default: the camera's depth size above")
    p.add_argument("--height", type=int)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--params", action="store_true", help="print the factory calibration as JSON and exit")
    args = p.parse_args()
    args.width, args.height = args.width or CAMERAS[args.camera][1][0], args.height or CAMERAS[args.camera][1][1]
    _ctx, dev = open_camera(args.camera)
    if args.params:
        OUT.write(json.dumps(params(dev, args.camera, args.width, args.height, args.fps)).encode() + b"\n")
        OUT.flush()
        return
    pipe = ob.Pipeline(dev)
    cfg = ob.Config()
    cfg.enable_stream(depth_profile(pipe, args.width, args.height, args.fps))
    pipe.start(cfg)
    out, seq, idle = OUT, 0, time.time()
    try:
        while True:
            fs = pipe.wait_for_frames(1000)
            f = fs.get_depth_frame() if fs else None
            if f is None:
                if time.time() - idle > 10:
                    raise SystemExit("no depth frame for 10 s")
                continue
            idle = time.time()
            w, h, scale = f.get_width(), f.get_height(), f.get_depth_scale()
            a = np.frombuffer(f.get_data(), np.uint16).reshape(h, w)
            if scale != 1.0:  # the SDK's unit, mm per value; send millimetres
                a = np.clip(a * scale + 0.5, 0, 65535).astype(np.uint16)
            out.write(HEADER.pack(b"DPTH", seq, time.time(), w, h) + a.tobytes())
            out.flush()
            seq += 1
    except BrokenPipeError:
        pass
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
