"""
Stage 2: scene understanding with Cosmos3-Nano (runs in the cosmos venv).

    /home/driley/cosmos-predict2.5/.venv/bin/python perceive.py --run run/

Reads run/world_camera.png, asks the model to name and locate the graspable
objects and the collection bin, and writes run/detections.json.

Runtime notes, taken from the working AOI setup on this machine:
  - transformers_cosmos3.Cosmos3ForConditionalGeneration, not a Cosmos 2.x path
  - AutoConfig must be loaded explicitly and passed to from_pretrained, to dodge
    a transformers bug where the model_type heuristic replaces the config with
    vision_config and drops text_config.rope_scaling
"""
import argparse, json, os, re, sys
from pathlib import Path

# Machine-specific locations, overridable by environment variable so this runs
# somewhere other than where it was written:
#   COSMOS3_ROOT  checkout of the cosmos framework
#   HF_HOME       HuggingFace cache holding nvidia/Cosmos3-Nano
COSMOS3_ROOT = Path(os.environ.get("COSMOS3_ROOT",
                                   "/home/driley/cosmos3/cosmos-framework"))
COSMOS3_PKG_ROOT = COSMOS3_ROOT / "packages/transformers-cosmos3"
sys.path.insert(0, str(COSMOS3_PKG_ROOT))
sys.path.insert(0, str(COSMOS3_ROOT))
os.environ.setdefault("HF_HOME", "/home/driley/Downloads/HF")


def _find_nano():
    """Locate the Cosmos3-Nano snapshot without hard-coding its hash."""
    env = os.environ.get("COSMOS3_NANO_PATH")
    if env:
        return Path(env)
    base = Path(os.environ["HF_HOME"]) / "hub" / "models--nvidia--Cosmos3-Nano"
    snaps = sorted((base / "snapshots").glob("*")) if base.exists() else []
    if not snaps:
        raise SystemExit(
            f"Cosmos3-Nano not found under {base}.\n"
            "Set HF_HOME to the cache that holds it, or point "
            "COSMOS3_NANO_PATH straight at the snapshot directory.")
    return snaps[-1]


COSMOS3_NANO_PATH = _find_nano()

PROMPT = """You are the vision system for a robot arm working at a bench.

Look at this image of the workbench and report every object on the white table
that the robot could pick up, plus the collection bin.

Report ONLY two kinds of object: small coloured plastic cubes, and black screws.
Do not report the dark fixture with the bottle, the black tape marks and printed
drawings on the table, the table, or the robot.

Reply with ONLY a JSON object, no other text, in exactly this form:

{
  "objects": [
    {"label": "red cube", "category": "cube", "colour": "red",
     "center": [0.42, 0.71], "bbox": [0.40, 0.69, 0.44, 0.73], "graspable": true}
  ],
  "bin": {"label": "collection bin", "center": [0.63, 0.52],
          "bbox": [0.57, 0.46, 0.71, 0.60]}
}

Coordinate rules, follow exactly:
- center and bbox are NORMALISED to the image: x from 0.0 at the left edge to
  1.0 at the right edge, y from 0.0 at the TOP edge to 1.0 at the BOTTOM edge.
- bbox is [x_min, y_min, x_max, y_max].
- "center" must be the centre of the object as it appears in the image.
- category must be exactly "cube" or "screw".
- graspable is true only for small loose objects the gripper could pick up.
- Never repeat the same object twice. Each entry must be a distinct object.
- There are at most 10 objects. Stop and close the JSON when you have listed
  every cube and screw once.
Report every cube and every screw you can see, each as a separate entry."""


def load_model(gpu=0):
    import torch
    from transformers import AutoConfig, AutoProcessor
    from transformers_cosmos3 import Cosmos3ForConditionalGeneration
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    ckpt = str(COSMOS3_NANO_PATH)
    print(f"[perceive] loading Cosmos3-Nano on GPU {gpu} ...", flush=True)
    config = AutoConfig.from_pretrained(ckpt, trust_remote_code=False)
    model = Cosmos3ForConditionalGeneration.from_pretrained(
        ckpt, config=config, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="sdpa")
    model.eval()
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    print("[perceive] model ready", flush=True)
    return model, processor


def ask(model, processor, image_path, prompt, max_new_tokens=1400):
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()
              if isinstance(v, torch.Tensor)}
    with torch.inference_mode():
        # repetition_penalty guards against the degenerate loop this model can
        # fall into, emitting the same object over and over until the budget
        # runs out and the JSON is left unterminated
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, repetition_penalty=1.05)
    new = out[:, inputs["input_ids"].shape[1]:]
    txt = processor.batch_decode(new, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
    return txt[0].strip() if txt else ""


def salvage_objects(text):
    """Recover whatever complete entries exist in a truncated/looping reply.

    Cosmos3-Nano sometimes repeats an object until it runs out of tokens, leaving
    the JSON unterminated. Rather than discarding the whole reply, pull out the
    well-formed {...} blocks and de-duplicate them.
    """
    out, seen = [], set()
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        o = canonicalise(o)
        if "center" not in o or "category" not in o:
            continue
        key = (o.get("label"), tuple(o.get("center", [])))
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
    return out


_ALIASES = {"centre": "center", "color": "colour", "bounding_box": "bbox",
            "box": "bbox", "bbox_2d": "bbox"}


def canonicalise(obj):
    """Normalise key spellings.

    The model mixes conventions inside a single reply -- in one run the cubes
    used "center" and the screws "centre", which silently dropped every screw.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[_ALIASES.get(k, k)] = canonicalise(v)
        return out
    if isinstance(obj, list):
        return [canonicalise(v) for v in obj]
    return obj


def parse_json(text):
    """Pull the first JSON object out of the reply, tolerating fences/prose."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = t.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


VERIFY_PROMPT = """You are the vision system for a robot arm at a bench.

The robot was asked to pick up every small coloured plastic cube and every black
screw from the table and drop them into the collection bin. This image was taken
after it finished.

Reply with ONLY a JSON object, no other text:

{
  "cubes_in_bin": <integer>,
  "cubes_still_on_table": <integer>,
  "screws_in_bin": <integer>,
  "screws_still_on_table": <integer>,
  "bin_contents": ["red cube", "yellow cube", "blue cube", "screw"],
  "task_complete": true,
  "notes": "one short sentence"
}

Count only loose cubes and screws. The dark fixture with the bottle, the black
tape marks and the printed drawings are not screws and are not part of the task."""


INVENTORY_PROMPT = """This image is from a camera mounted on a robot's wrist,
looking into a small yellow collection bin on a workbench.

List what is inside the bin. If any part of the robot's gripper is in the
frame, it is not bin contents. Ignore anything outside the bin.

Reply with ONLY a JSON object, no other text:

{
  "items": [
    {"object": "red cube", "count": 1},
    {"object": "black screw", "count": 2}
  ],
  "total_items": 3,
  "notes": "one short sentence"
}

Use simple names: "red cube", "yellow cube", "blue cube", "black screw".
Count each distinct object once. If something is partly hidden but identifiable,
still count it."""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--image", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="ask whether the task completed, instead of detecting")
    ap.add_argument("--inventory", action="store_true",
                    help="list what is inside the bin (use with the wrist view)")
    a = ap.parse_args()

    run = Path(a.run)
    img = Path(a.image) if a.image else run / "world_camera.png"
    if a.inventory:
        prompt, stem, budget = INVENTORY_PROMPT, "inventory", 600
    elif a.verify:
        prompt, stem, budget = VERIFY_PROMPT, "verify", 600
    else:
        prompt, stem, budget = PROMPT, "detections", 3000
    model, processor = load_model(a.gpu)
    # detection replies are pretty-printed JSON and run long; 8 objects with
    # bboxes overflowed a 1400-token budget mid-object and failed to parse
    raw = ask(model, processor, img, prompt, max_new_tokens=budget)

    (run / f"{stem}_raw.txt").write_text(raw)
    parsed = parse_json(raw)
    if parsed is None and not (a.verify or a.inventory):
        objs = salvage_objects(raw)
        if objs:
            parsed = {"objects": [o for o in objs
                                  if o.get("category") in ("cube", "screw")],
                      "bin": next((o for o in objs
                                   if o.get("category") == "bin"), None)}
            print(f"[perceive] reply was malformed; salvaged "
                  f"{len(parsed['objects'])} complete entries")
    if parsed is None:
        print(f"[perceive] FAILED to parse JSON; raw reply in {stem}_raw.txt")
        print(raw[:800])
        sys.exit(1)
    parsed = canonicalise(parsed)
    with open(run / f"{stem}.json", "w") as f:
        json.dump(parsed, f, indent=2)

    if a.inventory:
        print("[perceive] bin inventory:")
        for it in parsed.get("items", []):
            print(f"   {it.get('count', '?')} x {it.get('object', '?')}")
        print(f"   total: {parsed.get('total_items')}")
        if parsed.get("notes"):
            print(f"   notes: {parsed['notes']}")
    elif a.verify:
        print("[perceive] verification:")
        for k, v in parsed.items():
            print(f"   {k}: {v}")
    else:
        n = len(parsed.get("objects", []))
        print(f"[perceive] parsed {n} objects, bin={'yes' if parsed.get('bin') else 'no'}")
        for o in parsed.get("objects", [])[:15]:
            print(f"   {o.get('label','?'):20s} {o.get('category','?'):8s} "
                  f"center={o.get('center')}")
