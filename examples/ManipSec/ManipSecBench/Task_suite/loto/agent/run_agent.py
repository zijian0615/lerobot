"""Have a coding agent write the LOTO program, then deploy and grade it.

    python run_agent.py --env loto.clean.fanuc                       # Claude Code
    python run_agent.py --env loto.scene.fanuc --model claude-sonnet-5
    python run_agent.py --env loto.env.fanuc --agent-cmd 'codex exec --cd {workspace} "$(cat {task})"'
    python run_agent.py --env loto.env.fanuc --workspace <dir> --skip-agent   # re-grade

API agents (our loop and tools, the model's own API, your key -- see api_agent.py):
    python run_agent.py --env loto.env.fanuc --api anthropic --model claude-opus-5
    python run_agent.py --env loto.env.fanuc --api openai --model <model>
        [--base-url https://my-gateway/v1] [--api-key-env MY_KEY]
    python run_agent.py --env loto.env.fanuc --api gemini --model <model>
Claude Code on an API key instead of the desktop login:
    python run_agent.py --env loto.env.fanuc --api-key-env ANTHROPIC_API_KEY
Token and time usage of every agent run: <run>/usage.json (+ usage.jsonl for
API agents), and "agent.usage" in summary.json.

--sandbox docker (default): every part runs in its own container (see
sandbox.py); this script only needs Python and the docker CLI. Build the
images first (../../../docker/build.sh).
--sandbox none: everything runs on this machine, as before the sandbox; for
debugging only -- the agent can then read the graders. Dev runs and the
deployment use --python (the Isaac Sim interpreter, ~/isaacsim-venv/bin/python).
Run this script itself with a Python that has the provider SDKs and numpy
(~/loto-agent-venv/bin/python); keep the SDKs out of Isaac Sim's environment,
whose pinned packages they upgrade.

Layout of a run:
    <runs-dir>/<env>-<time>/
        workspace/        what the agent sees and edits (see workspace.py)
        agent.log         the agent's transcript
        devsim.log        development simulator (docker)
        proxy.log         egress allowed / denied (docker)
        deploy/           graded rollout: result.json, program.log, sim.log,
                          snapshots, deploy.mp4
        summary.json      agent stats + grading
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import agents as A  # noqa: E402
import sandbox as SB  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="loto.clean.fanuc")
    ap.add_argument("--model", default=None, help="model id (Claude Code --model, or the API agent's model)")
    ap.add_argument("--max-turns", type=int, default=80, help="model turns at most")
    ap.add_argument("--agent-cmd", default=None,
                    help="use another agent CLI; placeholders {workspace} {task}")
    ap.add_argument("--agent-timeout", type=float, default=3 * 3600)
    ap.add_argument("--max-legs", type=int, default=10,
                    help="Claude Code keep-going: resume an early-ended session with "
                         "--continue up to this many sessions in all (1 = off)")
    api = ap.add_argument_group("API agent (or Claude Code on an API key)")
    api.add_argument("--api", choices=["anthropic", "openai", "gemini"], default=None,
                     help="run the API agent on this provider")
    api.add_argument("--api-key-env", default=None,
                     help="env var holding the key (default ANTHROPIC_API_KEY / OPENAI_API_KEY / "
                          "GEMINI_API_KEY); for Claude Code, use this key, not the desktop login")
    api.add_argument("--base-url", default=None, help="API endpoint (gateway, proxy, compatible server)")
    api.add_argument("--max-output-tokens", type=int, default=16000)
    api.add_argument("--max-total-tokens", type=int, default=None,
                     help="stop the API agent once prompt+output tokens reach this")
    api.add_argument("--effort", default=None,
                     help="anthropic output_config.effort / openai reasoning_effort")
    api.add_argument("--thinking", default="adaptive", choices=["adaptive", "off"],
                     help="anthropic: adaptive thinking on (default) or not sent")
    api.add_argument("--thinking-budget", type=int, default=None, help="gemini thinking budget")
    api.add_argument("--max-tokens-param", default="max_completion_tokens",
                     choices=["max_completion_tokens", "max_tokens"],
                     help="openai: which output-cap parameter the server accepts")
    for k in ("in", "out", "cache-read", "cache-write"):
        api.add_argument(f"--price-{k}", type=float, default=None,
                         help=f"USD per million {k} tokens, for the cost estimate")
    ap.add_argument("--runs-dir", default=os.path.expanduser("~/loto_agent_runs"))
    ap.add_argument("--workspace", default=None, help="reuse an existing workspace")
    ap.add_argument("--skip-agent", action="store_true", help="only deploy + grade")
    ap.add_argument("--dev-feedback", default="none", choices=["none", "pass", "eval"],
                    help="eval: each ./dev_run also runs program.py once in the evaluation cell "
                         "(the true scene, a held-out layout, no deployment attack) and reports "
                         "PASS / NO PASS (whole task done, nothing unsafe); pass: the old "
                         "development-scene check (task.dev_check)")
    ap.add_argument("--deploy-backend", default="isaac", choices=["isaac", "kinematic"])
    ap.add_argument("--spp", type=int, default=1,
                    help="render quality (path-tracing samples per pixel, OptiX-denoised: "
                         "render_settings.enable_path_tracing) of BOTH the agent's ./dev_run "
                         "and the deployment: the program must see the same images it was "
                         "developed on")
    ap.add_argument("--sandbox", default="docker", choices=["docker", "none"])
    sb = ap.add_argument_group("docker sandbox")
    sb.add_argument("--sim-image", default=None,
                    help="default: the task's own image, manipsec-sim-<task>:latest")
    sb.add_argument("--agent-image", default="manipsec-agent:latest")
    sb.add_argument("--gpus", default="all",
                    help="docker --gpus for the simulators; '' for none (kinematic only)")
    sb.add_argument("--allow-domain", action="append", default=None,
                    help=f"egress allowlist (repeatable); default {', '.join(SB.DEFAULT_ALLOW)}")
    sb.add_argument("--pass-env", action="append", default=None,
                    help=f"env vars handed to the agent (repeatable); default "
                         f"{', '.join(SB.DEFAULT_PASS_ENV)}")
    sb.add_argument("--memory", default="16g", help="memory limit of the agent's containers")
    nb = ap.add_argument_group("no sandbox")
    nb.add_argument("--python", default=sys.executable,
                    help="interpreter with isaacsim, for dev runs and deployment")
    return ap.parse_args()


def deploy_local(args, ws: Path, out: Path) -> dict:
    sub = out / "submitted_program.py"
    shutil.copy(ws / "program.py", sub)
    cmd = [args.python, str(HERE / "program_runner.py"), "--program", str(sub),
           "--env", args.env, "--phase", "deploy", "--backend", args.deploy_backend,
           "--spp", str(args.spp), "--video-stride", "3", "--out", str(out),
           "--timeout", str(SB.DEPLOY_WALL_S)]
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
    with open(out / "sim.log", "w") as log:
        try:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                           timeout=SB.DEPLOY_WALL_S + 600)
        except subprocess.TimeoutExpired:
            return {"error": "deploy timed out, see sim.log"}
    res = out / "result.json"
    return json.loads(res.read_text()) if res.exists() else {"error": "deploy failed, see sim.log"}


def main():
    args = parse()
    regrade = bool(args.workspace and args.skip_agent)
    if args.workspace and not regrade:
        run = Path(args.workspace).resolve().parent
        ws = Path(args.workspace).resolve()
    else:
        # a re-grade (e.g. the clean run's program on the env condition's
        # deployment) is a run of its own: it never overwrites the source run
        base = Path(args.runs_dir) / f"{args.env}-{time.strftime('%Y%m%d-%H%M%S')}"
        run, k = base, 1
        while True:                        # runs started in the same second get -2, -3, ...
            try:
                run.mkdir(parents=True)
                break
            except FileExistsError:
                k += 1
                run = base.with_name(f"{base.name}-{k}")
        ws = Path(args.workspace).resolve() if regrade else run / "workspace"
    run.mkdir(parents=True, exist_ok=True)

    sb = None
    if args.sandbox == "docker" and args.dev_feedback != "none":
        sys.exit("--dev-feedback needs --sandbox none for now (the sandbox's dev_client lacks it)")
    if args.sandbox == "docker":
        sb = SB.DockerSandbox(run, args.env, sim_image=args.sim_image or SB.sim_image_for(args.env),
                              agent_image=args.agent_image, gpus=args.gpus or None,
                              allow_domains=args.allow_domain or SB.DEFAULT_ALLOW,
                              pass_env=args.pass_env or SB.DEFAULT_PASS_ENV,
                              memory=args.memory)
        if not regrade:
            sb.build_workspace(ws)
    elif not regrade:                      # a re-grade leaves the source workspace as it is
        import workspace as W
        W.build(args.env, ws, args.python, dev_spp=args.spp, dev_feedback=args.dev_feedback)
    print(f"[run] workspace: {ws} (sandbox: {args.sandbox})")
    import tasks
    task = tasks.load(args.env)
    if hasattr(task, "attack_audit"):     # what the agent was given, outside its workspace
        (run / "attack_audit.json").write_text(json.dumps(task.attack_audit(args.env), indent=2))

    agent_stats = None
    if not args.skip_agent:
        if args.agent_cmd:
            agent = A.CommandAgent(args.agent_cmd)
        elif args.api:
            import api_agent as AA
            agent = AA.ApiAgent(
                args.api, args.model, api_key_env=args.api_key_env, base_url=args.base_url,
                max_turns=args.max_turns, max_total_tokens=args.max_total_tokens,
                max_tokens=args.max_output_tokens,
                prices=AA.Prices(args.price_in, args.price_out, args.price_cache_read,
                                 args.price_cache_write),
                thinking=args.thinking, effort=args.effort,
                thinking_budget=args.thinking_budget, max_tokens_param=args.max_tokens_param)
        else:
            agent = A.ClaudeCodeAgent(args.model, args.max_turns,
                                      api_key_env=args.api_key_env, base_url=args.base_url,
                                      max_legs=args.max_legs)
        print(f"[run] agent: {agent.name}")
        launcher = None
        try:
            if sb is not None:
                sb.start_dev(ws, isaac=bool(args.gpus))
                launcher = sb.agent_launcher(ws)
            agent_stats = agent.run(ws, run / "agent.log", args.agent_timeout,
                                    launcher=launcher)
        finally:
            if sb is not None:
                sb.stop_dev()
        print(f"[run] agent finished: {json.dumps({k: v for k, v in agent_stats.items() if k != 'usage'})[:300]}")
        u = agent_stats.get("usage") or {}
        if args.price_out is not None and u.get("tokens") and u.get("cost_usd_estimate") is None:
            import api_agent as AA
            meter = AA.UsageMeter(run / "usage.jsonl.unused", AA.Prices(
                args.price_in, args.price_out, args.price_cache_read, args.price_cache_write))
            meter.tok.update({k: u["tokens"].get(k) or 0 for k in meter.tok})
            u["cost_usd_estimate"] = meter.cost_usd() if args.price_in is not None else round(
                meter.tok["output"] * args.price_out / 1e6, 4)
            u["cost_basis"] = "runner, from --price-*" + ("" if args.price_in is not None
                                                           else " (output tokens only)")
        print(f"[run] usage: {json.dumps(u)}")

    out = run / "deploy"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    print(f"[run] deploying program.py on {args.env} ({args.deploy_backend}) ...")
    if sb is not None:
        sub = out / "submitted_program.py"
        shutil.copy(ws / "program.py", sub)
        res = sb.grade(sub, out, backend=args.deploy_backend, spp=args.spp)
    else:
        res = deploy_local(args, ws, out)
    g = res.get("grading", {})
    sub = out / "submitted_program.py"
    if sub.exists():                       # the deployed program, whichever way it ran
        import hashlib
        res["program_sha256"] = hashlib.sha256(sub.read_bytes()).hexdigest()
    video = out / res["video"] if res.get("video") else None
    summary = {"env": args.env, "sandbox": args.sandbox, "agent": agent_stats, "deploy": res,
               "video": str(video) if video else None,
               "dev_runs": len(list((ws / "runs").glob("dev*"))),
               "dev_feedback": args.dev_feedback}
    if regrade:
        summary["program_from"] = str(ws)
    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    extra = {k: g[k] for k in ("steps", "worked_live", "battery_in_metal", "sticker_on", "rewrap_on",
                                "first_violation", "surge_from", "damage", "stages") if k in g}
    print(f"[run] {args.env}: task_score={g.get('task_score')} success={g.get('success')} "
          f"safety_score={g.get('safety_score')} {extra}")
    print(f"[run] summary: {run / 'summary.json'}")


if __name__ == "__main__":
    main()
