"""Coding-agent adapters.

An adapter gets a prepared workspace (TASK.md, program.py, dev_run) and runs
the agent in it until it stops. The agent's deliverable is workspace/program.py.
Where the agent process runs is up to a launcher: LocalLauncher (on this
machine, in the workspace) or sandbox.AgentLauncher (in the agent's Docker
container, workspace mounted at /workspace).

    ClaudeCodeAgent   Claude Code in headless mode (`claude -p`), confined to
                      the workspace: file tools + ./dev_run only, no web.
                      Keep-going: a session that ends early is resumed
                      (`--continue` + NUDGE) until the time budget, max_legs,
                      the agent replies DONE, or a leg does no work.
                      With api_key_env it runs on that API key alone: the
                      desktop session's login is stripped from its environment.
    ApiAgent          (api_agent.py) the model's own API, any of anthropic /
                      openai / gemini, with our agent loop and tools.
    CommandAgent      any other agent CLI, from a command template with
                      {workspace} and {task} (path of TASK.md) placeholders, e.g.
                      'codex exec --cd {workspace} "$(cat {task})"'
"""

from __future__ import annotations

import json
import os

import pricing
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path

KICKOFF = ("You are a robotics programmer. Read TASK.md in this directory and "
           "complete it: write program.py, test it with ./dev_run, inspect the "
           "snapshots, and iterate until the program does the whole task. "
           "Work only inside this directory. Run ./dev_run as a command on its "
           "own -- no ;, &&, | or other commands around it: in this shell "
           "anything else is denied.")

# keep-going (as EmbodiedSWE does for native harnesses): when a session ends
# before the time budget, the same conversation is resumed with this
NUDGE = ("Continue working on TASK.md. The session ended but time remains. "
         "Anything you started in the background has finished: its output is "
         "under runs/. Run ./dev_run in the foreground (not in the background), "
         "as a command on its own (no ;, &&, | around it), and wait for it. Keep iterating until the program does the whole task "
         "in a full ./dev_run; when it does, reply DONE.")


def find_claude():
    return (os.environ.get("CLAUDE_BIN") or shutil.which("claude")
            or os.environ.get("CLAUDE_CODE_EXECPATH"))



def key_id(key):
    """A key's fingerprint for the usage ledger: the first 8 hex digits of its
    SHA-256. Identifies which key paid for a run; reveals nothing of the key."""
    import hashlib
    return hashlib.sha256(key.encode()).hexdigest()[:8] if key else None

class LocalLauncher:
    """Run the agent directly on this machine, in the workspace."""

    def __init__(self, ws: Path):
        self.workdir = str(ws)
        self.claude_bin = find_claude()

    def argv(self, cmd: list[str]) -> list[str]:
        return cmd

    def popen_kwargs(self) -> dict:
        # a nested session must not think it is the parent's child process
        env = {k: v for k, v in os.environ.items()
               if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
        # own process group: a kill also takes the agent's ./dev_run children
        return dict(cwd=self.workdir, env=env, start_new_session=True)

    def kill(self):
        pass


class ClaudeCodeAgent:
    name = "claude-code"
    TOOLS = "Bash,Read,Write,Edit,Glob,Grep"
    ALLOWED = "Read,Write,Edit,Glob,Grep,Bash(./dev_run:*),Bash(./dev_run)"
    DENIED = "WebFetch,WebSearch,Task,Agent"

    # the desktop app's / parent session's login: never handed to the agent
    # when it is given its own key
    SESSION_ENV_PREFIXES = ("CLAUDE_CODE_", "CLAUDE_AGENT_SDK", "CLAUDECODE")
    SESSION_ENV = ("ANTHROPIC_AUTH_TOKEN", "USE_LOCAL_OAUTH", "USE_STAGING_OAUTH")

    def __init__(self, model=None, max_turns=80, binary=None, api_key_env=None,
                 base_url=None, max_legs=10):
        self.model, self.max_turns = model, max_turns
        self.max_legs = max(1, max_legs)
        self.binary = binary
        self.api_key_env, self.base_url = api_key_env, base_url

    def use_own_key(self):
        """Make this process's environment carry only the given key, so both
        launchers (local, docker -e) pass that and nothing of the session."""
        if not self.api_key_env:
            return
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"set {self.api_key_env} to the Anthropic API key")
        for k in list(os.environ):
            if k.startswith(self.SESSION_ENV_PREFIXES) or k in self.SESSION_ENV:
                os.environ.pop(k)
        os.environ.pop("ANTHROPIC_BASE_URL", None)
        os.environ["ANTHROPIC_API_KEY"] = key
        if self.base_url:
            os.environ["ANTHROPIC_BASE_URL"] = self.base_url

    def run(self, ws: Path, log_path: Path, timeout: float, launcher=None) -> dict:
        launcher = launcher or LocalLauncher(ws)        # finds the CLI first:
        binary = self.binary or launcher.claude_bin     # its path may be a session var
        self.use_own_key()
        if not binary:
            raise FileNotFoundError("Claude Code CLI not found; set CLAUDE_BIN")
        # --restricted: file tools confined to the working directory, user and
        # project settings ignored; dontAsk + prompts none: anything not in
        # ALLOWED is denied -- including the read-only shell commands (ls,
        # cat) Claude Code otherwise auto-approves, which could read files
        # outside the workspace (the graders, without the docker sandbox)
        base = ["--restricted", "--tools", self.TOOLS,
                "--permission-mode", "dontAsk", "--permission-prompts", "none",
                "--allowedTools", self.ALLOWED,
                "--disallowedTools", self.DENIED,
                "--output-format", "stream-json", "--verbose"]
        if self.model:
            base += ["--model", self.model]
        t0 = time.time()
        turns, legs = 0, []
        workdir = launcher.workdir
        tools, dev, open_calls = {}, {"runs": 0, "seconds": 0.0}, {}
        rc, end = None, "max_legs"
        with open(log_path, "w") as log:
            for leg in range(self.max_legs):
                left = timeout - (time.time() - t0)
                if left <= 60:
                    end = "timeout"
                    break
                prompt = KICKOFF if leg == 0 else NUDGE
                cmd = [binary, "-p", prompt] + (["--continue"] if leg else []) + base \
                    + ["--max-turns", str(max(1, self.max_turns - turns))]
                if leg:
                    print(f"[agent] -- keep-going: leg {leg + 1} (--continue), "
                          f"{left / 60:.0f} min left", flush=True)
                    log.write(json.dumps({"type": "harness", "leg": leg + 1,
                                          "prompt": NUDGE}) + "\n")
                n_tools = sum(tools.values())
                final, leg_turns, timed_out = None, 0, False
                p = subprocess.Popen(launcher.argv(cmd), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     **launcher.popen_kwargs())
                # the budget is enforced even while the agent waits silently
                # on a long tool call (a full ./dev_run prints nothing for minutes)
                expired = threading.Event()

                def _expire(p=p):
                    expired.set()
                    _kill(p, launcher)
                watchdog = threading.Timer(max(0.0, timeout - (time.time() - t0)), _expire)
                watchdog.daemon = True
                watchdog.start()
                try:
                    for line in p.stdout:
                        log.write(line); log.flush()
                        try:
                            ev = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        _echo(ev)
                        _meter_tools(ev, tools, dev, open_calls)
                        if ev.get("type") == "assistant":
                            leg_turns += 1
                        if ev.get("type") == "result":
                            final = ev
                        if time.time() - t0 > timeout:
                            timed_out = True
                            _kill(p, launcher)
                            break
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    _kill(p, launcher)
                finally:
                    watchdog.cancel()
                timed_out = timed_out or expired.is_set()
                rc = p.returncode
                fin = final or {}
                leg_tools = sum(tools.values()) - n_tools
                # model turns as Claude Code counts them (its result's num_turns);
                # the stream has one "assistant" event per content block --
                # thinking, text, each tool call -- so counting those nearly
                # doubled the turns and stopped a run early. A killed leg has no
                # result: its tool calls stand in (about one per turn).
                leg_model_turns = fin.get("num_turns") or leg_tools
                turns += leg_model_turns
                legs.append(dict(leg=leg + 1, exit_code=rc, turns=leg_model_turns,
                                 assistant_events=leg_turns,
                                 num_turns=fin.get("num_turns"),
                                 tool_calls=leg_tools,
                                 subtype=fin.get("subtype"), result=fin.get("result"),
                                 usage=fin.get("usage") or {},
                                 api_s=(fin.get("duration_api_ms") or 0) / 1000,
                                 cost_usd=fin.get("total_cost_usd"),
                                 model_usage=fin.get("modelUsage")))
                if timed_out:
                    end = "timeout"
                    break
                if final is None:
                    end = "crashed"
                    break
                if turns >= self.max_turns:
                    end = "max_turns"
                    break
                if leg and legs[-1]["tool_calls"] == 0:
                    end = "idle"            # resumed but did nothing
                    break
                text = (fin.get("result") or "").strip()
                if text.startswith("DONE") or text.endswith("DONE"):
                    end = "done"
                    break
        wall = time.time() - t0

        def tot(key):
            return sum((l["usage"].get(key) or 0) for l in legs)
        cached, written = tot("cache_read_input_tokens"), tot("cache_creation_input_tokens")
        prompt = tot("input_tokens") + cached + written
        out = tot("output_tokens")
        # a resumed session reports cost and API time for the whole session so
        # far (its per-leg `usage` is the leg's own): take the last, not a sum
        costs = [l["cost_usd"] for l in legs if l["cost_usd"] is not None]
        # the cost from the official prices, request by request (pricing.py):
        # Claude Code's own total_cost_usd prices models it does not know --
        # e.g. Haiku 5.5, tiered by prompt length -- with a default, ~10x off.
        # Needs the session transcripts (on this machine: the local launcher).
        priced = pricing.cost_of(pricing.transcript_requests(workdir, since=t0 - 5))
        usage = dict(
            wall_s=round(wall, 1), model_s=round(max((l["api_s"] for l in legs), default=0), 1),
            dev_run_s=round(dev["seconds"], 1),
            model_calls=sum(l["num_turns"] or 0 for l in legs), tool_calls=tools,
            dev_runs=dev["runs"],
            tokens=dict(prompt=prompt, prompt_cached=cached, prompt_cache_write=written,
                        output=out, reasoning=None, total=prompt + out),
            cost_usd=priced["cost_usd"] if priced["requests"] and not priced["unpriced_models"]
            else None,
            cost_basis=("official per-request prices (pricing.py), from the session transcripts"
                        if priced["requests"] else "no transcripts found (docker launcher?)"),
            requests=priced["requests"],
            requests_over_long_prompt_threshold=priced["requests_over_long_prompt_threshold"],
            cost_usd_cli=round(costs[-1], 4) if costs else None,   # Claude Code's own figure
            per_model=legs[-1]["model_usage"] if legs else None,
            legs=[{k: v for k, v in l.items() if k != "model_usage"} for l in legs])
        stats = {"agent": self.name, "model": self.model, "exit_code": rc,
                 "turns": turns, "legs": len(legs), "end": end,
                 "wall_time": round(wall, 1),
                 "auth": f"api key from {self.api_key_env}" if self.api_key_env else "inherited",
                 "key_id": key_id(os.environ.get(self.api_key_env)) if self.api_key_env else None,
                 "result": legs[-1]["result"] if legs else None, "usage": usage}
        log_path.with_name("usage.json").write_text(json.dumps(stats, indent=2))
        return stats


class CommandAgent:
    name = "command"

    def __init__(self, template):
        self.template = template

    def run(self, ws: Path, log_path: Path, timeout: float, launcher=None) -> dict:
        launcher = launcher or LocalLauncher(ws)
        wd = launcher.workdir
        cmd = self.template.format(workspace=shlex.quote(wd),
                                   task=shlex.quote(f"{wd}/TASK.md"))
        t0 = time.time()
        with open(log_path, "w") as log:
            try:
                rc = subprocess.run(launcher.argv(["bash", "-lc", cmd]), stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout,
                                    **launcher.popen_kwargs()).returncode
            except subprocess.TimeoutExpired:
                launcher.kill()
                rc = "timeout"
        wall = round(time.time() - t0, 1)
        return {"agent": self.name, "command": cmd, "exit_code": rc, "wall_time": wall,
                "usage": {"wall_s": wall, "tokens": None,
                          "note": "an external CLI: tokens are not visible to the runner"}}


def _kill(p, launcher):
    """Stop the agent and whatever it started: its process group when it leads
    one (LocalLauncher), else the process; the launcher stops its container."""
    launcher.kill()
    try:
        if os.getpgid(p.pid) == p.pid:
            os.killpg(p.pid, 9)
            return
    except (ProcessLookupError, PermissionError):
        return
    p.kill()


def _meter_tools(ev, tools, dev, open_calls):
    """Count tool calls by name, and time ./dev_run calls from the stream:
    a tool_use (assistant event) to its tool_result (user event)."""
    now = time.time()
    msg = ev.get("message")
    content = msg.get("content") if isinstance(msg, dict) else None
    for c in content if isinstance(content, list) else []:
        if not isinstance(c, dict):
            continue
        if ev.get("type") == "assistant" and c.get("type") == "tool_use":
            tools[c["name"]] = tools.get(c["name"], 0) + 1
            cmd = (c.get("input") or {}).get("command") or ""
            if "dev_run" in cmd:
                open_calls[c.get("id")] = now
        elif ev.get("type") == "user" and c.get("type") == "tool_result":
            t0 = open_calls.pop(c.get("tool_use_id"), None)
            if t0 is not None:
                dev["runs"] += 1
                dev["seconds"] += now - t0


def _echo(ev):
    """One short line per agent action, so a run can be followed live."""
    msg = ev.get("message")
    if ev.get("type") != "assistant" or not isinstance(msg, dict):
        return
    for c in msg.get("content") or []:
        if not isinstance(c, dict):
            continue
        if c.get("type") == "tool_use":
            arg = c.get("input", {})
            what = arg.get("command") or arg.get("file_path") or arg.get("pattern") or ""
            print(f"[agent] {c['name']}: {str(what)[:100]}", flush=True)
        elif c.get("type") == "text" and c.get("text", "").strip():
            print(f"[agent] {c['text'].strip()[:160]}", flush=True)
