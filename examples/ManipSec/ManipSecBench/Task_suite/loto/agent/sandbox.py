"""Docker sandbox for one agent run: what runs where, and who can reach what.

Development (the agent at work), on an internal network <p>-dev:
    <p>-agent    agent image. The agent CLI, its workspace at /workspace, and
                 the agent's program during ./dev_run. No benchmark code:
                 only the robot API (wire.py, robot_api.py, program_host.py,
                 dev_client.py). Read-only root, no capabilities, non-root.
    <p>-proxy    agent image, egress_proxy.py: HTTPS to the allowed domains
                 (model API, package index) is the only way out.
    <p>-devsim   sim image, dev_server.py: the simulator for ./dev_run, env
                 fixed, phase dev (no environment attack, no scores).

Deployment (grading), on a fresh internal network <p>-grade, after the agent
has stopped:
    <p>-gradesim sim image, program_runner.py --phase deploy --listen: the
                 graded rollout, environment attack on, scene attack off.
    <p>-program  agent image: the submitted program.py, copied in read-only,
                 and nothing else.

The agent's program only ever talks to the simulator over wire.py (JSON), so
it cannot run code in the simulator's container; graders, scene code and
attacks exist only in the sim image. Sim containers with the Isaac backend
are also attached to the default bridge, to fetch NVIDIA's assets.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
from pathlib import Path

SUITE_IN_IMAGE = "/bench/Task_suite"
AGENT_DIR_IN_IMAGE = f"{SUITE_IN_IMAGE}/loto/agent"
API_IN_AGENT_IMAGE = "/opt/loto"
DEV_PORT, PROGRAM_PORT, PROXY_PORT = 7000, 7100, 3128

DEFAULT_ALLOW = ("api.anthropic.com", "pypi.org", "files.pythonhosted.org")
DEFAULT_PASS_ENV = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL",
                    "OPENAI_API_KEY")
# keep Claude Code from calling home past the proxy's allowlist
AGENT_ENV = {"DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1",
             "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1"}

HARDEN = ["--cap-drop", "ALL", "--security-opt", "no-new-privileges",
          "--pids-limit", "1024"]


def docker(*args, check=True, capture=True, timeout=None) -> subprocess.CompletedProcess:
    r = subprocess.run(["docker", *args], check=False, text=True, timeout=timeout,
                       capture_output=capture)
    if check and r.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed ({r.returncode}):\n"
                           f"{(r.stderr or r.stdout or '').strip()[-3000:]}")
    return r


def sim_image_for(env_name: str, tag: str = "latest") -> str:
    """Each task package has its own simulator image (docker/Dockerfile.sim-task,
    on a shared base): loto.* -> manipsec-sim-loto; sorting.* (batteries) ->
    manipsec-sim-sorting_battery; conveyor.* -> manipsec-sim-sorting_conveyor;
    mail_sorting.* and mail_v2.* -> manipsec-sim-sorting_mail; assembly.* (GPU
    and instrument) -> manipsec-sim-assembly."""
    package = {"sorting": "sorting_battery", "conveyor": "sorting_conveyor",
               "mail_sorting": "sorting_mail", "mail_v2": "sorting_mail"}.get(
        env_name.split(".")[0], env_name.split(".")[0])
    return f"manipsec-sim-{package}:{tag}"


# the graded rollout is bounded by sim time (program_runner --max-sim-time);
# its wall clock only guards against a hang: rendering the video makes it
# several times slower than the agent's own dev runs, which must not count
DEPLOY_WALL_S = 4 * 3600

class DockerSandbox:
    def __init__(self, run_dir: Path, env_name: str, *, sim_image: str, agent_image: str,
                 gpus: str | None = "all", allow_domains=DEFAULT_ALLOW,
                 pass_env=DEFAULT_PASS_ENV, isaac_home: Path | None = None,
                 memory: str = "16g"):
        self.run_dir, self.env_name = run_dir, env_name
        self.sim_image, self.agent_image = sim_image, agent_image
        self.gpus, self.allow, self.pass_env = gpus, list(allow_domains), list(pass_env)
        self.memory = memory
        self.isaac_home = isaac_home or Path.home() / ".cache" / "manipsec" / "isaac-home"
        self.p = f"manipsec-{secrets.token_hex(4)}"
        self.user = f"{os.getuid()}:{os.getgid()}"
        self.dev_key = secrets.token_hex(16)
        self._made = []                      # (kind, name) to remove

    # ---- plumbing -----------------------------------------------------------
    def _network(self, suffix: str) -> str:
        name = f"{self.p}-{suffix}"
        docker("network", "create", "--internal", name)
        self._made.append(("network", name))
        return name

    def _sim_args(self) -> list[str]:
        self.isaac_home.mkdir(parents=True, exist_ok=True)
        a = ["--user", self.user, "-e", "HOME=/home/sim", "-e", "OMNI_KIT_ACCEPT_EULA=YES",
             "-v", f"{self.isaac_home}:/home/sim"]
        if self.gpus:
            a += ["--gpus", self.gpus]
        return a

    def _start(self, name: str, network: str, alias: str, args: list[str],
               image: str, cmd: list[str], egress: bool = False):
        self._made.append(("container", name))       # removed on teardown even if
        docker("run", "-d", "--name", name, "--network", network,   # the start fails
               "--network-alias", alias, *args, image, *cmd)
        if egress:
            docker("network", "connect", "bridge", name)

    def _remove(self, kind: str, name: str):
        if kind == "container":
            docker("rm", "-f", name, check=False)
        else:
            docker("network", "rm", name, check=False)

    def teardown(self):
        for kind, name in reversed(self._made):
            if kind == "container":
                self._remove(kind, name)
        for kind, name in reversed(self._made):
            if kind == "network":
                self._remove(kind, name)
        self._made = []

    def _save_logs(self, name: str, path: Path):
        r = docker("logs", name, check=False)
        path.write_text((r.stdout or "") + (r.stderr or ""))

    # ---- workspace ------------------------------------------------------------
    def build_workspace(self, ws: Path):
        """TASK.md is rendered from the prompts and layout, which live only in
        the sim image; the workspace gets the rendered text, not the sources."""
        ws.mkdir(parents=True, exist_ok=True)
        docker("run", "--rm", "--network", "none", "--user", self.user,
               "-v", f"{ws.resolve()}:/ws", self.sim_image,
               "python", f"{AGENT_DIR_IN_IMAGE}/workspace.py",
               "--env", self.env_name, "--out", "/ws", "--sandbox")

    # ---- development ----------------------------------------------------------
    def start_dev(self, ws: Path, isaac: bool = True):
        net = self._network("dev")
        self.dev_net = net
        self._start(f"{self.p}-proxy", net, "proxy", ["--user", self.user, *HARDEN,
                    "--read-only"], self.agent_image,
                    ["python3", f"{API_IN_AGENT_IMAGE}/egress_proxy.py",
                     "--listen", f"0.0.0.0:{PROXY_PORT}",
                     *[x for d in self.allow for x in ("--allow", d)]], egress=True)
        runs = (ws / "runs").resolve()
        runs.mkdir(parents=True, exist_ok=True)
        self._start(f"{self.p}-devsim", net, "devsim",
                    [*self._sim_args(), "-e", f"DEV_KEY={self.dev_key}",
                     "-v", f"{runs}:/runs"], self.sim_image,
                    ["python", f"{AGENT_DIR_IN_IMAGE}/dev_server.py", "--env", self.env_name,
                     "--listen", f"0.0.0.0:{DEV_PORT}", "--program-port", str(PROGRAM_PORT),
                     "--runs", "/runs"], egress=isaac)

    def stop_dev(self):
        for name, log in ((f"{self.p}-devsim", "devsim.log"), (f"{self.p}-proxy", "proxy.log")):
            if ("container", name) in self._made:
                self._save_logs(name, self.run_dir / log)
        self.teardown()

    def agent_launcher(self, ws: Path) -> "AgentLauncher":
        return AgentLauncher(self, ws)

    # ---- deployment -----------------------------------------------------------
    def grade(self, program: Path, out: Path, backend: str = "isaac", spp: int = 4,
              timeout: float = DEPLOY_WALL_S + 600) -> dict:
        net = self._network("grade")
        key = secrets.token_hex(16)
        out.mkdir(parents=True, exist_ok=True)
        sim, prog = f"{self.p}-gradesim", f"{self.p}-program"
        try:
            self._start(sim, net, "sim",
                        [*self._sim_args(), "-e", f"LOTO_KEY={key}",
                         "-v", f"{out.resolve()}:/out"], self.sim_image,
                        ["python", f"{AGENT_DIR_IN_IMAGE}/program_runner.py",
                         "--listen", f"0.0.0.0:{PROGRAM_PORT}", "--env", self.env_name,
                         "--phase", "deploy", "--backend", backend, "--spp", str(spp),
                         "--video-stride", "3", "--out", "/out",
                         "--timeout", str(DEPLOY_WALL_S)],
                        egress=backend == "isaac")
            self._start(prog, net, "program",
                        ["--user", self.user, *HARDEN, "--read-only", "--memory", self.memory,
                         "--tmpfs", "/tmp:rw,exec,size=1g",
                         "-e", f"LOTO_ADDR=sim:{PROGRAM_PORT}", "-e", f"LOTO_KEY={key}",
                         "-e", "PYTHONUNBUFFERED=1",
                         "-v", f"{program.resolve()}:/submission/program.py:ro"],
                        self.agent_image,
                        ["sh", "-c", "mkdir -p /tmp/run && cp /submission/program.py /tmp/run/"
                         " && cd /tmp/run && exec python3 "
                         f"{API_IN_AGENT_IMAGE}/program_host.py program.py"])
            rc = _wait(prog, timeout)
            self._save_logs(prog, out / "program.log")
            if rc is None:
                docker("rm", "-f", prog, check=False)
            # the program is gone: if it never connected, end the runner's wait
            docker("kill", "--signal", "TERM", sim, check=False)
            _wait(sim, 900)
            self._save_logs(sim, out / "sim.log")
        finally:
            self.teardown()
        res = out / "result.json"
        if not res.exists():
            return {"error": "deploy failed, see sim.log", "program_exit_code": rc}
        result = json.loads(res.read_text())
        result["exit_code"] = rc if rc is not None else "killed (timeout)"
        return result


def _wait(name: str, timeout: float):
    """Exit code of a container, or None if it is still running at timeout."""
    try:
        r = docker("wait", name, timeout=timeout)
        return int(r.stdout.strip())
    except subprocess.TimeoutExpired:
        return None


class AgentLauncher:
    """agents.py launcher: the agent CLI in <p>-agent, on the dev network."""

    workdir = "/workspace"
    claude_bin = "claude"

    def __init__(self, sb: DockerSandbox, ws: Path):
        self.sb, self.ws = sb, ws.resolve()
        self.name = f"{sb.p}-agent"

    def argv(self, cmd: list[str]) -> list[str]:
        sb = self.sb
        env = {**AGENT_ENV, "HOME": "/home/agent",
               "HTTPS_PROXY": f"http://proxy:{PROXY_PORT}",
               "HTTP_PROXY": f"http://proxy:{PROXY_PORT}",
               "https_proxy": f"http://proxy:{PROXY_PORT}",
               "http_proxy": f"http://proxy:{PROXY_PORT}",
               "NO_PROXY": "devsim,localhost,127.0.0.1", "no_proxy": "devsim,localhost,127.0.0.1",
               "DEV_SERVER": f"devsim:{DEV_PORT}", "DEV_KEY": sb.dev_key}
        e = [x for k, v in env.items() for x in ("-e", f"{k}={v}")]
        e += [x for k in sb.pass_env if k in os.environ for x in ("-e", k)]   # value from our env
        sb._made.append(("container", self.name))
        return ["docker", "run", "--rm", "--name", self.name, "--network", sb.dev_net,
                "--user", sb.user, *HARDEN, "--read-only", "--memory", sb.memory,
                "--tmpfs", "/tmp:rw,exec,size=4g",
                "--tmpfs", f"/home/agent:rw,exec,size=4g,uid={os.getuid()},gid={os.getgid()},mode=0700",
                "-v", f"{self.ws}:/workspace", "-w", "/workspace", *e,
                sb.agent_image, *cmd]

    def popen_kwargs(self) -> dict:
        return dict(env=os.environ)

    def kill(self):
        docker("rm", "-f", self.name, check=False)
