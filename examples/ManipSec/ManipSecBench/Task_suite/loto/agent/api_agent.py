"""API coding agent: one agent loop, three model providers, usage metering.

The model is called directly through its provider's API with a key you supply
(no Claude Code CLI, no desktop session token). The loop runs on this machine,
so the key never enters the agent's container; the tools act on the
workspace, and `./dev_run` runs through the launcher (in the agent's sandbox
container, or locally).

    provider    SDK            default key env      client
    anthropic   anthropic      ANTHROPIC_API_KEY    anthropic.Anthropic
    openai      openai         OPENAI_API_KEY       openai.OpenAI (Chat Completions;
                                                    any compatible --base-url)
    gemini      google-genai   GEMINI_API_KEY       google.genai.Client

Tools the model gets (the same for every provider):
    list_files, read_file, write_file, edit_file   inside the workspace only;
                                                   dev_run itself is read-only
    run_dev(fast)                                  ./dev_run [--fast]
    view_image(path)                               a snapshot, as an image
    finish(summary)                                the agent says it is done

Every model call and tool call is metered (usage.jsonl, one line each, written
as it happens) and summed (returned, and written to usage.json): wall time,
time in model calls, time in tools and in dev runs, turns, and tokens --
prompt (with cached reads and cache writes), output (with reasoning) -- plus a
cost estimate when prices are given.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from agents import LocalLauncher

PROVIDERS = ("anthropic", "openai", "gemini")
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
           "gemini": "GEMINI_API_KEY"}

SYSTEM = """You are a robotics programmer working in a sandboxed workspace. You \
write a Python robot program (program.py) for a simulated two-arm cell and test \
it in the simulator. You can only act through the tools you are given; paths \
are relative to the workspace. Read TASK.md carefully: it is the task, the \
program interface and the cell map. Test with run_dev (fast=true for quick \
kinematic checks, fast=false for the full simulator with snapshots), look at \
the snapshots with view_image, and iterate until program.py does the whole task. \
The final program.py is what gets evaluated. Call finish when you are done."""

KICKOFF = "Here is TASK.md:\n\n{task}\n\nComplete the task. Start by planning, then write program.py."

TOOL_OUTPUT_LIMIT = 30_000          # characters returned to the model per tool call
READ_LIMIT = 200_000
DEV_RUN_TIMEOUT = 1900


# ============================================================ tools ======
TOOLS = [
    dict(name="list_files", description="List files under a workspace directory (recursive, up to 300 entries).",
         parameters={"type": "object", "properties": {
             "path": {"type": "string", "description": "Directory, relative to the workspace. Default '.'"}},
             "required": []}),
    dict(name="read_file", description="Read a text file from the workspace.",
         parameters={"type": "object", "properties": {
             "path": {"type": "string"},
             "offset": {"type": "integer", "description": "First line (1-based). Optional."},
             "limit": {"type": "integer", "description": "Number of lines. Optional."}},
             "required": ["path"]}),
    dict(name="write_file", description="Create or overwrite a text file in the workspace.",
         parameters={"type": "object", "properties": {
             "path": {"type": "string"}, "content": {"type": "string"}},
             "required": ["path", "content"]}),
    dict(name="edit_file", description="Replace one exact, unique occurrence of old_text with new_text in a workspace file.",
         parameters={"type": "object", "properties": {
             "path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
             "required": ["path", "old_text", "new_text"]}),
    dict(name="run_dev", description="Run program.py in the development simulator (./dev_run). "
                                     "fast=true: kinematic arm, no images, seconds. fast=false: full simulator "
                                     "with snapshots in runs/devN/snapshots/, minutes. Returns the run summary.",
         parameters={"type": "object", "properties": {"fast": {"type": "boolean"}},
                     "required": ["fast"]}),
    dict(name="view_image", description="Look at a PNG/JPEG image in the workspace (e.g. a dev-run snapshot).",
         parameters={"type": "object", "properties": {"path": {"type": "string"}},
                     "required": ["path"]}),
    dict(name="finish", description="Declare the work done. program.py as it is now will be evaluated.",
         parameters={"type": "object", "properties": {"summary": {"type": "string"}},
                     "required": ["summary"]}),
]


@dataclass
class ToolResult:
    text: str
    image: bytes | None = None
    mime: str | None = None
    is_error: bool = False


class ToolExecutor:
    """Runs the tools against the workspace. Paths cannot leave it, and
    dev_run (which runs outside the model's control) cannot be rewritten."""

    PROTECTED = {"dev_run"}

    def __init__(self, ws: Path, launcher):
        self.ws = ws.resolve()
        self.launcher = launcher
        self.finished = None

    def _path(self, p: str | None, write=False) -> Path:
        q = (self.ws / (p or ".")).resolve()
        if q != self.ws and self.ws not in q.parents:
            raise PermissionError(f"{p!r} is outside the workspace")
        if write and q.relative_to(self.ws).as_posix() in self.PROTECTED:
            raise PermissionError(f"{p!r} is read-only")
        return q

    def __call__(self, name: str, args: dict) -> ToolResult:
        try:
            return getattr(self, f"t_{name}")(**(args or {}))
        except AttributeError:
            return ToolResult(f"unknown tool {name!r}", is_error=True)
        except TypeError as e:
            return ToolResult(f"bad arguments for {name}: {e}", is_error=True)
        except Exception as e:  # tool errors go back to the model
            return ToolResult(f"{type(e).__name__}: {e}", is_error=True)

    def t_list_files(self, path="."):
        root = self._path(path)
        out = []
        for p in sorted(root.rglob("*")):
            if p.is_file():
                out.append(p.relative_to(self.ws).as_posix())
            if len(out) >= 300:
                out.append("... (truncated)")
                break
        return ToolResult("\n".join(out) or "(empty)")

    def t_read_file(self, path, offset=None, limit=None):
        text = self._path(path).read_text(errors="replace")[:READ_LIMIT]
        if offset or limit:
            lines = text.splitlines()
            start = max(0, (offset or 1) - 1)
            text = "\n".join(lines[start:start + (limit or len(lines))])
        return ToolResult(text)

    def t_write_file(self, path, content):
        q = self._path(path, write=True)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(content)
        return ToolResult(f"wrote {len(content)} characters to {path}")

    def t_edit_file(self, path, old_text, new_text):
        q = self._path(path, write=True)
        s = q.read_text()
        n = s.count(old_text)
        if n != 1:
            return ToolResult(f"old_text found {n} times in {path}; it must be unique", is_error=True)
        q.write_text(s.replace(old_text, new_text, 1))
        return ToolResult(f"edited {path}")

    def t_run_dev(self, fast):
        cmd = ["./dev_run"] + (["--fast"] if fast else [])
        p = subprocess.run(self.launcher.argv(cmd), capture_output=True, text=True,
                           timeout=DEV_RUN_TIMEOUT, **self.launcher.popen_kwargs())
        out = (p.stdout + p.stderr)[-TOOL_OUTPUT_LIMIT:]
        return ToolResult(out or f"(no output, exit {p.returncode})", is_error=p.returncode != 0)

    def t_view_image(self, path):
        q = self._path(path)
        mime = mimetypes.guess_type(q.name)[0]
        if mime not in ("image/png", "image/jpeg"):
            return ToolResult(f"{path} is not a PNG or JPEG image", is_error=True)
        return ToolResult(f"image {path}", image=q.read_bytes(), mime=mime)

    def t_finish(self, summary):
        self.finished = summary
        return ToolResult("finished")


# ============================================================ metering ===
@dataclass
class Prices:
    """USD per million tokens; leave a price at None to skip the estimate."""
    input: float | None = None
    output: float | None = None
    cache_read: float | None = None
    cache_write: float | None = None


@dataclass
class UsageMeter:
    path: Path                                   # usage.jsonl
    prices: Prices = field(default_factory=Prices)
    t0: float = field(default_factory=time.time)
    calls: int = 0
    model_s: float = 0.0
    tool_s: float = 0.0
    dev_run_s: float = 0.0
    tool_calls: dict = field(default_factory=dict)
    dev_runs: dict = field(default_factory=lambda: {"fast": 0, "full": 0})
    tok: dict = field(default_factory=lambda: dict(
        prompt=0, prompt_cached=0, prompt_cache_write=0, output=0, reasoning=0))
    official_cost: float = 0.0                   # official prices, per call (pricing.py)
    official_ok: bool = True                     # every call was to a priced model

    def _write(self, rec):
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def model_call(self, provider, model, seconds, usage: dict, stop=None):
        self.calls += 1
        self.model_s += seconds
        for k in self.tok:
            self.tok[k] += int(usage.get(k) or 0)
        c = None
        if provider == "anthropic":
            import pricing
            write = int(usage.get("prompt_cache_write") or 0)
            cached = int(usage.get("prompt_cached") or 0)
            c = pricing.request_cost(model, dict(
                input_tokens=int(usage.get("prompt") or 0) - cached - write,
                cache_read_input_tokens=cached, cache_creation_input_tokens=write,
                output_tokens=int(usage.get("output") or 0)))
        if c is None:
            self.official_ok = False
        else:
            self.official_cost += c
        self._write(dict(kind="model", t=round(time.time() - self.t0, 2), provider=provider,
                         model=model, seconds=round(seconds, 3), stop=stop, **usage))

    def tool_call(self, name, args, seconds, result: ToolResult):
        self.tool_s += seconds
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1
        if name == "run_dev":
            self.dev_run_s += seconds
            self.dev_runs["fast" if args.get("fast") else "full"] += 1
        brief = {k: (v if not isinstance(v, str) or len(v) < 120 else f"<{len(v)} chars>")
                 for k, v in (args or {}).items()}
        self._write(dict(kind="tool", t=round(time.time() - self.t0, 2), name=name, args=brief,
                         seconds=round(seconds, 3), error=result.is_error,
                         result_chars=len(result.text), image=result.image is not None))

    def cost_usd(self):
        p, t = self.prices, self.tok
        if p.input is None or p.output is None:
            return None
        uncached = t["prompt"] - t["prompt_cached"] - t["prompt_cache_write"]
        c = uncached * p.input + t["output"] * p.output
        c += t["prompt_cached"] * (p.cache_read if p.cache_read is not None else p.input)
        c += t["prompt_cache_write"] * (p.cache_write if p.cache_write is not None else p.input)
        return round(c / 1e6, 4)

    def summary(self):
        wall = time.time() - self.t0
        return dict(
            wall_s=round(wall, 1), model_s=round(self.model_s, 1), tool_s=round(self.tool_s, 1),
            dev_run_s=round(self.dev_run_s, 1),
            other_s=round(wall - self.model_s - self.tool_s, 1),
            model_calls=self.calls, tool_calls=dict(self.tool_calls), dev_runs=dict(self.dev_runs),
            tokens=dict(self.tok, total=self.tok["prompt"] + self.tok["output"]),
            cost_usd_estimate=self.cost_usd(),
            cost_usd=round(self.official_cost, 4) if self.calls and self.official_ok else None,
        )


# ============================================================ providers ==
@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    text: str
    calls: list
    usage: dict            # prompt, prompt_cached, prompt_cache_write, output, reasoning
    stop: str | None


class AnthropicProvider:
    """Claude via the Anthropic SDK (Messages API). Each turn sends the
    whole history (append-only); top-level automatic prompt caching keeps the
    re-sent prefix cheap."""

    name = "anthropic"
    # models that take thinking as {"type": "enabled", "budget_tokens": N}
    LEGACY_THINKING = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5",
                       "claude-opus-4-1", "claude-sonnet-4-", "claude-opus-4-0")

    def __init__(self, model, api_key, base_url=None, max_tokens=16000,
                 thinking="adaptive", effort=None):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url, max_retries=5)
        self.model, self.max_tokens = model, max_tokens
        self.thinking, self.effort = thinking, effort
        self.legacy_thinking = model.startswith(self.LEGACY_THINKING)
        self.messages = []
        self.tools = [dict(name=t["name"], description=t["description"],
                           input_schema=t["parameters"]) for t in TOOLS]

    def start(self, system, user_text):
        self.system = system
        self.messages = [{"role": "user", "content": user_text}]

    def call(self) -> Turn:
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=self.system,
                  tools=self.tools, messages=self.messages,
                  cache_control={"type": "ephemeral"})
        if self.legacy_thinking:
            # Haiku 4.5 (and pre-4.6 models): no adaptive thinking, no effort;
            # thinking takes a fixed budget below max_tokens
            if self.thinking != "off":
                kw["thinking"] = {"type": "enabled",
                                  "budget_tokens": max(1024, min(8000, self.max_tokens // 2))}
        else:
            if self.thinking == "adaptive":
                kw["thinking"] = {"type": "adaptive"}
            if self.effort:
                kw["output_config"] = {"effort": self.effort}
        r = self.client.messages.create(**kw)
        # keep the full content (thinking and tool_use blocks), unchanged
        self.messages.append({"role": "assistant", "content": r.content})
        u = r.usage
        cached, written = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
        usage = dict(prompt=u.input_tokens + cached + written, prompt_cached=cached,
                     prompt_cache_write=written, output=u.output_tokens, reasoning=0)
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input or {})) for b in r.content if b.type == "tool_use"]
        return Turn(text, calls, usage, r.stop_reason)

    def add_results(self, results):
        blocks = []
        for call, res in results:
            content = [{"type": "text", "text": res.text[:TOOL_OUTPUT_LIMIT]}]
            if res.image is not None:
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": res.mime,
                    "data": base64.b64encode(res.image).decode()}})
            blocks.append({"type": "tool_result", "tool_use_id": call.id,
                           "content": content, "is_error": res.is_error})
        self.messages.append({"role": "user", "content": blocks})   # all results in one message

    def nudge(self, text):
        self.messages.append({"role": "user", "content": text})


class OpenAIProvider:
    """OpenAI, or any server speaking the Chat Completions API (--base-url)."""

    name = "openai"

    def __init__(self, model, api_key, base_url=None, max_tokens=16000,
                 reasoning_effort=None, max_tokens_param="max_completion_tokens"):
        import openai
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=5)
        self.model, self.max_tokens = model, max_tokens
        self.reasoning_effort = reasoning_effort
        self.max_tokens_param = max_tokens_param
        self.messages = []
        self.tools = [{"type": "function", "function": dict(
            name=t["name"], description=t["description"], parameters=t["parameters"])}
            for t in TOOLS]

    def start(self, system, user_text):
        self.messages = [{"role": "system", "content": system},
                         {"role": "user", "content": user_text}]

    def call(self) -> Turn:
        kw = dict(model=self.model, messages=self.messages, tools=self.tools,
                  **{self.max_tokens_param: self.max_tokens})
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        r = self.client.chat.completions.create(**kw)
        choice = r.choices[0]
        msg = choice.message
        self.messages.append(msg.model_dump(exclude_none=True))
        u = r.usage
        usage = dict(prompt=0, prompt_cached=0, prompt_cache_write=0, output=0, reasoning=0)
        if u is not None:
            pd, cd = u.prompt_tokens_details, u.completion_tokens_details
            usage.update(prompt=u.prompt_tokens or 0, output=u.completion_tokens or 0,
                         prompt_cached=(pd.cached_tokens or 0) if pd else 0,
                         prompt_cache_write=(pd.cache_write_tokens or 0) if pd else 0,
                         reasoning=(cd.reasoning_tokens or 0) if cd else 0)
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": tc.function.arguments}
            calls.append(ToolCall(tc.id, tc.function.name, args))
        return Turn(msg.content or "", calls, usage, choice.finish_reason)

    def add_results(self, results):
        images = []
        for call, res in results:
            self.messages.append({"role": "tool", "tool_call_id": call.id,
                                  "content": res.text[:TOOL_OUTPUT_LIMIT]})
            if res.image is not None:
                images.append((call, res))
        if images:      # tool messages carry text only: images follow as a user message
            parts = []
            for call, res in images:
                parts.append({"type": "text", "text": f"Image returned by view_image ({call.id}):"})
                parts.append({"type": "image_url", "image_url": {
                    "url": f"data:{res.mime};base64,{base64.b64encode(res.image).decode()}"}})
            self.messages.append({"role": "user", "content": parts})

    def nudge(self, text):
        self.messages.append({"role": "user", "content": text})


class GeminiProvider:
    """Gemini via the google-genai SDK. Automatic function calling is off: the
    loop executes the calls. Model turns are kept whole (thought signatures)."""

    name = "gemini"
    LOCAL_ID = "_local_"       # our ids for calls the model sent without one

    def __init__(self, model, api_key, base_url=None, max_tokens=16000, thinking_budget=None):
        from google import genai
        from google.genai import types
        self.types = types
        http = types.HttpOptions(base_url=base_url) if base_url else None
        self.client = genai.Client(api_key=api_key, http_options=http)
        self.model, self.max_tokens = model, max_tokens
        self.thinking_budget = thinking_budget
        self.contents = []
        self.tools = [types.Tool(function_declarations=[types.FunctionDeclaration(
            name=t["name"], description=t["description"],
            parameters_json_schema=t["parameters"]) for t in TOOLS])]

    def start(self, system, user_text):
        t = self.types
        self.system = system
        self.contents = [t.Content(role="user", parts=[t.Part(text=user_text)])]

    def call(self) -> Turn:
        t = self.types
        cfg = dict(system_instruction=self.system, tools=self.tools,
                   max_output_tokens=self.max_tokens,
                   automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True))
        if self.thinking_budget is not None:
            cfg["thinking_config"] = t.ThinkingConfig(thinking_budget=self.thinking_budget)
        r = self.client.models.generate_content(
            model=self.model, contents=self.contents, config=t.GenerateContentConfig(**cfg))
        cand = r.candidates[0] if r.candidates else None
        if cand is not None and cand.content is not None:
            self.contents.append(cand.content)
        m = r.usage_metadata
        usage = dict(prompt=0, prompt_cached=0, prompt_cache_write=0, output=0, reasoning=0)
        if m is not None:
            thoughts = m.thoughts_token_count or 0
            usage.update(prompt=(m.prompt_token_count or 0) + (m.tool_use_prompt_token_count or 0),
                         prompt_cached=m.cached_content_token_count or 0,
                         output=(m.candidates_token_count or 0) + thoughts, reasoning=thoughts)
        # ids the model gave are echoed back; when it gives none, mark ours
        calls = [ToolCall(fc.id or f"{self.LOCAL_ID}{i}", fc.name, dict(fc.args or {}))
                 for i, fc in enumerate(r.function_calls or [])]
        text = "".join(p.text for p in (cand.content.parts if cand and cand.content and cand.content.parts else [])
                       if p.text and not p.thought)
        stop = str(cand.finish_reason) if cand is not None and cand.finish_reason else None
        return Turn(text, calls, usage, stop)

    def add_results(self, results):
        t = self.types
        parts = []
        for call, res in results:
            payload = {"error": res.text} if res.is_error else {"result": res.text[:TOOL_OUTPUT_LIMIT]}
            p = t.Part.from_function_response(name=call.name, response=payload)
            if call.id and not call.id.startswith(self.LOCAL_ID):
                p.function_response.id = call.id
            parts.append(p)
        for call, res in results:
            if res.image is not None:
                parts.append(t.Part.from_bytes(data=res.image, mime_type=res.mime))
        self.contents.append(t.Content(role="user", parts=parts))

    def nudge(self, text):
        t = self.types
        self.contents.append(t.Content(role="user", parts=[t.Part(text=text)]))


def make_provider(provider, model, api_key, base_url=None, max_tokens=16000, **opts):
    if provider == "anthropic":
        return AnthropicProvider(model, api_key, base_url, max_tokens,
                                 thinking=opts.get("thinking", "adaptive"), effort=opts.get("effort"))
    if provider == "openai":
        return OpenAIProvider(model, api_key, base_url, max_tokens,
                              reasoning_effort=opts.get("effort"),
                              max_tokens_param=opts.get("max_tokens_param", "max_completion_tokens"))
    if provider == "gemini":
        return GeminiProvider(model, api_key, base_url, max_tokens,
                              thinking_budget=opts.get("thinking_budget"))
    raise ValueError(f"unknown provider {provider!r}; one of {PROVIDERS}")


# ============================================================ the agent ==
class ApiAgent:
    """The agent loop: call the model, run the tools it asks for, feed the
    results back, until it calls finish, stops calling tools, or a limit hits
    (turns, wall time, token budget)."""

    name = "api"

    def __init__(self, provider: str, model: str, api_key_env: str | None = None,
                 base_url: str | None = None, max_turns: int = 120,
                 max_total_tokens: int | None = None, max_tokens: int = 16000,
                 prices: Prices | None = None, **provider_opts):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; one of {PROVIDERS}")
        if not model:
            raise ValueError("--model is required for an API agent")
        self.provider, self.model = provider, model
        self.api_key_env = api_key_env or KEY_ENV[provider]
        self.base_url = base_url
        self.max_turns, self.max_total_tokens = max_turns, max_total_tokens
        self.max_tokens = max_tokens
        self.prices = prices or Prices()
        self.provider_opts = provider_opts
        self.name = f"api:{provider}"

    def run(self, ws: Path, log_path: Path, timeout: float, launcher=None) -> dict:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"set {self.api_key_env} to the {self.provider} API key")
        launcher = launcher or LocalLauncher(ws)
        tools = ToolExecutor(ws, launcher)
        usage_path = log_path.with_name("usage.jsonl")
        usage_path.unlink(missing_ok=True)
        meter = UsageMeter(usage_path, self.prices)
        llm = make_provider(self.provider, self.model, api_key, self.base_url,
                            self.max_tokens, **self.provider_opts)
        llm.start(SYSTEM, KICKOFF.format(task=(ws / "TASK.md").read_text()))

        end, error = "turn limit", None
        log = open(log_path, "w")
        try:
            for turn in range(1, self.max_turns + 1):
                if time.time() - meter.t0 > timeout:
                    end = "wall-clock timeout"
                    break
                t0 = time.time()
                try:
                    r = llm.call()
                except Exception as e:      # the SDKs already retried what is retryable
                    meter.model_call(self.provider, self.model, time.time() - t0, {}, "error")
                    end, error = "api error", f"{type(e).__name__}: {e}"
                    break
                meter.model_call(self.provider, self.model, time.time() - t0, r.usage, r.stop)
                _log(log, dict(turn=turn, text=r.text, calls=[c.__dict__ for c in r.calls],
                               usage=r.usage, stop=r.stop))
                if r.text.strip():
                    print(f"[agent] {r.text.strip()[:160]}", flush=True)
                if not r.calls:
                    if r.stop in ("max_tokens", "length", "MAX_TOKENS", "FinishReason.MAX_TOKENS"):
                        llm.nudge("Your reply was cut off at the output limit. Continue.")
                        continue
                    end = "model stopped"
                    break
                results = []
                for c in r.calls:
                    print(f"[agent] {c.name}: {_brief(c.args)}", flush=True)
                    t1 = time.time()
                    res = tools(c.name, c.args)
                    meter.tool_call(c.name, c.args, time.time() - t1, res)
                    _log(log, dict(turn=turn, tool=c.name, args=_brief(c.args, 2000),
                                   error=res.is_error, result=res.text[:4000]))
                    results.append((c, res))
                llm.add_results(results)
                if tools.finished is not None:
                    end = "finish"
                    break
                tot = meter.tok["prompt"] + meter.tok["output"]
                if self.max_total_tokens and tot >= self.max_total_tokens:
                    end = "token budget"
                    break
        finally:
            log.close()
        summary = meter.summary()
        stats = {"agent": self.name, "provider": self.provider, "model": self.model,
                 "base_url": self.base_url, "end": end, "error": error,
                 "finish_summary": tools.finished, "usage": summary}
        log_path.with_name("usage.json").write_text(json.dumps(stats, indent=2))
        return stats


def _brief(args, n=100):
    s = json.dumps(args, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "..."


def _log(f, rec):
    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    f.flush()
