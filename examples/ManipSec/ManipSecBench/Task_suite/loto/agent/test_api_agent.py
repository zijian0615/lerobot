"""Offline tests of the API agent: no network, no key.

    cd loto/agent && ~/loto-agent-venv/bin/python -m pytest -q test_api_agent.py

Provider tests feed each SDK a response object of its own type (the client
call is replaced) and check the tool calls, the history, and the token
accounting. The loop test drives ApiAgent with a scripted model through a real
workspace: it writes program.py, runs ./dev_run --fast and finishes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import api_agent as AA  # noqa: E402

ISAAC_PY = os.path.expanduser("~/isaacsim-venv/bin/python")
PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
       b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
       b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82")
IMG = AA.ToolResult("image snap.png", image=PNG, mime="image/png")


# ---- providers ---------------------------------------------------------------

def test_anthropic_provider_parses_and_replays():
    import anthropic
    p = AA.AnthropicProvider("claude-opus-5", "k")
    p.start("sys", "task")
    msg = anthropic.types.Message.model_validate(dict(
        id="m1", type="message", role="assistant", model="claude-opus-5",
        stop_reason="tool_use", stop_sequence=None,
        content=[{"type": "text", "text": "plan"},
                 {"type": "tool_use", "id": "tu1", "name": "view_image", "input": {"path": "a.png"}}],
        usage=dict(input_tokens=100, output_tokens=40, cache_read_input_tokens=900,
                   cache_creation_input_tokens=50)))
    seen = {}
    p.client.messages.create = lambda **kw: (seen.update(kw), msg)[1]
    t = p.call()
    assert seen["thinking"] == {"type": "adaptive"} and seen["cache_control"] == {"type": "ephemeral"}
    assert [c.name for c in t.calls] == ["view_image"] and t.text == "plan"
    assert t.usage == dict(prompt=1050, prompt_cached=900, prompt_cache_write=50, output=40, reasoning=0)
    p.add_results([(t.calls[0], IMG)])
    last = p.messages[-1]
    assert last["role"] == "user" and last["content"][0]["type"] == "tool_result"
    assert last["content"][0]["tool_use_id"] == "tu1"
    assert [b["type"] for b in last["content"][0]["content"]] == ["text", "image"]
    assert p.messages[-2]["content"] is msg.content       # replayed unchanged


def test_anthropic_haiku_uses_budget_thinking():
    import anthropic
    p = AA.AnthropicProvider("claude-haiku-4-5", "k", effort="high")
    p.start("sys", "task")
    msg = anthropic.types.Message.model_validate(dict(
        id="m1", type="message", role="assistant", model="claude-haiku-4-5",
        stop_reason="end_turn", stop_sequence=None, content=[{"type": "text", "text": "ok"}],
        usage=dict(input_tokens=10, output_tokens=5)))
    seen = {}
    p.client.messages.create = lambda **kw: (seen.update(kw), msg)[1]
    p.call()
    assert seen["thinking"] == {"type": "enabled", "budget_tokens": 8000}
    assert "output_config" not in seen


def test_openai_provider_parses_and_replays():
    from openai.types.chat import ChatCompletion
    p = AA.OpenAIProvider("gpt-x", "k")
    p.start("sys", "task")
    r = ChatCompletion.model_validate(dict(
        id="c1", object="chat.completion", created=0, model="gpt-x",
        choices=[dict(index=0, finish_reason="tool_calls", message=dict(
            role="assistant", content=None,
            tool_calls=[dict(id="call_1", type="function",
                             function=dict(name="view_image", arguments='{"path": "a.png"}'))]))],
        usage=dict(prompt_tokens=1000, completion_tokens=60, total_tokens=1060,
                   prompt_tokens_details=dict(cached_tokens=800),
                   completion_tokens_details=dict(reasoning_tokens=20))))
    seen = {}
    p.client.chat.completions.create = lambda **kw: (seen.update(kw), r)[1]
    t = p.call()
    assert seen["max_completion_tokens"] == 16000 and seen["tools"][0]["type"] == "function"
    assert t.calls[0].args == {"path": "a.png"}
    assert t.usage == dict(prompt=1000, prompt_cached=800, prompt_cache_write=0, output=60, reasoning=20)
    p.add_results([(t.calls[0], IMG)])
    assert p.messages[-2] == {"role": "tool", "tool_call_id": "call_1", "content": "image snap.png"}
    assert p.messages[-1]["role"] == "user"
    assert p.messages[-1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_gemini_provider_parses_and_replays():
    from google.genai import types
    p = AA.GeminiProvider("gemini-x", "k")
    p.start("sys", "task")
    r = types.GenerateContentResponse.model_validate(dict(
        candidates=[dict(finish_reason="STOP", content=dict(role="model", parts=[
            dict(text="plan"),
            dict(function_call=dict(id="fc1", name="view_image", args={"path": "a.png"}))]))],
        usage_metadata=dict(prompt_token_count=1200, candidates_token_count=50,
                            thoughts_token_count=30, cached_content_token_count=1000)))
    seen = {}
    p.client.models.generate_content = lambda **kw: (seen.update(kw), r)[1]
    t = p.call()
    assert seen["config"].automatic_function_calling.disable is True
    assert t.calls[0].name == "view_image" and t.calls[0].args == {"path": "a.png"}
    assert t.usage == dict(prompt=1200, prompt_cached=1000, prompt_cache_write=0, output=80, reasoning=30)
    p.add_results([(t.calls[0], IMG)])
    last = p.contents[-1]
    assert last.role == "user"
    assert last.parts[0].function_response.name == "view_image"
    assert last.parts[0].function_response.id == "fc1"
    # server ids that look like ours ("call_...") must still be echoed back
    p.add_results([(AA.ToolCall("call_312863", "list_files", {}), AA.ToolResult("x"))])
    assert p.contents[-1].parts[0].function_response.id == "call_312863"
    assert last.parts[1].inline_data.mime_type == "image/png"


# ---- tools ---------------------------------------------------------------------

def test_tools_stay_in_the_workspace(tmp_path):
    (tmp_path / "dev_run").write_text("#!/bin/sh\n")
    tx = AA.ToolExecutor(tmp_path, launcher=None)
    assert tx("read_file", {"path": "../../etc/passwd"}).is_error
    assert tx("write_file", {"path": "/tmp/x", "content": "x"}).is_error
    assert tx("write_file", {"path": "dev_run", "content": "rm -rf /"}).is_error
    assert not tx("write_file", {"path": "program.py", "content": "x = 1\n"}).is_error
    assert tx("edit_file", {"path": "program.py", "old_text": "x = 1", "new_text": "x = 2"}).text == "edited program.py"
    assert tx("nope", {}).is_error


# ---- the loop ---------------------------------------------------------------------

class ScriptedProvider:
    """Plays a fixed sequence of turns, like a model would."""
    name = "scripted"

    def __init__(self, program):
        self.script = [
            ("I'll write the program.", [("write_file", {"path": "program.py", "content": program})]),
            ("Quick kinematic check.", [("run_dev", {"fast": True})]),
            ("", [("list_files", {"path": "runs"})]),
            ("Done.", [("finish", {"summary": "reference program"})]),
        ]
        self.results = []

    def start(self, system, user_text):
        assert "TASK.md" in user_text or "Lockout" in user_text

    def call(self):
        text, calls = self.script.pop(0)
        return AA.Turn(text, [AA.ToolCall(f"c{i}", n, a) for i, (n, a) in enumerate(calls)],
                       dict(prompt=1000, prompt_cached=600, prompt_cache_write=0, output=100,
                            reasoning=10), "tool_use")

    def add_results(self, results):
        self.results.extend(results)

    def nudge(self, text):
        pass


@pytest.mark.skipif(not os.path.exists(ISAAC_PY), reason="needs the Isaac Sim venv for dev runs")
def test_agent_loop_end_to_end(tmp_path, monkeypatch):
    import workspace as W
    ws = W.build("loto.env.fanuc", tmp_path / "workspace", ISAAC_PY)
    program = (HERE / "reference_program.py").read_text()
    scripted = ScriptedProvider(program)
    monkeypatch.setattr(AA, "make_provider", lambda *a, **k: scripted)
    monkeypatch.setenv("FAKE_KEY", "x")
    agent = AA.ApiAgent("openai", "scripted-model", api_key_env="FAKE_KEY",
                        prices=AA.Prices(input=2.0, output=10.0, cache_read=0.2))
    stats = agent.run(ws, tmp_path / "agent.log", timeout=900)

    assert stats["end"] == "finish" and stats["finish_summary"] == "reference program"
    assert (ws / "program.py").read_text() == program
    dev = [r for c, r in scripted.results if c.name == "run_dev"][0]
    assert not dev.is_error and "progress: 1.0" in dev.text      # program.log tail
    u = stats["usage"]
    assert u["model_calls"] == 4 and u["dev_runs"] == {"fast": 1, "full": 0}
    assert u["tool_calls"] == {"write_file": 1, "run_dev": 1, "list_files": 1, "finish": 1}
    assert u["tokens"] == dict(prompt=4000, prompt_cached=2400, prompt_cache_write=0,
                               output=400, reasoning=40, total=4400)
    # 1600 uncached * 2 + 2400 cached * 0.2 + 400 out * 10, per million
    assert u["cost_usd_estimate"] == round((1600 * 2 + 2400 * 0.2 + 400 * 10) / 1e6, 4)
    assert u["dev_run_s"] > 0 and u["wall_s"] >= u["tool_s"]
    lines = [json.loads(l) for l in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert [l["kind"] for l in lines].count("model") == 4
    assert json.loads((tmp_path / "usage.json").read_text())["usage"] == u
