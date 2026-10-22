"""Official Claude API prices (USD per million tokens), and the cost of one
request from its usage. From platform.claude.com/docs/en/about-claude/pricing
(checked 2026-10-12); update PRICES when it changes.

Claude Haiku 5.5 is priced by each request's prompt length: a request whose
prompt -- input + cache reads + cache writes -- is over 100,000 tokens pays the
higher prices for all of its tokens, output included. Every other model has one
price at any length. Cache reads are 0.1x the input price except where listed.

Why this exists: Claude Code's own `total_cost_usd` prices models it does not
know with a default (its modelUsage says costBasis "unknown"); for Haiku 5.5
that came out about ten times too high.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

#            input   5m write  1h write  cache read  output
_ROW = ("input", "write_5m", "write_1h", "cache_read", "output")
PRICES = {
    "claude-fable-5-1": (10.0, 12.50, 20.0, 0.25, 50.0),
    "claude-mythos-5-1": (10.0, 12.50, 20.0, 0.25, 50.0),
    "claude-fable-5": (10.0, 12.50, 20.0, 1.00, 50.0),
    "claude-mythos-5": (10.0, 12.50, 20.0, 1.00, 50.0),
    "claude-opus-5-5": (4.0, 5.0, 8.0, 0.20, 20.0),
    "claude-opus-5": (5.0, 6.25, 10.0, 0.50, 25.0),
    "claude-opus-4-8": (5.0, 6.25, 10.0, 0.50, 25.0),
    "claude-opus-4-7": (5.0, 6.25, 10.0, 0.50, 25.0),
    "claude-opus-4-6": (5.0, 6.25, 10.0, 0.50, 25.0),
    "claude-opus-4-5": (5.0, 6.25, 10.0, 0.50, 25.0),
    "claude-sonnet-5-5": (2.0, 2.50, 4.0, 0.10, 10.0),
    "claude-sonnet-5": (2.0, 2.50, 4.0, 0.20, 10.0),
    "claude-sonnet-4-6": (3.0, 3.75, 6.0, 0.30, 15.0),
    "claude-sonnet-4-5": (3.0, 3.75, 6.0, 0.30, 15.0),
    "claude-haiku-4-5": (1.0, 1.25, 2.0, 0.10, 5.0),
}
# prompt-length tiers: (threshold, prices at or under it, prices over it)
TIERED = {
    "claude-haiku-5-5": (100_000, (0.10, 0.125, 0.20, 0.01, 0.50),
                         (0.50, 0.625, 1.00, 0.05, 2.50)),
}


def _base(model: str) -> str:
    """'claude-haiku-4-5-20251001' -> 'claude-haiku-4-5'."""
    m = (model or "").lower()
    for k in sorted(list(PRICES) + list(TIERED), key=len, reverse=True):
        if m == k or m.startswith(k + "-") or m.startswith(k + "["):
            return k
    return m


def known(model: str) -> bool:
    b = _base(model)
    return b in PRICES or b in TIERED


def request_cost(model: str, usage: dict) -> float | None:
    """USD for one request (Anthropic usage dict), or None for an unknown model."""
    b = _base(model)
    inp = usage.get("input_tokens") or 0
    read = usage.get("cache_read_input_tokens") or 0
    write = usage.get("cache_creation_input_tokens") or 0
    cc = usage.get("cache_creation") or {}
    w1 = cc.get("ephemeral_1h_input_tokens") or 0
    w5 = cc.get("ephemeral_5m_input_tokens")
    w5 = write - w1 if w5 is None else w5
    out = usage.get("output_tokens") or 0
    if b in TIERED:
        limit, low, high = TIERED[b]
        p = high if inp + read + write > limit else low
    elif b in PRICES:
        p = PRICES[b]
    else:
        return None
    pi, pw5, pw1, pr, po = p
    return (inp * pi + w5 * pw5 + w1 * pw1 + read * pr + out * po) / 1e6


def transcript_requests(workdir: str | Path, since: float = 0.0) -> list[tuple[str, dict]]:
    """(model, final usage) of every request in the Claude Code session
    transcripts of a working directory (~/.claude/projects/<dir>/*.jsonl),
    written after `since` (epoch s). A request's content blocks are logged as
    separate entries; the one with the most output tokens holds its final usage."""
    key = str(Path(workdir).resolve()).replace("/", "-").replace(".", "-").replace("_", "-")
    files = [f for f in glob.glob(os.path.expanduser(f"~/.claude/projects/{key}/*.jsonl"))
             if os.path.getmtime(f) >= since]
    reqs = {}
    for f in files:
        with open(f, errors="replace") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = e.get("message")
                if e.get("type") != "assistant" or not isinstance(m, dict) or not m.get("usage"):
                    continue
                mid = m.get("id") or e.get("uuid")
                u = m["usage"]
                old = reqs.get(mid)
                if old is None or (u.get("output_tokens") or 0) > (old[1].get("output_tokens") or 0):
                    reqs[mid] = (m.get("model") or "", u)
    return list(reqs.values())


def cost_of(requests) -> dict:
    """Total cost of [(model, usage)], with how many requests were over a
    long-prompt threshold and which models had no price."""
    total, over, unknown = 0.0, 0, set()
    for model, u in requests:
        c = request_cost(model, u)
        if c is None:
            unknown.add(model)
            continue
        total += c
        b = _base(model)
        if b in TIERED:
            limit = TIERED[b][0]
            if (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) \
                    + (u.get("cache_creation_input_tokens") or 0) > limit:
                over += 1
    return {"cost_usd": round(total, 4), "requests": len(requests),
            "requests_over_long_prompt_threshold": over, "unpriced_models": sorted(unknown)}
