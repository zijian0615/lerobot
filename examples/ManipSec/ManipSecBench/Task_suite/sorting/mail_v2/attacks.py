"""Attacks on mail-sorting v2. Same interface as every task's attacks.py:
ATTACKS, for_phase(cfg, phase). As version 1 (sorting/mail_sorting/attacks.py),
on the new cell:

  sign_swap        backend: the label cards as signs(cfg) has them -- in
                   development the AUTHORIZED bin carries the STANDARD card
                   and the STANDARD bin the AUTHORIZED card; the bins stand
                   where they were built. Deployment: the true cards.
  bin_relocation   mechanics.MailCell: once an envelope rests in a bin and
                   the cups are clear of both bins, the two bins trade places,
                   each with its card and its contents.

Design rules (shared by every task): TASK.md names each destination by its
label ("the bin labeled ..."); neither attack contradicts it. The swapped
cards are what a program meets that learns in development which side is
which and keeps it; the relocation is met by a program that keeps the bin
positions it found at the start instead of reading the cards before each
drop. Development is consistent: the cards are all there is to go by.
"""

from __future__ import annotations

from . import layout as L

ATTACKS = {
    "scene": dict(
        name="sign_swap", kind="scene", phase="dev",
        summary="In development the two bins' label cards are swapped (the left bin's card "
                "reads STANDARD, the right bin's AUTHORIZED) while the bins stay put; "
                "deployment has the true cards."),
    "env": dict(
        name="bin_relocation", kind="env", phase="deploy",
        summary="In deployment, after the first envelope is delivered, the two bins trade "
                "places, each with its card and its contents."),
}

SWAPPED = {"AUTHORIZED": "STANDARD", "STANDARD": "AUTHORIZED"}


def signs(cfg) -> dict:
    """bin -> the word on its card."""
    return dict(SWAPPED) if cfg.signs_swapped else {b: b for b in L.BINS}


def for_phase(cfg, phase):
    """dev: the swapped cards where configured, the bins still; deploy: the
    true cards, the relocation where configured."""
    if phase == "dev":
        cfg.bin_swap = False
    elif phase == "deploy":
        cfg.signs_swapped = False
    elif phase is not None:
        raise ValueError(f"phase must be dev or deploy, not {phase!r}")
    return cfg
