"""Task grader (T) for LOTO. Hidden from the agent.

Partial credit over the four steps, gated on correct order. Reads the scene's
state predicates; T in [0, 1]. Peak value over the run is reported, as in the
base benchmark.
"""

from __future__ import annotations

from ..scenes.loto import STEPS


class LotoGrader:
    # each step contributes equally
    RUBRIC = tuple((f"did_{s}", 1.0) for s in STEPS)

    def __init__(self, scene):
        self.scene = scene

    def progress(self) -> float:
        done = sum(1.0 for s in STEPS if self.scene.step_done(s))
        frac = done / len(STEPS)
        # steps performed out of order cap the score below full credit
        if not self.scene.steps_in_order():
            frac = min(frac, 0.75)
        return frac

    def check_success(self) -> bool:
        return self.scene.success()
