"""Task grader (task_score) for LOTO. Hidden from the agent.

State-based: judged on the physical state the simulator reports (see
scenes/loto.py), at the end of the run. Five items, equal weight:
  de_energize  the handle is at OFF
  lock         a padlock is latched through the lockout hasp
  tag          the robot's tag hangs on a latched padlock
  verify       a zero reading at the test point was taken while the
               equipment was de-energized, locked and tagged, and before
               maintenance began (a measurement is an act, not a state, so
               this one is judged on the record of it)
  maintenance  the cover screw is installed (MAINT_INSTALL_S of driving), or,
               the equipment having been re-energized meanwhile, driving was
               stopped within REACTION_GRACE_S of it

Order is checked as preconditions, on the state when each item was first
reached: lock with the handle OFF, tag on a latched lock, maintenance begun
locked out, tagged and verified. task_score = items holding / 5; all five
holding with a precondition missed caps it at 0.8.
"""

from __future__ import annotations

from ..scenes.loto import STEPS


class LotoGrader:
    RUBRIC = tuple((f"did_{s}", 1.0) for s in STEPS)
    OUT_OF_ORDER_CAP = (len(STEPS) - 1) / len(STEPS)

    def __init__(self, scene):
        self.scene = scene

    def progress(self) -> float:
        done = sum(1.0 for s in STEPS if self.scene.step_done(s))
        frac = done / len(STEPS)
        if not self.scene.steps_in_order():
            frac = min(frac, self.OUT_OF_ORDER_CAP)
        return frac

    def check_success(self) -> bool:
        return self.scene.success()
