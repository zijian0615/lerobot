"""Child process: load the agent's program and call its entry point:
solve(env) if it defines one (the assembly suite's interface), else
main(robot). Both get the same robot_api.Robot.

    python program_host.py program.py      (LOTO_ADDR / LOTO_KEY in the env)
"""

import importlib.util
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robot_api import Robot  # noqa: E402


def main():
    path = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(path.parent))
    robot = Robot()
    try:
        spec = importlib.util.spec_from_file_location("agent_program", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "solve"):
            mod.solve(robot)
        elif hasattr(mod, "main"):
            mod.main(robot)
        else:
            raise AttributeError(f"{path.name} must define solve(env) or main(robot)")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
