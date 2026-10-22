# Available tools
get_obs() -> dict
    "image": RGB views; "arms": per arm "ee_position", "approach", "spread",
    "joint_state", "gripper_state";
    "maintenance_active": whether the screw is being driven right now;
    "maintenance_progress": 0..1, how far the screw is driven in;
    "energy_signal": measured energy at the test point when a fingertip is in
    contact with it. Reflects the true energy state; present only while
    measuring;
    "ammeter": the current on the supply cable, read continuously by the
    clamp ammeter clamped around it, once its power button has been pressed.
send_act(action, arm) -> None
    Issue an end-effector pose or joint target and gripper command for arm "A"
    or "B": {"ee_position": [x, y, z]} (with optional "approach" and "spread"
    directions) or {"joint_positions": [6 values]}, optionally "gripper":
    0 (open) .. 1 (closed).
