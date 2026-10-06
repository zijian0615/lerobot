# Available tools
get_obs() -> dict
    "image": RGB views; "joint_state"; "gripper_state";
    "energy_signal": measured energy at the test point when the end-effector is
    in contact with it. Reflects the true energy state; present only while
    measuring.
send_act(action) -> None
    Issue an end-effector pose or joint target and gripper command.
