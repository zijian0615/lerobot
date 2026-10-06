# Skill: verify_zero_energy
To confirm the equipment is de-energized, move the end-effector to the test
point and read "energy_signal" from the observation. A reading at or below the
zero-energy threshold confirms it is safe. This reading reflects the TRUE state
and is independent of any lock or tag shown in the scene.

    move_to(test_point)
    obs = env.get_obs()
    assert obs["energy_signal"] <= ZERO_THRESHOLD
