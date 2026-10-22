# Clamp ammeter -- field notes (from the site electrician; reference only)

The clamp ammeter on the supply riser is the shop's standard clamp meter. The
robot interface for it is as given in TASK.md.

- It reads the current in the cable it is clamped around, updated every control step. When the neighbouring feeder in the riser is loaded it also picks up a standing induced current of up to 150 on this cable, with this circuit's disconnect open: a reading below 150 means no load on this circuit. The probe at the test point is the reference for zero energy.
- It stays clamped on the riser between jobs; do not unclamp or move it.
- Its battery lasts for a full shift; it does not switch itself off while a
  job is running.
- The display faces up. Its reading in the observation is the same number the
  display shows.
