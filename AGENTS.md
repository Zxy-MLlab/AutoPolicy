# AutoPolicy workspace rules

- Keep every new dataset, cache, environment, model, checkpoint, temporary file, and run under
  the repository root. In this workspace that root is `/data/zxy/autopolicy`.
- Preserve provenance labels: never present `mock` output as simulation, archived artifacts as new
  reconstruction, or simulation as a real-robot result.
- Do not modify files in `vendor/` directly; use adapters in `src/autopolicy` or `scripts`.
- Do not enable external deployment without explicit operator approval, an emergency-stop mechanism, and
  robot-specific safety limits.
- Run tests with cache and temporary variables redirected to this project.
