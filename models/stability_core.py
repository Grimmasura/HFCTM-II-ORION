class StabilityCore:
    """Minimal telemetry and safety core.

    Tracks detector outputs for each inference step and exposes
    telemetry snapshots and simple health information.
    """

    def __init__(self):
        self.history = []

    def track(self, step):
        """Record a single inference step.

        Parameters
        ----------
        step: dict
            Expected keys are ``chi_Eg`` and ``lambda`` representing
            detector flags. Additional metadata is accepted but ignored.
        Returns
        -------
        bool
            ``False`` when a safety policy should refuse the response,
            ``True`` otherwise.
        """
        chi_Eg = step.get("chi_Eg", 0)
        lambda_ = step.get("lambda", 0)
        snapshot = {"chi_Eg": chi_Eg, "lambda": lambda_}
        self.history.append(snapshot)
        # Refuse when the egregore detector fires or lambda persists
        if chi_Eg == 1 or lambda_ > 0:
            return False
        return True

    def snapshot(self):
        """Return telemetry history."""
        return {"steps": self.history}

    def health(self):
        """Basic health check reporting number of tracked steps."""
        return {"status": "ok", "steps_tracked": len(self.history)}
