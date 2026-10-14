"""
LUNA Invariance Verification Suite (Stage 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Minimal reproduction tools for multi-view shadow and physics invariance validation.

Usage:
    >>> from luna.easter_egg import run_reproduction, plot_reproduction
    >>> metrics = run_reproduction()
    >>> plot_reproduction()

CLI:
    $ luna-verify-physics
    $ luna-render-tufte
    $ luna-render-sweep
"""

def run_reproduction(*args, **kwargs):
    from .verify_physics import run_evaluation
    return run_evaluation(*args, **kwargs)


def plot_reproduction(*args, **kwargs):
    from .render_tufte import generate_plot
    return generate_plot(*args, **kwargs)


def plot_sweep(*args, **kwargs):
    from .render_sweep import generate_sweep_plot
    return generate_sweep_plot(*args, **kwargs)


__all__ = ["run_reproduction", "plot_reproduction", "plot_sweep"]
