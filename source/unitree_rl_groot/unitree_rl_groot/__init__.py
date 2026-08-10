"""Unitree G1 RL and GR00T integration.

Task registration is deliberately explicit: runners import
``unitree_rl_groot.tasks`` after Isaac Lab is available. Keeping the package
root lightweight also lets the GR00T bridge run in a separate environment.
"""

__version__ = "0.1.0"
