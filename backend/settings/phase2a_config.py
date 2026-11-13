"""Opt-in scripted joint control; original DQN configuration remains untouched."""
import os

JOINT_RIC_ENABLE = os.getenv("JOINT_RIC_ENABLE", "0").lower() in {"1", "true"}
