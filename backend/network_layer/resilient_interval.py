"""Causal collection at successive post-service RIC callbacks.

An action issued at callback k is held for the next P simulator service ticks.
Rewards sum the existing per-tick reward over those ticks. Nothing here advances
the simulation. The last observation and mask refer to the same decision boundary.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action_idx: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: list
    info: dict


class CausalInterval:
    def __init__(self, env, period, build_state):
        self.env = env
        self.period = period
        self.build_state = build_state
        self.pending = None

    def begin(self, action_idx):
        if self.pending is not None:
            raise RuntimeError("Previous action interval has not completed")
        mask = self.env._action_mask()
        if not 0 <= action_idx < len(mask) or not mask[action_idx]:
            raise AssertionError("Selected action is not feasible at decision boundary")
        obs = self.env._collect_obs()
        state = self.build_state(obs, self.env.prev_action)
        previous = self.env.prev_action
        action = self.env.actions[action_idx]
        self.env._apply_action(action)
        self.env.prev_action = action
        self.pending = dict(state=state.copy(), action_idx=action_idx, action=action,
                            previous=previous, ticks=0, reward=0.0, parts={})

    def observe(self):
        if self.pending is None:
            return None
        p = self.pending
        obs = self.env._collect_obs()
        reward, parts = self.env._compute_reward(obs, p["action"], p["previous"])
        p["ticks"] += 1
        p["reward"] += reward
        for name, value in parts.items():
            p["parts"][name] = p["parts"].get(name, 0.0) + value
        done = (self.env._steps_in_episode + 1 >= self.env._episode_step_limit
                and p["ticks"] >= self.period) or self.env._traces_consumed()
        if p["ticks"] < self.period and not done:
            return None
        self.env._steps_in_episode += 1
        next_state = self.build_state(obs, p["action"])
        next_mask = self.env._action_mask()
        info = dict(episode_id=self.env._current_episode.get("id"), obs=obs,
                    action=p["action"], prev_action=p["previous"], action_idx=p["action_idx"],
                    reward_parts=p["parts"], interval_ticks=p["ticks"])
        transition = Transition(p["state"], p["action_idx"], p["reward"], next_state,
                                done, next_mask, info)
        self.pending = None
        return transition

    def reset(self):
        if self.pending is not None:
            raise RuntimeError("Cannot silently discard an unfinished control interval")
