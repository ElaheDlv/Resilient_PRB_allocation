"""Small CPU-only masked DQN with versioned joint checkpoints and bounded replay."""

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import random

import numpy as np
import torch
from torch import nn

from .deadline_risk import FEATURE_NAMES
from .joint_dqn_schema import ACTIONS, ACTION_VERSION, LEVELS, LEGACY_RETAINED, STATE_DIM, STATE_VERSION
from .joint_reward import REWARD_VERSION, RewardConfig


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 3456
    replay_capacity: int = 512
    batch_size: int = 16
    gamma: float = 0.99
    learning_rate: float = 0.0001
    target_every: int = 20
    epsilon: float = 0.25

    def __post_init__(self):
        if not 1 <= self.batch_size <= self.replay_capacity <= 4096:
            raise ValueError('Require 1 <= batch <= replay capacity <= 4096')
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError('Invalid discount or exploration probability')
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0 or self.target_every < 1:
            raise ValueError('Invalid learning configuration')


class JointNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(STATE_DIM, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, len(ACTIONS)))

    def forward(self, state):
        return self.net(state)


def masked_td_targets(rewards, dones, next_q, next_masks, gamma):
    if (~next_masks.any(dim=1) & ~dones.bool()).any():
        raise ValueError('A nonterminal next state has no feasible actions')
    best = next_q.masked_fill(~next_masks, -torch.inf).max(dim=1).values
    bootstrap = torch.where(dones.bool(), torch.zeros_like(best), best)
    return rewards + gamma*bootstrap


class JointDQNAgent:
    state_dimension = STATE_DIM
    action_count = len(ACTIONS)
    gradient_clip = 10.0

    def make_network(self):
        return JointNetwork()

    def loss_function(self, prediction, targets):
        return nn.functional.smooth_l1_loss(prediction, targets)

    def __init__(self, config=TrainingConfig(), reward_config=RewardConfig(), observation_config=None):
        self.config = config
        self.reward_config = reward_config
        self.observation_config = dict(observation_config or dict(window_s=0.100, minimum_matured=20))
        torch.set_num_threads(1)
        # Never infer device from GPU availability; all tensors are explicitly CPU.
        self.device = torch.device('cpu')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.policy = self.make_network().to(self.device)
            self.target = self.make_network().to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.replay = deque(maxlen=config.replay_capacity)
        self.rng = random.Random(config.seed)
        self.updates = 0
        self.interactions = 0
        self.last_update = None

    def select(self, state, mask, *, explore=True, epsilon=None):
        epsilon = self.config.epsilon if epsilon is None else epsilon
        if not 0 <= epsilon <= 1 or len(mask) != self.action_count:
            raise ValueError("Invalid epsilon or action-mask shape")
        valid = [i for i, ok in enumerate(mask) if ok]
        if not valid:
            raise ValueError('No feasible joint action')
        if explore and self.rng.random() < epsilon:
            return self.rng.choice(valid)
        with torch.no_grad():
            q = self.policy(torch.as_tensor(state, dtype=torch.float32, device=self.device))
            q = q.masked_fill(~torch.tensor(mask, dtype=torch.bool, device=self.device), -torch.inf)
            return int(q.argmax().item())

    def remember(self, transition):
        if not 0 <= transition.action_idx < self.action_count or not np.isfinite(transition.reward):
            raise ValueError('Invalid transition action/reward')
        if (np.shape(transition.state) != (self.state_dimension,)
                or np.shape(transition.next_state) != (self.state_dimension,)
                or len(transition.next_mask) != self.action_count
                or not np.isfinite(transition.state).all()
                or not np.isfinite(transition.next_state).all()):
            raise ValueError('Invalid replay dimensions or values')
        self.interactions += 1
        self.replay.append((transition.state.copy(), transition.action_idx,
                            transition.reward, transition.next_state.copy(),
                            bool(transition.done), list(transition.next_mask)))

    def learn(self):
        if len(self.replay) < self.config.batch_size:
            return None
        batch = self.rng.sample(list(self.replay), self.config.batch_size)
        states, actions, rewards, next_states, dones, masks = zip(*batch)
        s = torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        ns = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.bool, device=self.device)
        m = torch.tensor(masks, dtype=torch.bool, device=self.device)
        prediction = self.policy(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            targets = masked_td_targets(r, d, self.target(ns), m, self.config.gamma)
        loss = self.loss_function(prediction, targets)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite DQN loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.last_update = dict(loss=float(loss.item()),
            mean_abs_td=float((targets-prediction).detach().abs().mean()),
            max_abs_q=float(prediction.detach().abs().max()),
            max_abs_target=float(targets.abs().max()))
        self.updates += 1
        if self.updates % self.config.target_every == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    def fingerprint(self):
        digest = hashlib.sha256()
        for value in self.policy.state_dict().values():
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def metadata(self):
        return dict(state_schema=dict(version=STATE_VERSION, inputs=STATE_DIM,
                        retained_legacy_indices=LEGACY_RETAINED, risk_names=FEATURE_NAMES,
                        hol_transform='h/(1+h)', validity_inputs=6, exposure_inputs=4),
                    action_schema=dict(version=ACTION_VERSION, actions=98, levels=LEVELS,
                        modes=('PROPORTIONAL', 'FEASIBILITY_AWARE'), order='e,u,mode'),
                    observation_config=self.observation_config,
                    reward_version=REWARD_VERSION, reward_config=asdict(self.reward_config),
                    training_config=asdict(self.config), seed=self.config.seed,
                    device='cpu', checkpoint_purpose='model-and-optimizer; replay/history not saved')

    def save(self, path, *, run_metadata=None):
        torch.save(dict(metadata=self.metadata(), run_metadata=run_metadata or {},
                        policy=self.policy.state_dict(),
                        target=self.target.state_dict(), optimizer=self.optimizer.state_dict(),
                        updates=self.updates), path)

    def load(self, path):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(payload, dict) or payload.get('metadata') != self.metadata():
            raise ValueError('Incompatible joint checkpoint schema/configuration; legacy checkpoints require legacy mode')
        self.policy.load_state_dict(payload['policy'], strict=True)
        self.target.load_state_dict(payload['target'], strict=True)
        self.optimizer.load_state_dict(payload['optimizer'])
        self.updates = int(payload['updates'])
        self.checkpoint_run_metadata = payload.get('run_metadata', {})
