from typing import Tuple, Callable

import numpy as np
from torch import Tensor

from src.utils.observations.obs import Observation


# === Observation Types are defined in the utils.observations module. ===

# === Action Types === (for further information on the action space, see the guide in doc/ .)
ActionRaw = Tensor
ActionSedi = Tensor

# === Training Types ===
Reward = Tuple[Tensor, Tensor]  # a tensor of reward values, paired with a tensor of observer IDs of matching shape
ValueTensor = Tensor
LogprobTensor = Tensor  # double precision
EntropyTensor = Tensor  # double precision
InferenceTimeTensor = Tensor

ReturnTensor = Tensor  # double precision
AdvantageTensor = Tensor  # double precision
RatioTensor = Tensor  # double precision


# === Function Types ===

# A step result includes:
# - obs (for all agents incl. birdseye view),
# - reward (a tensor of reward values, paired with a tensor of observer IDs of matching shape),
# - done (= truncated, since our ns3 sim never reaches a "terminal" state but is cut off after a certain time),
# - infos dict
StepResult = Tuple[Observation, Reward, bool, dict]  # (obs, reward, done, infos)

# env step function receives actions for the entire network as selected edge destination indices (sedi),
# and returns the classical RL step result. It converts the action sedi to a suitable graph shape using the latest obs.
StepFunc = Callable[[ActionSedi], StepResult]

# env reset function returns the initial observation (global, local).
# It can optionally reset the RNG and seed it with a new value.
ResetFunc = Callable[..., Observation]  # optional input:: reset_rng: bool, new_rng_seed: int

# function to obtain an action from the trainable algorithms or baseline.
# Receives the observation (type 'Observation') as first arg, and any other things needed (e.g. env and shortest-path provider).
GetActionFunc = Callable[..., Tuple[ActionRaw, ActionSedi]]

# function to assemble the given (potentially local) action components into a global shape, using the provided obs.
AssembleActionFunc = Callable[[ActionRaw, ActionSedi, Observation], Tuple[ActionRaw, ActionSedi]]


# === Other Types ===
TrafficMatrix = np.ndarray  # traffic matrix (2D numpy array)
