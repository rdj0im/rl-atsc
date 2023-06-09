"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal

TIME_STEPS=1000000
GAMMA=0.99
ALPHA=0.75 # spacial discount base
A=0.5 #tradeoff coeff
lr_actor=5e-4
lr_critic=2.5e-4
Beta=0.01 #regularization coeff
batch_size=120
wait_n=4
wave_n=4
neighbour_n=15*(wait_n+wave_n)
phases_n=2


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)


## might break- upate observation space to reflect list being returned
    # def __call__(self) -> np.ndarray:
    def __call__(self) :
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        wave=self.ts.get_wave()
        wait=self.ts.get_wait()
        # density = self.ts.get_lanes_density()
        # queue = self.ts.get_lanes_queue()
        # observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        observation={'phase':phase_id,'min_green':min_green,'wave':wave,'wait':wait}
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
