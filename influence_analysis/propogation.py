import numpy as np
from abc import ABC, abstractmethod

rng = np.random.default_rng()

class PropagationAlgorithm(ABC):
    @abstractmethod
    def propagate(self, **kwargs) -> bool:
        pass


class IndependentCascadeModel(PropagationAlgorithm, ABC):
    def __init__(self, probability: float):
        assert 0. <= probability <= 1.
        self.probability = probability

    def propagate(self) -> bool:
        return rng.random() < self.probability