from typing import TypeAlias

from chex import ArrayNumpy, Array

Actions: TypeAlias = ArrayNumpy
JaxActions: TypeAlias = Array

Observations: TypeAlias = ArrayNumpy
JaxObservations: TypeAlias = Array

Costs: TypeAlias = ArrayNumpy
JaxCosts: TypeAlias = Array
JaxLoss: TypeAlias = Array

Probabilities: TypeAlias = ArrayNumpy
JaxProbabilities: TypeAlias = Array

Logits: TypeAlias = Array
