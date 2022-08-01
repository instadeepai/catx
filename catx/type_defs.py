try:
    from typing import Any, Dict, TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

from chex import Array, ArrayNumpy

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

NetworkExtras: TypeAlias = Dict[str, Any]
