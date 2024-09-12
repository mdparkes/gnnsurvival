import numpy as np
import pandas as pd

from torch import Tensor
from torch_geometric.data import Data, HeteroData
from typing import Any, Dict, List, Set, Tuple, Union

# Custom types for type hints
NodeInfoDict = Dict[str, Dict[str, Any]]
Numeric = Union[float, int, bool]
AdjacencyDict = Dict[str, Union[Set[str], Set[Tuple[str, Tuple[str]]]]]
ArrayLike = Union[List, np.ndarray, Tensor]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, Tensor]
LabeledGraphInputTuple = Tuple[Tuple[List[Union[Data, HeteroData]], float, float], Tuple[float, int]]  # For GNN inputs