import torch

from torch import Tensor
from torch_geometric.transforms import BaseTransform
from typing import List

from custom_data_types import LabeledGraphInputTuple


class RangeScaler(torch.nn.Module):
    """Scale Tensor values to a 0-1 range according to where they fall in the min-max range of the columns or rows"""

    def __init__(self, dim: int):
        """
        :param dim: Dimension over which to take the min and max. Must be 0 or 1.
        """
        super().__init__()
        if dim not in [0, 1]:
            raise ValueError(f"`dim` must be 0 or 1, got dim={dim}")
        self.dim = dim

    def forward(self, data: Tensor) -> Tensor:
        new_shape = torch.Size([1, -1]) if self.dim == 0 else torch.Size([-1, 1])
        min_tensor = data.min(dim=self.dim).values.reshape(new_shape)
        max_tensor = data.max(dim=self.dim).values.reshape(new_shape)
        range = max_tensor - min_tensor
        scaled_data = (data - min_tensor) / range
        return scaled_data


class GraphRangeScaler(BaseTransform):
    """Scale Tensor values to a 0-1 range according to where they fall in the min-max range of the columns or rows"""

    def __init__(self, attrs: List[str], dim: int):
        """
        :param attrs: A list of names of attributes to scale
        :param dim: Dimension over which to take the min and max. Must be 0 or 1.
        """
        super().__init__()
        if dim not in [0, 1]:
            raise ValueError(f"`dim` must be 0 or 1, got dim={dim}")
        self.attrs = attrs
        self.dim = dim

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(attrs={self.attrs}, dim={self.dim})"

    def forward(self, data: LabeledGraphInputTuple) -> LabeledGraphInputTuple:
        new_shape = torch.Size([1, -1]) if self.dim == 0 else torch.Size([-1, 1])
        (graph_list, x1, x2), label_data = data
        for graph in graph_list:
            for store in graph.stores:
                for key, value in store.items(*self.attrs):
                    min_tensor = value.min(dim=self.dim).values.reshape(new_shape)
                    max_tensor = value.max(dim=self.dim).values.reshape(new_shape)
                    range = max_tensor - min_tensor
                    store[key] = (value - min_tensor) / range
        feature_data = (graph_list, x1, x2)
        return feature_data, label_data


class StandardizeFeatures(torch.nn.Module):

    def __init__(self, correction: int = 1):
        """
        :param correction: The difference between the sample size and sample degrees of freedom
        """
        super().__init__()
        self.correction = correction

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(correction={self.correction})"

    def forward(self, data: Tensor) -> Tensor:
        std, mean = torch.std_mean(data, dim=0, correction=self.correction, keepdim=True)
        standardized = (data - mean) / std
        return standardized


class StandardizeGraphFeatures(BaseTransform):

    def __init__(self, attrs: List[str], correction: int = 1):
        """
        :param attrs: A list of names of attributes to standardize
        :param correction: The difference between the sample size and sample degrees of freedom
        """
        super().__init__()
        self.attrs = attrs
        self.correction = correction

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(attrs={self.attrs}, correction={self.correction})"

    def forward(self, data: LabeledGraphInputTuple) -> LabeledGraphInputTuple:
        (graph_list, x1, x2), label_data = data
        for graph in graph_list:
            for store in graph.stores:
                for key, value in store.items(*self.attrs):
                    std, mean = torch.std_mean(value, dim=0, correction=self.correction, keepdim=True)
                    store[key] = (value - mean) / std
        feature_data = (graph_list, x1, x2)
        return feature_data, label_data
