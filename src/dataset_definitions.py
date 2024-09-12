import os
import torch

from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data.data import BaseData
from typing import Callable, List, Optional, Tuple, Union, Sequence


class CancerDataset(Dataset):

    def __init__(self, root: str, data_files: Sequence[str], transform: Optional[Callable] = None) -> None:

        super().__init__()

        self.root = root
        self.data_files = [os.path.basename(file) for file in data_files]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx) -> Tuple[Tuple[Tensor, ...], Tuple[float, int]]:
        raw_file_path = os.path.join(self.root, "raw", self.data_files[idx])
        (exprs_tensor, x1, x2), label_data = torch.load(raw_file_path)
        if self.transform:
            exprs_tensor = self.transform(exprs_tensor)
        return (exprs_tensor, x1, x2), label_data


class CancerGraphDataset(GraphDataset):

    def __init__(self, root: str, data_files: Sequence[str], **kwargs):

        super().__init__(root, **kwargs)

        self._data_files = [os.path.basename(file) for file in data_files]

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._data_files

    def len(self) -> int:
        return len(self.raw_file_names)

    def get(self, idx: int) -> BaseData:
        raw_file_path = self.raw_paths[idx]
        data = torch.load(raw_file_path)
        return data
