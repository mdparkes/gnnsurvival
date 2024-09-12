"""
Portions of this code are forked from https://github.com/haiderstats/survival_evaluation, which is licensed under
the MIT License:
---
Copyright (c) 2020 Humza Haider

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
---
"""
import numpy as np
import torch

from collections import OrderedDict
from dataclasses import InitVar, dataclass, field
from torch import Tensor
from torch_geometric.nn.aggr import Set2Set
from torch_geometric.nn.conv import SAGEConv, GATv2Conv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.pool import SAGPooling
from torchmtlr import MTLR
from typing import Optional


class SparseMLP(torch.nn.Module):
    """
    An MLP with sparse connections from the input (genes) to the first hidden layer (pathways). Input genes are only
    connected to pathway units in the first hidden layer if they participate in that pathway.

    If a pathway is not connected to any inputs (i.e. none of the members of the pathway are among the inputs),
    the output of the corresponding unit in the hidden pathway layer derives entirely from the bias term.
    Conceptually, the bias can account for "unknown contributions" of the pathway despite receiving no input in the
    form of gene expression values from the previous layer.
    """
    def __init__(self, pathway_mask: Tensor) -> None:
        super().__init__()
        self.n_pathways = pathway_mask.shape[0]
        self.n_genes = pathway_mask.shape[1]
        self.pathway = torch.nn.Linear(in_features=self.n_genes, out_features=self.n_pathways, bias=True)
        self.pathway_weight_mask = pathway_mask

    def forward(self, x):
        self.pathway.weight.data = self.pathway.weight.data.mul(self.pathway_weight_mask)
        x = self.pathway(x)
        x = torch.nn.Tanh()(x)  # Liang et al. used ReLU here, but tanh is used to activate the pathway scores in GNN
        return x


class IndividualPathsMPNN(torch.nn.Module):

    def __init__(self, message_passing: str, num_nodes: int, use_sagpool: bool, ratio: Optional[float] = None) -> None:
        if message_passing.lower() not in ["gatv2", "graphsage"]:
            raise ValueError(f"Argument to \"message_passing\" should be \"gatv2\" or \"graphsage\", "
                             f"got message_passing=\"{message_passing}\"")

        super().__init__()
        self._message_passing_type = message_passing.lower()
        self._use_sagpool = use_sagpool
        self._ratio = ratio
        self._num_nodes = num_nodes

        self.input_transform = torch.nn.Linear(in_features=1, out_features=8, bias=True)

        if self._message_passing_type == "graphsage":
            self.conv1 = SAGEConv(in_channels=8, out_channels=8, project=False)
            self.conv2 = SAGEConv(in_channels=8, out_channels=8, project=False)
            self.conv3 = SAGEConv(in_channels=8, out_channels=8, project=False)
        else:  # GATv2
            self.conv1 = GATv2Conv(in_channels=8, out_channels=8, add_self_loops=False)
            self.conv2 = GATv2Conv(in_channels=8, out_channels=8, add_self_loops=False)
            self.conv3 = GATv2Conv(in_channels=8, out_channels=8, add_self_loops=False)

        self.graphnorm2 = GraphNorm(8)
        self.graphnorm3 = GraphNorm(8)

        if self._use_sagpool:
            self.sagpool1 = SAGPooling(in_channels=8, ratio=self._ratio)
            self.sagpool2 = SAGPooling(in_channels=8, ratio=self._ratio)
            self.sagpool3 = SAGPooling(in_channels=8, ratio=self._ratio)

        # self.aggregate = MeanAggregation()
        self.aggregate1 = Set2Set(8, 3)
        self.aggregate2 = Set2Set(8, 3)
        self.aggregate3 = Set2Set(8, 3)

        # The output transform used by Liang et al. was a small MLP; here we perform a simple linear transformation
        self.output_transform = torch.nn.Linear(in_features=48, out_features=1, bias=True)

    def forward(self, x, edge_index, batch):

        x = torch.tanh(self.input_transform(x))

        # GNN Block 1
        x = torch.tanh(self.conv1(x, edge_index))
        if self._use_sagpool:
            x, edge_index, _, batch, perm1, _ = self.sagpool1(x, edge_index, batch=batch)
        else:
            perm1 = None
        x1 = self.aggregate1(x, batch)

        # GNN Block 2
        x = self.graphnorm2(x, batch)
        x = torch.tanh(self.conv2(x, edge_index))
        if self._use_sagpool:
            x, edge_index, _, batch, perm2, _ = self.sagpool2(x, edge_index, batch=batch)
        else:
            perm2 = None
        x2 = self.aggregate2(x, batch)

        # GNN Block 3
        x = self.graphnorm3(x, batch)
        x = torch.tanh(self.conv3(x, edge_index))
        if self._use_sagpool:
            x, edge_index, _, batch, perm3, _ = self.sagpool3(x, edge_index, batch=batch)
        else:
            perm3 = None
        x3 = self.aggregate3(x, batch)

        # Concatenation
        x = torch.cat([x1, x2, x3], dim=-1)
        # Transform to scalar
        x = torch.tanh(self.output_transform(x))

        return x, batch, (perm1, perm2, perm3)


class NeuralNetworkMTLR(torch.nn.Module):

    def __init__(self, num_time_bins: int, in_features: int) -> None:

        super().__init__()
        self.nn_module = torch.nn.Sequential(OrderedDict([
            ("linear1", torch.nn.Linear(in_features=in_features, out_features=512, bias=True)),
            ("relu1", torch.nn.ReLU()),
            ("linear2", torch.nn.Linear(in_features=512, out_features=256, bias=True)),
            ("dropout1", torch.nn.Dropout(0.4)),
            ("relu2", torch.nn.ReLU()),
            ("linear3", torch.nn.Linear(in_features=256, out_features=128, bias=True)),
            ("relu3", torch.nn.ReLU()),
            ("linear4", torch.nn.Linear(in_features=128, out_features=32, bias=True)),
        ]))
        self.mtlr = MTLR(in_features=32, num_time_bins=num_time_bins)

    def forward(self, x):
        # data is a list of DataBatch objects
        x = self.nn_module(x)
        y = self.mtlr(x)
        return y


@dataclass
class KaplanMeier:
    """From https://github.com/haiderstats/survival_evaluation by Humza Haider"""
    event_times: InitVar[np.array]
    event_indicators: InitVar[np.array]
    survival_times: np.array = field(init=False)
    survival_probabilities: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        population_count = np.flip(np.flip(unique_times[1]).cumsum())

        event_counter = np.append(0, unique_times[1].cumsum()[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        self.survival_probabilities = np.empty(population_count.size)
        survival_probability = 1
        counter = 0
        for population, event_num in zip(population_count, events):
            survival_probability *= 1 - event_num / population
            self.survival_probabilities[counter] = survival_probability
            counter += 1

    def predict(self, prediction_times: np.array):
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]

        return probabilities
