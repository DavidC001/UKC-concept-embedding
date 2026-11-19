# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Regularizers, all called as reg(factors, model) and combined with --regularizer N3 RootAnchor --reg w1 w2."""
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn

from utils import ROOT_ID

REGULARIZERS = ["N3", "F2", "RootAnchor"]


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor], model: nn.Module):
        """factors: the embeddings of the batch returned by the model; model: the KGModel (for whole-table terms)."""
        pass


def build_regularizers(names, weights, entity_to_id=None, root_target="origin"):
    """One regularizer per name, with the weight at the same position: {name: Regularizer}."""
    if len(names) != len(weights):
        raise ValueError(f"--regularizer {names} needs one --reg weight each, got {weights}")
    if len(set(names)) != len(names):
        raise ValueError(f"--regularizer lists a regularizer twice: {names}")
    regs = {}
    for name, w in zip(names, weights):
        if name == "RootAnchor":
            root = str(ROOT_ID)  # entity_to_id keys are concept ids as strings
            if entity_to_id is None or root not in entity_to_id:
                raise ValueError(f"RootAnchor needs the root entity {root!r} in the dataset's entity_to_id")
            regs[name] = RootAnchor(w, entity_to_id[root], root_target)
        else:
            regs[name] = globals()[name](w)
    return regs


class F2(Regularizer):
    def __init__(self, weight: float):
        super(F2, self).__init__()
        self.weight = weight

    def forward(self, factors, model=None):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(f ** 2)
        return norm / factors[0].shape[0]


class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors, model=None):
        """Regularized complex embeddings https://arxiv.org/pdf/1806.07297.pdf"""
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]


class RootAnchor(Regularizer):
    """
    Pull one entity (the hierarchy root, UKC concept 1 "entity") to a fixed point of the entity space.

    penalty = w * ||e_root - target||^2, with target either
    origin   : the zero vector of the raw entity table
    centroid : the mean of all entity rows, detached, so the root moves to the centroid and the
                centroid is not dragged toward the root
    Graph-only, no LLM. Applied on every training step, on the whole entity table.
    """
    def __init__(self, weight, root_index, target="origin"):
        super(RootAnchor, self).__init__()
        if target not in ("origin", "centroid"):
            raise ValueError("root_target must be 'origin' or 'centroid'")
        self.root = int(root_index)
        self.weight = float(weight)
        self.target = target

    def forward(self, factors, model):
        emb = model.entity.weight
        x = emb[self.root]
        if self.target == "origin":
            diff = x
        else:
            diff = x - emb.mean(dim=0).detach()
        return self.weight * (diff * diff).sum()

