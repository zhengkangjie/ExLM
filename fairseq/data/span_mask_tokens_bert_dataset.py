# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
from fairseq.data import Dictionary, data_utils

# import random

from . import BaseWrapperDataset, LRUCacheDataset


def generate_random_lengths(K, L_min, L_max, R, rng):
    # Check for feasibility
    total_min = K * L_min
    total_max = K * L_max
    if R < total_min or R > total_max:
        raise ValueError("No solution exists with the given parameters.")

    # Initialize variables
    s = R - K * L_min
    U = L_max - L_min
    x = []

    # Generate K non-negative integers x_i', each ≤ U, sum to s
    for i in range(K):
        rem = K - i - 1  # Remaining positions
        min_remain = 0   # Minimum sum of remaining positions
        max_remain = rem * U  # Maximum sum of remaining positions

        # Calculate lower and upper bounds for current integer
        l_i = max(0, s - max_remain)
        u_i = min(U, s - min_remain)

        # Randomly select current integer within bounds
        # x_i_prime = random.randint(l_i, u_i)
        x_i_prime = rng.integers(low=l_i, high=u_i, endpoint=True)
        x.append(x_i_prime + L_min)
        s -= x_i_prime
    # random.shuffle(x)
    rng.shuffle(x)
    return x

def sample_subsequences(total_length, span_num, masked_length, L_min, L_max, rng):
    assert span_num * L_min <= masked_length <= span_num * L_max
    assert masked_length <= total_length

    lengths = generate_random_lengths(span_num, L_min, L_max, masked_length, rng)
    D = total_length - masked_length
    # sum_shifts = random.randint(min(span_num, D), D)
    sum_shifts = rng.integers(low=min(span_num, D), high=D, endpoint=True)
    if sum_shifts >= span_num:
        shifts = generate_random_lengths(span_num, 1, sum_shifts, sum_shifts, rng)
    else:
        shifts = generate_random_lengths(span_num, 0, sum_shifts, sum_shifts, rng)
    offsets = 0
    # positions = []
    noise_mask = np.full(total_length, False)
    for i in range(span_num):
        start = offsets + shifts[i]
        end = offsets + shifts[i] + lengths[i]
        noise_mask[start : end] = True
        # positions.append((offsets + shifts[i], offsets + shifts[i] + lengths[i] - 1))
        offsets = offsets + shifts[i] + lengths[i]
    # return positions
    return noise_mask


class SpanMaskTokensBERTDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.

    Input items are masked according to the specified masking probability.

    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        bpe: BPE to use for whole-word masking.
        mask_multiple_length : repeat each mask index multiple times. Default
            value is 1.
    """

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=False)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=True)),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        return_masked_tokens: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        mask_multiple_length: int = 1,
        skip_masking: bool = False,
        min_span_length: int = 1,
        max_span_length: int = 10,
    ):
        assert 0.0 < mask_prob < 1.0
        assert mask_multiple_length >= 1

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.mask_multiple_length = mask_multiple_length
        self.skip_masking = skip_masking
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length

        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.seed, self.epoch, index)

    @lru_cache(maxsize=8)
    def __getitem_cached__(self, seed: int, epoch: int, index: int):
        seed = int(hash((seed, epoch, index)) % 1e6)
        rng = np.random.default_rng(seed)
        item = self.dataset[index]
        sz = len(item)

        assert (
            self.mask_idx not in item
        ), "Dataset contains mask_idx (={}), this is not expected!".format(
            self.mask_idx,
        )
        if self.skip_masking:
            return torch.from_numpy(np.copy(item))

        # decide elements to mask
        # mask = np.full(sz, False)
        num_mask = int(
            # add a random number for probabilistic rounding
            self.mask_prob * sz / float(self.mask_multiple_length)
            + rng.random()
        )

        avg_span_len = (self.min_span_length + self.max_span_length) / 2
        num_noise_spans = int(np.round(num_mask / avg_span_len))
        num_noise_spans = max(num_noise_spans, 1)

        max_span_length = self.max_span_length
        min_span_length = self.min_span_length

        if self.min_span_length * num_noise_spans > num_mask:
            num_mask = self.min_span_length * num_noise_spans

        if self.max_span_length * num_noise_spans < num_mask:
            num_mask = self.max_span_length * num_noise_spans
            if num_mask > sz:
                num_mask = sz
                max_span_length = int(np.ceil(num_mask / num_noise_spans))

        mask = sample_subsequences(sz, num_noise_spans, num_mask, min_span_length, max_span_length, rng)

        if self.return_masked_tokens:
            # exit early if we're just returning the masked tokens
            # (i.e., the targets for masked LM training)
            new_item = np.full(len(mask), self.pad_idx)
            new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
            return torch.from_numpy(new_item)

        new_item = np.copy(item)
        new_item[mask] = self.mask_idx

        return torch.from_numpy(new_item)
