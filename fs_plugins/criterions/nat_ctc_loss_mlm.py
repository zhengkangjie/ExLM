##########################################################################
# Copyright (C) 2022 COAI @ Tsinghua University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#         http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

import math
import re
import logging
from functools import reduce
import numpy as np
from typing import Union, Tuple, Optional
import sys

import torch
from torch import Tensor
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch.autograd import Function
# from ..custom_ops import dag_loss, dag_best_alignment, dag_logsoftmax_gather_inplace, torch_dag_loss, torch_dag_best_alignment, torch_dag_logsoftmax_gather_inplace

# from .utilities import parse_anneal_argument, get_anneal_value

from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from dataclasses import dataclass

logger = logging.getLogger(__name__)

########### gpu use tracker ###########
# import inspect
SHOW_MEMORY_USE=False
if SHOW_MEMORY_USE:
    from fairseq.gpu_mem_track import MemTracker
    gpu_tracker = MemTracker()
########################################

# @dataclass
# class CrossEntropyCriterionConfig(FairseqDataclass):
#     sentence_avg: bool = II("optimization.sentence_avg")

@register_criterion("nat_ctc_loss_mlm")
class NATCTCMLMLoss(FairseqCriterion):

    def __init__(self, cfg, task):
        super().__init__(task)
        self.cfg = cfg
        self.ext_state_num = cfg.ext_state_num
        # assert cfg.label_smoothing == 0, "DAG does not support label smoothing"

        # self.set_update_num(0)

        self.bos_idx = task.dictionary.bos()
        self.eos_idx = task.dictionary.eos()
        self.pad_idx = task.dictionary.pad()
        self.unk_idx = task.dictionary.unk()

        # self.sentence_avg = sentence_avg



    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument("--label-smoothing", type=float, default=0, help="DA-Transformer does not use label smoothing for now")


    def _compute_loss(self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = utils.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                        nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss_nofactor = loss
        loss = loss * factor

        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor, "ntokens": outputs.shape[0], "loss_nofactor": loss_nofactor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # import gc
        # gc.collect()
        if SHOW_MEMORY_USE:
            print(torch.cuda.memory_reserved() / 1024 / 1024, file=sys.stderr, flush=True)
            gpu_tracker.clear_cache()
        # gpu_tracker.track()

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        target = sample["target"]
        bsz = target.size(0)

        sample["net_input"]["ext_mask"] = target.ne(self.pad_idx)

        net_output = model(**sample["net_input"])

        words_logits = net_output[0]
        extra = net_output[1]
        links = extra["links"]
        ext_gather_pos = extra["ext_gather_pos"]

        max_len = torch.max(ext_gather_pos).item()

        assert max_len > -1

        max_len = max_len + 1
        max_len = max_len + (self.ext_state_num - max_len % self.ext_state_num) % self.ext_state_num
        
        target[src_tokens.eq(self.bos_idx)] = self.bos_idx
        target[src_tokens.eq(self.eos_idx)] = self.eos_idx
        target_valid_mask = target.ne(self.pad_idx)
        new_length = torch.max(torch.sum(target_valid_mask.long(), dim=-1)).item()
        new_pos = torch.cumsum(target_valid_mask.long(), dim=-1) - 1
        new_pos[~target_valid_mask] = new_length
        new_target = target.new_zeros(target.size(0), new_length + 1) + self.pad_idx
        new_target.scatter_(dim=-1, index=new_pos, src=target)

        targets = new_target[:, :-1]
        # target_masks = targets.ne(self.pad_idx)

        # words_logits: [bsz, seq_len, vocab_size]
        # ext_gather_pos: [bsz, seq_len]

        ext_gather_pos[ext_gather_pos == -1] = max_len

        words_logits = words_logits.float()

        dense_words_logits = words_logits.new_zeros(words_logits.size(0), max_len + 1, words_logits.size(2))
        ext_gather_pos_logits = ext_gather_pos.unsqueeze(-1).expand_as(words_logits)
        dense_words_logits = torch.scatter(input=dense_words_logits, dim=1, index=ext_gather_pos_logits, src=words_logits)
        dense_words_logits = dense_words_logits[:, :-1, :] # [bsz, max_len, vocab_size]

        dense_words_logits = dense_words_logits.reshape(dense_words_logits.size(0), -1, self.ext_state_num, dense_words_logits.size(-1)) # [bsz, max_len // k, k,vocab_size]
        # dense_words_logits = dense_words_logits.permute(0, 2, 1, 3) 
        dense_words_logits = dense_words_logits.reshape(-1, self.ext_state_num, dense_words_logits.size(-1)) # [bsz * max_len // k, k, vocab_size]
        dense_words_logits = torch.log_softmax(dense_words_logits, dim=-1)

        ext_src_tokens = extra["ext_src_tokens"] # [bsz, seq_len]
        dense_src_tokens = ext_src_tokens.new_zeros(ext_src_tokens.size(0), max_len + 1) + self.pad_idx
        dense_src_tokens = torch.scatter(input=dense_src_tokens, dim=1, index=ext_gather_pos, src=ext_src_tokens)
        dense_src_tokens = dense_src_tokens[:, :-1]

        dense_src_tokens = dense_src_tokens.reshape(dense_src_tokens.size(0), -1, self.ext_state_num)
        dense_src_tokens = dense_src_tokens.reshape(-1, self.ext_state_num) # [bsz * max_len // k, k]
        output_masks = dense_src_tokens.ne(self.pad_idx)

        block_num = max_len // self.ext_state_num
        if targets.size(1) > block_num:
            targets = targets[:, :block_num]
        elif targets.size(1) < block_num:
            pad_tgt = targets.new_zeros(targets.size(0), block_num - targets.size(1)) + self.pad_idx
            targets = torch.cat([targets, pad_tgt], dim=1)
        targets = targets.reshape(-1, 1) # [bsz * max_len // k, 1]
        
        blank_idx = self.unk_idx
        targets[targets == blank_idx] = self.pad_idx
        target_masks = targets.ne(self.pad_idx)

        input_length = output_masks.sum(dim=-1)
        target_length = target_masks.sum(dim=-1)

        dense_words_logits = dense_words_logits.permute(1, 0, 2) 

        # print("dense_words_logits size:", dense_words_logits.size())
        # print("targets size:", targets.size())
        # print("input_length size:", input_length.size())
        # print("target_length size:", target_length.size())
        
        loss = F.ctc_loss(log_probs=dense_words_logits, targets=targets, input_lengths=input_length, target_lengths=target_length, blank=blank_idx, reduction='none', zero_infinity=False) # [bsz * block_num]
        loss = loss.masked_fill(target_length == 0, 0)
        loss = loss.reshape(bsz, -1)
        target_length = target_length.reshape(bsz, -1)
        loss = torch.sum(loss, dim=-1) / torch.sum(target_length, dim=-1).float()
        loss = torch.mean(loss)
        # print("loss size:", loss.size())
        # exit()
        # sample_size = (
        #     sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        # )
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": bsz,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "nll_loss", nll_loss_sum / sample_size / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
