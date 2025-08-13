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
from ..custom_ops import dag_loss, dag_best_alignment, dag_logsoftmax_gather_inplace, torch_dag_loss, torch_dag_best_alignment, torch_dag_logsoftmax_gather_inplace

from .utilities import parse_anneal_argument, get_anneal_value

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

@register_criterion("nat_dag_loss_mlm")
class NATDAGMLMLoss(FairseqCriterion):

    def __init__(self, cfg, task):
        super().__init__(task)
        self.cfg = cfg
        # assert cfg.label_smoothing == 0, "DAG does not support label smoothing"

        # self.set_update_num(0)

        self.bos_idx = task.dictionary.bos()
        self.eos_idx = task.dictionary.eos()
        self.pad_idx = task.dictionary.pad()

        # self.sentence_avg = sentence_avg



    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument("--label-smoothing", type=float, default=0, help="DA-Transformer does not use label smoothing for now")

        parser.add_argument("--torch-dag-logsoftmax-gather", action="store_true", help="Use torch implementation for logsoftmax-gather, which supports GPU and CPU device. (Cuda implementation only supports GPU)")
        parser.add_argument("--torch-dag-best-alignment", action="store_true", help="Use torch implementation for dag-best-alignment, which supports GPU and CPU device. (Cuda implementation only supports GPU)")
        parser.add_argument("--torch-dag-loss", action="store_true", help="Use torch implementation for dag-loss, which supports GPU and CPU device. (Cuda implementation only supports GPU)")

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

    def _compute_dag_loss(self, outputs, output_masks, targets, target_masks, links, label_smoothing=0.0, name="loss",
                factor=1.0, matchmask=None, keep_word_mask=None, model=None):

        batch_size = outputs.shape[0]
        prelen = outputs.shape[1]
        tarlen = targets.shape[1]

        output_length = output_masks.sum(dim=-1)
        target_length = target_masks.sum(dim=-1)

        if self.cfg.torch_dag_logsoftmax_gather:
            outputs, match_all = torch_dag_logsoftmax_gather_inplace(outputs, targets.unsqueeze(1).expand(-1, prelen, -1))
        else:
            outputs, match_all = dag_logsoftmax_gather_inplace(outputs, targets.unsqueeze(1).expand(-1, prelen, -1))
        match_all = match_all.transpose(1, 2)

        if matchmask is not None and not self.cfg.no_force_emit:
            glat_prev_mask = keep_word_mask.unsqueeze(1)
            match_all = match_all.masked_fill(glat_prev_mask, 0) + match_all.masked_fill(~matchmask, float("-inf")).masked_fill(~glat_prev_mask, 0).detach()
        nvalidtokens = output_masks.sum()

        if self.cfg.torch_dag_loss:
            if model.args.max_transition_length != -1:
                links = model.restore_valid_links(links)
            loss_result = torch_dag_loss(match_all, links, output_length, target_length)
        else:
            assert model.args.max_transition_length != -1, "cuda dag loss does not support max_transition_length=-1. You can use a very large number such as 99999"
            loss_result = dag_loss(match_all, links, output_length, target_length)

        invalid_masks = loss_result.isinf().logical_or(loss_result.isnan())
        loss_result.masked_fill_(invalid_masks, 0)
        invalid_nsentences = invalid_masks.sum().detach()

        loss = -(loss_result / target_length).mean()
        nll_loss = loss.detach()
        nsentences, ntokens = targets.shape[0], targets.ne(self.task.dictionary.pad()).sum()

        loss_nofactor = loss
        loss = loss * factor

        return {"name": name, "loss": loss, "nll_loss": nll_loss,
                "factor": factor, "ntokens": ntokens, "nvalidtokens": nvalidtokens, "nsentences": nsentences,
                "loss_nofactor": loss_nofactor, "invalid_nsentences": invalid_nsentences}

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

        sample["net_input"]["ext_mask"] = target.ne(self.pad_idx)

        net_output = model(**sample["net_input"])

        words_logits = net_output[0]
        extra = net_output[1]
        links = extra["links"]

        ext_src_tokens = extra["ext_src_tokens"]
        output_masks = ext_src_tokens.ne(self.pad_idx)

        
        target[src_tokens.eq(self.bos_idx)] = self.bos_idx
        target[src_tokens.eq(self.eos_idx)] = self.eos_idx
        target_valid_mask = target.ne(self.pad_idx)
        new_length = torch.max(torch.sum(target_valid_mask.long(), dim=-1)).item()
        new_pos = torch.cumsum(target_valid_mask.long(), dim=-1) - 1
        new_pos[~target_valid_mask] = new_length
        new_target = target.new_zeros(target.size(0), new_length + 1) + self.pad_idx
        new_target.scatter_(dim=-1, index=new_pos, src=target)

        targets = new_target[:, :-1]
        target_masks = targets.ne(self.pad_idx)



        losses = []

        # DAG loss
        dag_loss = self._compute_dag_loss(
            words_logits,
            output_masks,
            targets,
            target_masks,
            links,
            name="dag-loss",
            factor=1,
            matchmask=None,
            keep_word_mask=None,
            model=model
        )

        loss = dag_loss["loss"]
        # sample_size = (
        #     sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        # )
        sample_size = 1
        logging_output = {
            "loss": dag_loss["nll_loss"].data,
            "nll_loss": dag_loss["nll_loss"].data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
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
