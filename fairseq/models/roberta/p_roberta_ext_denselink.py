# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, ESM2TransformerEncoder
from fairseq.modules import LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import safe_getattr, safe_hasattr

from .hub_interface import RobertaHubInterface

from fairseq.utils import new_arange

from fairseq.modules import (
    PositionalEmbedding,
    LayerNorm,
)

logger = logging.getLogger(__name__)

from torch import Tensor, nn, jit

@jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))

@register_model("p_roberta_ext_denselink")
class ESM2RobertaExtDenseLinkModel(FairseqEncoderModel):
    @classmethod
    def hub_models(cls):
        return {
            "roberta.base": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz",
            "roberta.large": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz",
            "roberta.large.mnli": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz",
            "roberta.large.wsc": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz",
        }

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
        self.ext_state_num = args.ext_state_num
        self.token_dropout = args.token_dropout

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # super().load_state_dict(state_dict, strict=False, **kwargs)
        super().load_state_dict(state_dict, strict=True, **kwargs)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        parser.add_argument(
            "--untie-weights-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        # args for "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al. 2020)
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )
        parser.add_argument(
            "--token-dropout",
            action="store_true",
            default=False,
            help="Apply token dropout",
        )
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            "--min-params-to-wrap",
            type=int,
            metavar="D",
            default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=(
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            ),
        )
        # args for AdaPruning
        # In short, it adds regularizarion for the multihead attention module and feed forward neural nets
        # For more details, please refer to the paper https://openreview.net/forum?id=_CMSV7FTzGI
        parser.add_argument(
            "--mha-reg-scale-factor",
            type=float,
            metavar="D",
            default=0.0,
            help="scaling factor for regularization term in adptive pruning, recommendation is 0.000375",
        )
        parser.add_argument(
            "--ffn-reg-scale-factor",
            type=float,
            metavar="D",
            default=0.0,
            help="scaling factor for regularization term in adptive pruning, recommendation is 0.000375",
        )
        parser.add_argument(
            "--mha-heads-to-keep",
            type=int,
            metavar="D",
            default=-1,
            help="number of heads to keep in each multi-head attention module, -1 means keeping all heads",
        )
        parser.add_argument(
            "--ffn-blocks-to-remove",
            type=int,
            metavar="D",
            default=-1,
            help="number of feedforward blocks to remove in each transformer layer, -1 means keeping all ffn blocks",
        )
        parser.add_argument(
            "--ext-state-num",
            type=int,
            metavar="D",
            default=4,
            help="number of extended states",
        )
        parser.add_argument('--links-feature', type=str, default="feature:position", help="Features used to predict transition.")
        parser.add_argument('--max-transition-length', type=int, default=99999, help="Max transition distance. -1 means no limitation, \
                        which cannot be used for cuda custom operations. To use cuda operations with no limitation, please use a very large number such as 99999.")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        from omegaconf import OmegaConf

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, False)

        # make sure all arguments are present
        p_base_architecture(args)

        if not safe_hasattr(args, "max_positions"):
            if not safe_hasattr(args, "tokens_per_sample"):
                args.tokens_per_sample = task.max_positions()
            args.max_positions = args.tokens_per_sample

        encoder = ESM2RobertaEncoder(args, task.source_dictionary)

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, True)

        return cls(args, encoder)

    def forward(
        self,
        src_tokens,
        ext_mask=None,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        need_head_weights=False, 
        return_contacts=False,
        require_links=True,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True

        x, extra = self.encoder(src_tokens, ext_mask, features_only, return_all_hiddens, token_dropout=self.token_dropout, need_head_weights=need_head_weights, return_contacts=return_contacts, require_links=require_links, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def _get_adaptive_head_loss(self):
        norm_loss = 0
        scaling = float(self.args.mha_reg_scale_factor)
        for layer in self.encoder.sentence_encoder.layers:
            norm_loss_layer = 0
            for i in range(layer.self_attn.num_heads):
                start_idx = i * layer.self_attn.head_dim
                end_idx = (i + 1) * layer.self_attn.head_dim
                norm_loss_layer += scaling * (
                    torch.sum(
                        torch.abs(
                            layer.self_attn.q_proj.weight[
                                start_idx:end_idx,
                            ]
                        )
                    )
                    + torch.sum(
                        torch.abs(layer.self_attn.q_proj.bias[start_idx:end_idx])
                    )
                )
                norm_loss_layer += scaling * (
                    torch.sum(
                        torch.abs(
                            layer.self_attn.k_proj.weight[
                                start_idx:end_idx,
                            ]
                        )
                    )
                    + torch.sum(
                        torch.abs(layer.self_attn.k_proj.bias[start_idx:end_idx])
                    )
                )
                norm_loss_layer += scaling * (
                    torch.sum(
                        torch.abs(
                            layer.self_attn.v_proj.weight[
                                start_idx:end_idx,
                            ]
                        )
                    )
                    + torch.sum(
                        torch.abs(layer.self_attn.v_proj.bias[start_idx:end_idx])
                    )
                )

            norm_loss += norm_loss_layer
        return norm_loss

    def _get_adaptive_ffn_loss(self):
        ffn_scale_factor = float(self.args.ffn_reg_scale_factor)
        filter_loss = 0
        for layer in self.encoder.sentence_encoder.layers:
            filter_loss += torch.sum(
                torch.abs(layer.fc1.weight * ffn_scale_factor)
            ) + torch.sum(torch.abs(layer.fc2.weight * ffn_scale_factor))
            filter_loss += torch.sum(
                torch.abs(layer.fc1.bias * ffn_scale_factor)
            ) + torch.sum(torch.abs(layer.fc2.bias * ffn_scale_factor))
        return filter_loss

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = RobertaClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )

    @property
    def supported_targets(self):
        return {"self"}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )

        logger.info(x["args"])
        return RobertaHubInterface(x["args"], x["task"], x["models"][0])

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder") :]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # rename emb_layer_norm -> layernorm_embedding
        for k in list(state_dict.keys()):
            if ".emb_layer_norm." in k:
                new_k = k.replace(".emb_layer_norm.", ".layernorm_embedding.")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v

            # adapt data2vec models
            if (
                "encoder._ema" in state_dict
                and "encoder.lm_head.weight" not in state_dict
            ):
                lm_state = self.encoder.lm_head.state_dict()
                for k, v in lm_state.items():
                    state_dict["encoder.lm_head." + k] = v

            for k in list(state_dict.keys()):
                if k.startswith("encoder.regression_head") or k == "encoder._ema":
                    del state_dict[k]


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ESM2RobertaEncoder(FairseqEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        

        # set any missing default values
        p_base_architecture(args)
        self.args = args
        self.bos_idx = dictionary.bos()
        self.eos_idx = dictionary.eos()
        self.pad_idx = dictionary.pad()

        self.ext_state_num = args.ext_state_num

        self.init_link_feature(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        embed_tokens = self.build_embedding(
            len(dictionary), args.encoder_embed_dim, dictionary.pad()
        )

        self.sentence_encoder = self.build_encoder(args, dictionary, embed_tokens)

        self.lm_head = self.build_lm_head(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=(
                self.sentence_encoder.embed_tokens.weight
                if not args.untie_weights_roberta
                else None
            ),
        )

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = ESM2TransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)

    def init_link_feature(self, args):
        links_feature = self.args.links_feature.split(":")
        links_dim = 0
        if "feature" in links_feature:
            links_dim += args.encoder_embed_dim
        if "position" in links_feature:
            self.link_positional = PositionalEmbedding(args.max_positions, args.encoder_embed_dim, self.pad_idx, True)
            links_dim += args.encoder_embed_dim
        elif "sinposition" in links_feature:
            self.link_positional = PositionalEmbedding(args.max_positions, args.encoder_embed_dim, self.pad_idx, False)
            links_dim += args.encoder_embed_dim
        else:
            self.link_positional = None
        self.query_linear = nn.Linear(links_dim, args.encoder_embed_dim)
        self.key_linear = nn.Linear(links_dim, args.encoder_embed_dim)
        self.gate_linear = nn.Linear(links_dim, args.encoder_attention_heads)


    def forward(
        self,
        src_tokens,
        ext_mask=None,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        token_dropout=True,
        need_head_weights=False, 
        return_contacts=False,
        require_links=True,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        if ext_mask is None:
            ext_mask = torch.zeros_like(src_tokens).bool()
        bsz = src_tokens.size(0)
        seq_length = torch.sum(src_tokens.ne(self.pad_idx).long(), dim=-1)
        bos_mask = src_tokens.eq(self.bos_idx)
        eos_mask = src_tokens.eq(self.eos_idx)
        ext_mask[bos_mask] = True
        ext_mask[eos_mask] = True
        ext_mask_long = ext_mask.long()
        pos_len = ext_mask_long * (self.ext_state_num - 1) + 1
        ext_pos = torch.cumsum(pos_len, dim=-1) - pos_len
        expend_length = torch.sum(ext_mask_long, dim=-1) * (self.ext_state_num - 1) + seq_length
        max_length = torch.max(expend_length)
        # ext_pos[ext_pos >= max_length] = 0
        # ext_index = src_tokens.new_zeros(bsz, max_length)

        ext_pos[ext_pos >= max_length] = max_length
        ext_index = ext_mask_long.new_zeros(bsz, max_length + 1)

        ext_index.scatter_(dim=-1, index=ext_pos, src=torch.ones_like(ext_mask_long))
        ext_index = ext_index[:, :-1]

        # aa_mask = ext_index.bool()

        # is_ext_state = src_tokens.new_zeros(bsz, max_length)
        is_ext_state = ext_mask_long.new_zeros(bsz, max_length + 1)
        src_tensor = torch.ones_like(ext_mask_long)
        src_tensor.masked_fill_(ext_mask, -1)
        is_ext_state.scatter_(dim=-1, index=ext_pos, src=src_tensor)
        is_ext_state[:, 0] = src_tensor[:, 0]
        is_ext_state.masked_fill_(is_ext_state==-1, 0)
        is_ext_state = (1 - is_ext_state).bool()

        is_ext_state = is_ext_state[:, :-1]
        is_ext_state[new_arange(is_ext_state) >= expend_length.unsqueeze(-1)] = 0

        is_ext_state_cum = torch.cumsum(is_ext_state.long(), dim=-1)
        is_ext_state_cum[is_ext_state == 0] = 0
        ext_gather_pos = is_ext_state_cum - 1
        # pos_dim2 = (is_ext_state_cum + self.ext_state_num - 1) // self.ext_state_num
        pos_dim2 = (is_ext_state_cum + self.ext_state_num - 1) % self.ext_state_num + 1
        pos_dim2[is_ext_state == 0] = 0
        pos_dim1 = torch.cumsum(ext_index, dim=-1) - 1

        aa_mask = (pos_dim1, pos_dim2)

        ext_index = torch.cumsum(ext_index, dim=-1) - ext_index[:, 0][:, None]
        ext_src_tokens = torch.gather(input=src_tokens, dim=-1, index=ext_index)


        x, extra = self.extract_features(
            ext_src_tokens, aa_mask=aa_mask, return_all_hiddens=return_all_hiddens, token_dropout=token_dropout,
            need_head_weights=need_head_weights, return_contacts=return_contacts,
        )

        extra["ext_index"] = ext_index
        extra["ext_src_tokens"] = ext_src_tokens
        extra["ext_gather_pos"] = ext_gather_pos

        links = None
        if require_links:
            links = self.extract_links(x, 
                        ext_index, 
                        is_ext_state, 
                        src_tokens,
                        ext_src_tokens,
                        self.link_positional, 
                        self.query_linear, 
                        self.key_linear, 
                        self.gate_linear, 
                    )
        extra["links"] = links

        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra

    def extract_features(self, src_tokens, aa_mask=None, return_all_hiddens=False, token_dropout=True, need_head_weights=False, return_contacts=False, **kwargs):
        encoder_out = self.sentence_encoder(
            src_tokens,
            aa_mask=aa_mask,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
            token_dropout=token_dropout,
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None

        return features, {"inner_states": inner_states}

    def extract_links(self, features, ext_index, is_ext_state, src_tokens, ext_src_tokens,
        link_positional, query_linear, key_linear, gate_linear):

        k = self.ext_state_num
        batch_size = features.shape[0]
        seqlen = features.shape[1]
        
        both_ext_mask = is_ext_state.unsqueeze(1) & is_ext_state.unsqueeze(2)
        valid_pair_mask = both_ext_mask
        # valid_pair_mask = (ext_index.unsqueeze(1) > ext_index.unsqueeze(2)) & both_ext_mask

        # print(valid_pair_mask)
        # exit()

        links_feature = vars(self.args).get("links_feature", "feature:position").split(":")

        links_feature_arr = []
        if "feature" in links_feature:
            links_feature_arr.append(features)
        if "position" in links_feature or "sinposition" in links_feature:
            # link_pe = link_positional(src_tokens)# [batch_size, real_seqlen, pe_dim]
            # pe_dim = link_pe.size(-1) 
            # link_pe = torch.gather(input=link_pe, dim=1, index=ext_index.unsqueeze(-1).expand(-1, -1, pe_dim)) # [batch_size, expend_len]
            # links_feature_arr.append(link_pe)
            link_pe = link_positional(ext_src_tokens)# [batch_size, real_seqlen, pe_dim]
            links_feature_arr.append(link_pe)

        features_withpos = torch.cat(links_feature_arr, dim=-1)

        chunk_num = self.args.encoder_attention_heads
        chunk_size = self.args.encoder_embed_dim // self.args.encoder_attention_heads
        ninf = float("-inf")
        target_dtype = torch.float

        query_chunks = query_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        key_chunks = key_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        log_gates = F.log_softmax(gate_linear(features_withpos), dim=-1, dtype=target_dtype) # batch_size * seqlen * chunk_num

        log_multi_content = (torch.einsum("bicf,bjcf->bijc", query_chunks.to(dtype=target_dtype), key_chunks.to(dtype=target_dtype)) / (chunk_size ** 0.5))
   
        link_mask = torch.ones(seqlen, seqlen, device=ext_src_tokens.device, dtype=bool).triu_(1).unsqueeze(0) & ext_src_tokens.ne(self.pad_idx).unsqueeze(1)
        link_mask = link_mask & valid_pair_mask
        link_nouse_mask = link_mask.sum(dim=2, keepdim=True) == 0 # [batch_size, seqlen, 1]
        link_mask.masked_fill_(link_nouse_mask, True) # [batch_size, seqlen, seqlen]

        log_multi_content.masked_fill_(~link_mask.unsqueeze(-1), ninf) # [batch_size, real_seqlen * k, seqlen, chunk_num]

        if self.args.max_transition_length != -1:
            # log_multi_content = log_multi_content[:, :, 1:, :] # [batch_size, real_seqlen * k, seqlen - 1, chunk_num]
            # ext_src_tokens = torch.repeat_interleave(prev_output_tokens, repeats=k, dim=-1) # [bsz, seq_len * k]
            log_multi_content_extract, link_nouse_mask = self.extract_valid_links(log_multi_content, ext_src_tokens.ne(self.pad_idx))
                    # batch * seqlen * trans_len * chunk_num, batch * seqlen * trans_len
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
            log_multi_content_extract = F.log_softmax(log_multi_content_extract, dim=2)
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
            links = logsumexp(log_multi_content_extract + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * trans_len
        else:
            log_multi_attention = F.log_softmax(log_multi_content, dim=2) # [batch_size, real_seqlen * k, seqlen, chunk_num]
            log_multi_attention = log_multi_attention.masked_fill(link_nouse_mask.unsqueeze(-1), ninf)
            links = logsumexp(log_multi_attention + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * seqlen

        return links

    def extract_valid_links(self, content, valid_mask):
        # batch * prelen * prelen * chunk, batch * prelen

        prelen = content.shape[1]
        translen: int = self.args.max_transition_length
        if translen > prelen - 1:
            translen = prelen - 1
        valid_links_idx = torch.arange(prelen, dtype=torch.long, device=content.device).unsqueeze(1) + \
                    torch.arange(translen, dtype=torch.long, device=content.device).unsqueeze(0) + 1
        invalid_idx_mask = valid_links_idx >= valid_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)
        valid_links_idx = valid_links_idx.unsqueeze(0).masked_fill(invalid_idx_mask, 0)

        res = content.gather(2, valid_links_idx.unsqueeze(-1).expand(-1, -1, -1, content.shape[-1]))
        res.masked_fill_(invalid_idx_mask.unsqueeze(-1), float("-inf"))

        return res, invalid_idx_mask.all(-1) # batch * prelen * trans_len * chunk, batch * prelen * trans_len

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


@register_model_architecture("p_roberta_ext_denselink", "p_roberta_ext_denselink")
def p_base_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 12)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.0)

    args.max_source_positions = safe_getattr(args, "max_positions", 512)
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )

    # BERT has a few structural differences compared to the original Transformer
    args.encoder_learned_pos = safe_getattr(args, "encoder_learned_pos", False)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", True)
    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", True)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = safe_getattr(
        args, "encoder_normalize_before", False
    )
    args.pooler_activation_fn = safe_getattr(args, "pooler_activation_fn", "tanh")
    args.untie_weights_roberta = safe_getattr(args, "untie_weights_roberta", False)

    # Adaptive input config
    args.adaptive_input = safe_getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layerdrop = safe_getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layers_to_keep = safe_getattr(args, "encoder_layers_to_keep", None)

    # Quantization noise config
    args.quant_noise_pq = safe_getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = safe_getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = safe_getattr(args, "quant_noise_scalar", 0)

    # R4F config
    args.spectral_norm_classification_head = safe_getattr(
        args, "spectral_norm_classification_head", False
    )


@register_model_architecture("p_roberta_ext_denselink", "p_roberta_ext_denselink_prenorm")
def p_roberta_prenorm_architecture(args):
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", False)
    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", True)
    p_base_architecture(args)


@register_model_architecture("p_roberta_ext_denselink", "p_roberta_ext_denselink_base")
def p_roberta_base_architecture(args):
    p_base_architecture(args)


@register_model_architecture("p_roberta_ext_denselink", "p_roberta_ext_denselink_large")
def p_roberta_large_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
    p_base_architecture(args)


@register_model_architecture("p_roberta_ext_denselink", "p_ext_denselink_xlm")
def p_xlm_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 16)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1280)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 1280 * 4)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
    p_base_architecture(args)
