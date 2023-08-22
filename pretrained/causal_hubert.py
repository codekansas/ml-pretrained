"""Defines an API for interacting with a causal HuBERT model.

This model is trained to predict HuBERT tokens from the previous N audio
embedding vectors, rather than using a bidirectional transformer. This lends
itself better to real-time applications, as the model can be run in a causal
manner.

One difference from the original HuBERT model is that this model uses a
convolutional encoder wich kernel sizes matching the stride. While this can
have worse performance than the original convolutional encoder, it allows us
to process chunks of audio as they come in.

.. highlight:: python
.. code-block:: python

    from pretrained.causal_hubert import pretrained_causal_hubert

    model = pretrained_causal_hubert("base-conv-encoder")
    state = None

    for waveform_chunk in waveform_chunks:
        tokens, state = model(waveform_chunk, state)
"""

import argparse
import logging
import math
import sys
from typing import Literal, NamedTuple, cast, get_args

import safetensors.torch
import torch
import torch.nn.functional as F
from ml.models.activations import ActivationType
from ml.models.embeddings import get_positional_embeddings, rotary_embeddings
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer
from torch import Tensor, nn

from pretrained.hubert import HubertFeatureEncoder, HubertFeatureProjection

logger = logging.getLogger(__name__)

DEFAULT_CONV_DIM: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
DEFAULT_CONV_STRIDE: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)

PretrainedCausalHubertSize = Literal["base-conv-encoder", "base-linear-encoder", "base-linear-encoder-better"]


def cast_pretrained_causal_hubert_key(s: str) -> PretrainedCausalHubertSize:
    if s not in get_args(PretrainedCausalHubertSize):
        raise KeyError(f"Invalid HuBERT key: {s} Expected one of: {get_args(PretrainedCausalHubertSize)}")
    return cast(PretrainedCausalHubertSize, s)


class SelfAttentionState(NamedTuple):
    key: Tensor
    value: Tensor


class CausalHubertState(NamedTuple):
    offset: int
    waveform_leftover: Tensor
    attn_states: list[SelfAttentionState]


class Attention(nn.Module):
    __constants__ = ["num_heads", "head_dim", "local_attn"]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        local_attn: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads."

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.local_attn = local_attn

        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        self.in_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        state: SelfAttentionState | None = None,
    ) -> tuple[Tensor, SelfAttentionState]:
        assert x.dim() == 3

        x = self.norm(x)
        x = self.in_proj(x)

        query, key, value = x.chunk(3, dim=2)
        query = query.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)
        key = key.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)
        value = value.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)

        if state is not None:
            key = torch.cat([state.key, key], dim=2)
            value = torch.cat([state.value, value], dim=2)

        (bsz, _, qtsz, _), (_, _, ktsz, _) = query.shape, key.shape
        mask = mask[-qtsz:, -ktsz:]

        x = F.scaled_dot_product_attention(query.flatten(0, 1), key.flatten(0, 1), value.flatten(0, 1), attn_mask=mask)
        x = x.unflatten(0, (bsz, self.num_heads)).transpose(1, 2).flatten(2, 3)

        state_out = SelfAttentionState(
            key=key[:, :, -self.local_attn :, :],
            value=value[:, :, -self.local_attn :, :],
        )

        return self.dropout(x), state_out


class FeedForward(nn.Module):
    __constants__ = ["norm_first"]

    def __init__(
        self,
        hidden_size: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(hidden_size, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dim_feedforward: int,
        local_attn: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.attention = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            local_attn=local_attn,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )

        self.feedforward = FeedForward(
            hidden_size=hidden_size,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        state: SelfAttentionState | None = None,
    ) -> tuple[Tensor, SelfAttentionState]:
        xh, state_out = self.attention(x, mask, state)
        x = x + xh
        x = x + self.feedforward(x)
        return x, state_out


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
        local_attn: int,
        max_tsz: int = 2048,
    ) -> None:
        super().__init__()

        layers = [
            SelfAttentionLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                local_attn=local_attn,
            )
            for _ in range(num_layers)
        ]
        self.layers = cast(list[SelfAttentionLayer], nn.ModuleList(layers))

        # Builds the causal mask.
        # upper_mask = torch.triu(torch.ones(max_tsz, max_tsz), diagonal=1).bool()
        # lower_mask = torch.tril(torch.ones(max_tsz, max_tsz), diagonal=-local_attn).bool()
        # mask = torch.zeros(max_tsz, max_tsz).masked_fill(upper_mask | lower_mask, float("-inf"))
        # self.register_buffer("mask", mask, persistent=False)

        upper_mask = torch.triu(torch.ones(max_tsz, max_tsz + local_attn), diagonal=local_attn + 1).bool()
        lower_mask = torch.tril(torch.ones(max_tsz, max_tsz + local_attn), diagonal=-1).bool()
        mask = torch.zeros(max_tsz, max_tsz + local_attn).masked_fill(upper_mask | lower_mask, float("-inf"))
        self.register_buffer("mask", mask, persistent=False)

    mask: Tensor

    def forward(
        self,
        x: Tensor,
        states: list[SelfAttentionState] | None = None,
    ) -> tuple[Tensor, list[SelfAttentionState]]:
        states_out: list[SelfAttentionState] = []
        for i, layer in enumerate(self.layers):
            x, state = layer(x, self.mask, None if states is None or len(states) == 0 else states[i])
            states_out.append(state)
        return x, states_out


class ConvExtractor(nn.Module):
    __constants__ = ["receptive_field_size"]

    def __init__(
        self,
        hidden_size: int,
        conv_dim: tuple[int, ...] = DEFAULT_CONV_DIM,
        conv_stride: tuple[int, ...] = DEFAULT_CONV_STRIDE,
        conv_bias: bool = True,
        feat_extract_norm: Literal["group", "layer"] = "layer",
        feat_extract_activation: ActivationType = "gelu",
        layer_norm_eps: float = 1e-5,
        feat_proj_dropout: float = 0.0,
        feat_proj_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.receptive_field_size = math.prod(conv_stride)

        self.hubert_extractor = HubertFeatureEncoder(
            conv_dim=conv_dim,
            conv_stride=conv_stride,
            conv_kernel=conv_stride,
            conv_bias=conv_bias,
            feat_extract_norm=feat_extract_norm,
            feat_extract_activation=feat_extract_activation,
        )

        self.hubert_projector = HubertFeatureProjection(
            input_size=conv_dim[-1],
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            feat_proj_dropout=feat_proj_dropout,
            feat_proj_layer_norm=feat_proj_layer_norm,
        )

    def forward(self, waveform: Tensor) -> Tensor:
        x = self.hubert_extractor(waveform).transpose(1, 2)
        x = self.hubert_projector(x)
        return x


class LinearExtractor(nn.Module):
    __constants__ = ["receptive_field_size"]

    def __init__(
        self,
        hidden_size: int,
        receptive_field_size: int,
    ) -> None:
        super().__init__()

        self.receptive_field_size = receptive_field_size

        self.extractor = nn.Sequential(
            nn.Linear(
                in_features=receptive_field_size,
                out_features=hidden_size,
            ),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

    def forward(self, waveform: Tensor) -> Tensor:
        x = waveform.unflatten(1, (-1, self.receptive_field_size))
        x = self.extractor(x)
        return x


class CausalHubert(nn.Module):
    __constants__ = ["receptive_field_size"]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
        num_hubert_tokens: int,
        local_attn: int,
        extractor: ConvExtractor | LinearExtractor,
        max_tsz: int = 2048,
    ) -> None:
        super().__init__()

        self.hubert_pos_embs = get_positional_embeddings(max_tsz, hidden_size, "rotary")

        self.extractor = extractor
        self.receptive_field_size = extractor.receptive_field_size

        self.hubert_transformer = SelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            local_attn=local_attn,
            max_tsz=max_tsz,
        )

        self.token_projector = nn.Linear(hidden_size, num_hubert_tokens)

    def forward(self, waveform: Tensor, state: CausalHubertState | None = None) -> tuple[Tensor, CausalHubertState]:
        # Prepends the leftover waveform from the previous call.
        attn_states: list[SelfAttentionState] | None = None
        if state is not None:
            waveform = torch.cat([state.waveform_leftover, waveform], dim=1)
            attn_states = state.attn_states
        _, tsz = waveform.shape
        tsz_leftover = tsz % self.receptive_field_size
        waveform, waveform_leftover = waveform[:, : tsz - tsz_leftover], waveform[:, tsz - tsz_leftover :]

        x = self.extractor(waveform)

        # Adds the positional embeddings.
        offset = 0 if state is None else state.offset
        x = self.hubert_pos_embs(x, offset=offset)
        offset += x.shape[1]

        # Runs the transformer.
        x, attn_states_out = self.hubert_transformer(x, attn_states)

        # Predicts the output tokens.
        x = self.token_projector(x)

        # Gets the new state.
        state_out = CausalHubertState(
            offset=offset,
            waveform_leftover=waveform_leftover,
            attn_states=attn_states_out,
        )

        return x, state_out

    def predictor(self) -> "CausalHubertPredictor":
        return CausalHubertPredictor(self)


class CausalHubertPredictor(nn.Module):
    def __init__(self, hubert: CausalHubert) -> None:
        super().__init__()

        self.extractor = hubert.extractor
        self.receptive_field_size = hubert.receptive_field_size

        self.hubert_transformer = hubert.hubert_transformer
        self.token_projector = hubert.token_projector

    def forward(self, waveform: Tensor, state: CausalHubertState | None = None) -> tuple[Tensor, CausalHubertState]:
        # Prepends the leftover waveform from the previous call.
        attn_states: list[SelfAttentionState] | None = None
        if state is not None:
            waveform = torch.cat([state.waveform_leftover, waveform], dim=1)
            attn_states = state.attn_states
        _, tsz = waveform.shape
        tsz_leftover = tsz % self.receptive_field_size

        if tsz_leftover == tsz:
            state_out = CausalHubertState(
                offset=0 if state is None else state.offset,
                waveform_leftover=waveform,
                attn_states=[] if state is None else state.attn_states,
            )
            return torch.empty((waveform.shape[0], 0), dtype=torch.long, device=waveform.device), state_out

        waveform, waveform_leftover = waveform[:, : tsz - tsz_leftover], waveform[:, tsz - tsz_leftover :]

        x = self.extractor(waveform)

        # Adds the positional embeddings.
        offset = 0 if state is None else state.offset
        x = rotary_embeddings(x, offset=offset)
        offset += x.shape[1]

        # Runs the transformer.
        x, attn_states_out = self.hubert_transformer(x, attn_states)

        # Predicts the output tokens.
        x = self.token_projector(x).argmax(-1)

        # Gets the new state.
        state_out = CausalHubertState(
            offset=offset,
            waveform_leftover=waveform_leftover,
            attn_states=attn_states_out,
        )

        return x, state_out


def _load_pretrained_causal_hubert(
    size: PretrainedCausalHubertSize,
    ckpt_url: str,
    sha256: str,
    hidden_size: int,
    num_heads: int,
    dim_feedforward: int,
    num_layers: int,
    num_hubert_tokens: int,
    local_attn: int,
    load_weights: bool = True,
    conv_dim: tuple[int, ...] = DEFAULT_CONV_DIM,
    conv_stride: tuple[int, ...] = DEFAULT_CONV_STRIDE,
    conv_bias: bool = True,
    feat_extract_norm: Literal["group", "layer"] = "layer",
    feat_extract_activation: ActivationType = "gelu",
    layer_norm_eps: float = 1e-5,
    feat_proj_dropout: float = 0.0,
    feat_proj_layer_norm: bool = True,
    max_tsz: int = 2048,
) -> CausalHubert:
    with Timer("building empty model", spinner=True):
        model = CausalHubert(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            num_hubert_tokens=num_hubert_tokens,
            local_attn=local_attn,
            extractor=ConvExtractor(
                hidden_size=hidden_size,
                conv_dim=conv_dim,
                conv_stride=conv_stride,
                conv_bias=conv_bias,
                feat_extract_norm=feat_extract_norm,
                feat_extract_activation=feat_extract_activation,
                layer_norm_eps=layer_norm_eps,
                feat_proj_dropout=feat_proj_dropout,
                feat_proj_layer_norm=feat_proj_layer_norm,
            ),
            max_tsz=max_tsz,
        )

    # Loads the model weights.
    if load_weights:
        model_fname = f"{size}.bin"

        with Timer("downloading checkpoint"):
            model_path = ensure_downloaded(ckpt_url, "causal-hubert", model_fname, sha256=sha256)

        with Timer("loading checkpoint", spinner=True):
            ckpt = safetensors.torch.load_file(model_path, device="cpu")
            model.load_state_dict(ckpt)

    return model


def _load_pretrained_causal_hubert_linear(
    size: PretrainedCausalHubertSize,
    ckpt_url: str,
    sha256: str,
    hidden_size: int,
    num_heads: int,
    dim_feedforward: int,
    num_layers: int,
    num_hubert_tokens: int,
    local_attn: int,
    load_weights: bool = True,
    receptive_field_size: int = 320,
    max_tsz: int = 2048,
) -> CausalHubert:
    with Timer("building empty model", spinner=True):
        model = CausalHubert(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            num_hubert_tokens=num_hubert_tokens,
            local_attn=local_attn,
            extractor=LinearExtractor(
                hidden_size=hidden_size,
                receptive_field_size=receptive_field_size,
            ),
            max_tsz=max_tsz,
        )

    # Loads the model weights.
    if load_weights:
        model_fname = f"{size}.bin"

        with Timer("downloading checkpoint"):
            model_path = ensure_downloaded(ckpt_url, "causal-hubert", model_fname, sha256=sha256)

        with Timer("loading checkpoint", spinner=True):
            ckpt = safetensors.torch.load_file(model_path, device="cpu")
            model.load_state_dict(ckpt)

    return model


def pretrained_causal_hubert(
    size: PretrainedCausalHubertSize,
    load_weights: bool = True,
) -> CausalHubert:
    match size:
        case "base-conv-encoder":
            return _load_pretrained_causal_hubert(
                size=size,
                ckpt_url="https://huggingface.co/codekansas/causal-hubert/resolve/main/base-conv-encoder.bin",
                sha256="b3af6671bf6288d9c8f8a5fd141ebe238feb66a7df1cc4115a7bb746be4a3c4e",
                hidden_size=768,
                dim_feedforward=2048,
                num_heads=12,
                num_layers=6,
                num_hubert_tokens=100,
                local_attn=32,
                load_weights=load_weights,
                feat_extract_norm="layer",
                conv_bias=True,
            )

        case "base-linear-encoder":
            return _load_pretrained_causal_hubert_linear(
                size=size,
                ckpt_url="https://huggingface.co/codekansas/causal-hubert/resolve/main/base-linear-encoder.bin",
                sha256="33f0f28da68c36bd8163a12bdf6940d38df36ee5fd45b5b1e4bb74b96c9a17f2",
                hidden_size=768,
                dim_feedforward=2048,
                num_heads=12,
                num_layers=6,
                num_hubert_tokens=100,
                local_attn=32,
                load_weights=load_weights,
            )

        case "base-linear-encoder-better":
            return _load_pretrained_causal_hubert_linear(
                size=size,
                ckpt_url="https://huggingface.co/codekansas/causal-hubert/resolve/main/base-linear-encoder-better.bin",
                sha256="e6d7d12e6f87f63acad6aa8f63bac1f6d80bd925402f8b95d185740a0bc00ab3",
                hidden_size=768,
                num_heads=12,
                dim_feedforward=2048,
                num_layers=6,
                num_hubert_tokens=100,
                local_attn=125,
                load_weights=load_weights,
            )

        case _:
            raise NotImplementedError(f"Invalid size: {size}")


def test_causal_hubert() -> None:
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--key", choices=get_args(PretrainedCausalHubertSize), default="base-conv-encoder")
    parser.add_argument("-c", "--chunk-size", type=int, default=16000 // 10)
    parser.add_argument("-n", "--num-chunks", type=int, default=50 * 10)
    args = parser.parse_args()

    model = pretrained_causal_hubert(args.key)
    predictor = model.predictor()
    state: CausalHubertState | None = None

    import sounddevice as sd  # type: ignore[import]

    with sd.InputStream(samplerate=16000, channels=1, dtype="float32") as stream:
        sys.stdout.write("Codes:\n")
        i = 0
        for _ in range(args.num_chunks):
            data, _ = stream.read(args.chunk_size)
            waveform = torch.from_numpy(data.reshape(1, -1)).float()
            codes, state = predictor(waveform, state)
            for code in codes.squeeze(0).cpu().tolist():
                i += 1
                s = f"{code}"
                sys.stdout.write(f"{s:>4s}")
                if i % 20 == 0:
                    sys.stdout.write("\n")
            sys.stdout.flush()


if __name__ == "__main__":
    # python -m pretrained.causal_hubert
    test_causal_hubert()
