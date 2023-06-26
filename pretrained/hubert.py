"""Defines a simple API for using Meta's pretrained Hubert model.

.. highlight:: python
.. code-block:: python

    from pretrained.hubert import pretrained_hubert

    model = pretrained_hubert("base")
    predictor = model.predictor()

    # Gets HuBERT embeddings for a waveform.
    predictor.predict(torch.randn(1, 16_000), output_layer=None)

    # Gets HuBERT embeddings for a long waveform, in batches.
    predictor.predict_in_chunks(torch.randn(1, 160_000), 16_000, output_layer=None)

The choices for the model key are:

- ``"base"`` - 12 layers, 768 hidden size, 12 attention heads.
- ``"large"`` - 24 layers, 1024 hidden size, 16 attention heads.
- ``"extra_large"`` - 48 layers, 1280 hidden size, 16 attention heads.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.sox_effects as ta_sox
from ml.models.activations import ActivationType, get_activation
from ml.models.kmeans import KMeans
from ml.utils.audio import Reader, get_audio_props, read_audio
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.device.auto import AutoDevice
from ml.utils.device.base import BaseDevice
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer
from torch import Tensor, nn

PretrainedHubertSize = Literal["base", "large", "extra_large"]


@dataclass
class HubertConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_act: ActivationType
    hidden_dropout: float
    activation_dropout: float
    feat_proj_layer_norm: bool
    feat_proj_dropout: float
    layer_norm_eps: float
    feat_extract_norm: str
    feat_extract_dropout: float
    feat_extract_activation: ActivationType
    conv_dim: tuple[int, ...]
    conv_stride: tuple[int, ...]
    conv_kernel: tuple[int, ...]
    conv_bias: bool
    num_conv_pos_embeddings: int
    num_conv_pos_embedding_groups: int
    do_stable_layer_norm: bool
    pre_normalize: bool

    @property
    def num_feat_extract_layers(self) -> int:
        return len(self.conv_dim)


def normalize_output_layer(output_layer: int | float | None, num_layers: int) -> int | None:
    if output_layer is not None:
        if isinstance(output_layer, float):
            output_layer = round(output_layer * num_layers)
        if output_layer < 0:
            output_layer += num_layers
        if not (0 <= output_layer < num_layers):
            raise ValueError(f"output_layer={output_layer} is outside the range of available layers")
    return output_layer


class HubertSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings: int) -> None:
        super().__init__()

        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class HubertPositionalConvEmbedding(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        self.conv = nn.utils.weight_norm(conv, dim=2)
        self.padding = HubertSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = get_activation(config.feat_extract_activation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

    def remove_weight_norm_(self) -> None:
        self.conv = nn.utils.remove_weight_norm(self.conv)


class HubertAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"`embed_dim` must be divisible by num_heads (got {self.embed_dim=} and {num_heads=}).")

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int) -> Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: Tensor, causal: bool = False) -> Tensor:
        """Runs the HuBERT attention layer.

        Args:
            hidden_states: Input states for the attention layer.
            causal: If set, use causal attention.

        Returns:
            The attention outputs.
        """
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=causal)
        attn_output = attn_output.transpose(1, 2).flatten(2)
        final_output = self.out_proj(attn_output)

        return final_output


class HubertFeedForward(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = get_activation(config.hidden_act)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class HubertEncoderLayer(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        self.attention = HubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, causal: bool = False) -> Tensor:
        attn_residual = hidden_states
        hidden_states = self.attention.forward(hidden_states, causal=causal)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class HubertEncoder(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        layers = nn.ModuleList([HubertEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layers = cast(list[HubertEncoderLayer], layers)
        self.gradient_checkpointing = False

    def forward(self, hidden_states: Tensor, causal: bool = False, output_layer: int | float | None = None) -> Tensor:
        position_embeddings = self.pos_conv_embed.forward(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_layer = normalize_output_layer(output_layer, len(self.layers))
        for i, layer in enumerate(self.layers):
            hidden_states = layer.forward(hidden_states, causal=causal)
            if output_layer is not None and i == output_layer:
                break
        return hidden_states

    def remove_weight_norm_(self) -> None:
        self.pos_conv_embed.remove_weight_norm_()


class HubertGroupNormConvLayer(nn.Module):
    def __init__(self, config: HubertConfig, layer_id: int = 0) -> None:
        super().__init__()

        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = get_activation(config.feat_extract_activation)

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class HubertNoLayerNormConvLayer(nn.Module):
    def __init__(self, config: HubertConfig, layer_id: int = 0) -> None:
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = get_activation(config.feat_extract_activation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class HubertLayerNormConvLayer(nn.Module):
    def __init__(self, config: HubertConfig, layer_id: int = 0) -> None:
        super().__init__()

        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.activation = get_activation(config.feat_extract_activation)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


class HubertFeatureEncoder(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        conv_layers: list[nn.Module] = []
        if config.feat_extract_norm == "group":
            conv_layers += [HubertGroupNormConvLayer(config, layer_id=0)]
            for i in range(config.num_feat_extract_layers - 1):
                conv_layers += [HubertNoLayerNormConvLayer(config, layer_id=i + 1)]
        elif config.feat_extract_norm == "layer":
            for i in range(config.num_feat_extract_layers):
                conv_layers += [HubertLayerNormConvLayer(config, layer_id=i)]
        else:
            raise ValueError(f"{config.feat_extract_norm=}, but has to be one of ['group', 'layer']")
        self.conv_layers = nn.ModuleList(conv_layers)

    def _freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_values: Tensor) -> Tensor:
        hidden_states = input_values[:, None]
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class HubertFeatureProjection(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        self.feat_proj_layer_norm = config.feat_proj_layer_norm
        if self.feat_proj_layer_norm:
            self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.feat_proj_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class HubertEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        self.attention = HubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, causal: bool = False) -> Tensor:
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.attention.forward(hidden_states, causal=causal)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward.forward(self.final_layer_norm(hidden_states))
        return hidden_states


class HubertEncoderStableLayerNorm(nn.Module):
    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        layers = [HubertEncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        self.layers = cast(list[HubertEncoderLayerStableLayerNorm], nn.ModuleList(layers))

    def forward(self, hidden_states: Tensor, causal: bool = False, output_layer: int | float | None = None) -> Tensor:
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)
        output_layer = normalize_output_layer(output_layer, len(self.layers))
        for i, layer in enumerate(self.layers):
            hidden_states = layer.forward(hidden_states, causal=causal)
            if output_layer is not None and i == output_layer:
                break
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def remove_weight_norm_(self) -> None:
        self.pos_conv_embed.remove_weight_norm_()


class Hubert(nn.Module):
    __constants__ = ["conv_kernel", "conv_stride"]

    def __init__(self, config: HubertConfig) -> None:
        super().__init__()

        self.conv_kernel = config.conv_kernel
        self.conv_stride = config.conv_stride
        self.pre_normalize = config.pre_normalize

        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = HubertFeatureProjection(config)
        self.encoder: HubertEncoderStableLayerNorm | HubertEncoder
        if config.do_stable_layer_norm:
            self.encoder = HubertEncoderStableLayerNorm(config)
        else:
            self.encoder = HubertEncoder(config)

    def forward(
        self,
        input_values: Tensor | None,
        causal: bool = False,
        output_layer: int | float | None = None,
    ) -> Tensor:
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        hidden_states = self.feature_projection(extract_features)
        hidden_states = self.encoder.forward(hidden_states, causal=causal, output_layer=output_layer)
        return hidden_states

    def predictor(
        self,
        kmeans: KMeans | None = None,
        *,
        device: BaseDevice | None = None,
    ) -> "HubertPredictor":
        return HubertPredictor(self, kmeans, device=device)

    def remove_weight_norm_(self) -> None:
        assert isinstance(self.encoder, (HubertEncoderStableLayerNorm, HubertEncoder))
        self.encoder.remove_weight_norm_()


class HubertPredictor:
    def __init__(
        self,
        hubert_model: Hubert,
        kmeans: KMeans | None = None,
        *,
        device: BaseDevice | None = None,
    ) -> None:
        """Provides an API for doing predictoins with a HuBERT model.

        Note that this module is not an `nn.Module`, so you can use it in your
        module without worrying about storing all the weights on accident.

        Args:
            hubert_model: The HuBERT model to use for predictions.
            kmeans: The kmeans model to use for quantization. If `None`, don't
                quantize.
            device: The device to use for predictions. If `None`, will use the
                device returned by AutoDevice.detect_device().
        """
        super().__init__()

        # Remove weight norm for inference, if it exists.
        hubert_model.remove_weight_norm_()

        self.device = AutoDevice.detect_device() if device is None else device
        self.model = hubert_model.eval()
        self.kmeans = kmeans.eval() if kmeans is not None else None
        self.device.module_to(self.model)
        if self.kmeans is not None:
            self.device.module_to(self.kmeans)
        self.sample_rate = 16_000  # True for all HuBERT models.

    def predict(
        self,
        waveform: np.ndarray | Tensor,
        output_layer: int | float | None = None,
        causal: bool = False,
    ) -> Tensor:
        """Gets the hidden states for the given waveform.

        Args:
            waveform: The waveform to get hidden states for, with shape (B, T)
            output_layer: The layer to get hidden states from. If `None`, will
                return the hidden states from the last layer. If an `int`, will
                return the hidden states from that layer. If a `float`, will
                return the hidden states from the layer at that percentage of
                the model. For example, `0.5` will return the hidden states
                from the middle layer. Negative values will wrap around.
            causal: If set, use a causal attention mask.

        Returns:
            The hidden states for the given waveform, with shape (B, T, D)
        """
        waveform = self.device.tensor_to(waveform)
        features = self.model.forward(waveform, causal=causal, output_layer=output_layer)
        if self.kmeans is not None:
            features = self.kmeans.forward(features)
        return features

    def predict_in_chunks(
        self,
        waveform: Tensor | np.ndarray,
        chunk_size: int = 16_000 * 10,
        output_layer: int | float | None = None,
        causal: bool = False,
    ) -> Tensor:
        """Gets the hidden states for the given waveform, in chunks.

        This is useful for processing very long waveforms, as it allows you to
        process the waveform in chunks, rather than loading the entire waveform
        into memory at once.

        Args:
            waveform: The waveform to get hidden states for, with shape (B, T)
            chunk_size: The size of each chunk to process, in frames.
            output_layer: The layer to get hidden states from. If `None`, will
                return the hidden states from the last layer. If an `int`, will
                return the hidden states from that layer. If a `float`, will
                return the hidden states from the layer at that percentage of
                the model. For example, `0.5` will return the hidden states
                from the middle layer. Negative values will wrap around.
            causal: If set, use a causal attention mask.

        Returns:
            The hidden states for the given waveform, with shape (B, T, D)
        """
        with torch.inference_mode(), self.device.autocast_context():
            x = self.device.tensor_to(waveform)  # Loads entire waveform into device memory.

            if self.model.pre_normalize:
                x = F.layer_norm(x, x.shape)

            feat = []
            for start in range(0, x.size(1), chunk_size):
                x_chunk = x[:, start : start + chunk_size]
                feat_chunk = self.model.forward(x_chunk, causal=causal, output_layer=output_layer)
                if self.kmeans is not None:
                    feat_chunk = self.kmeans.forward(feat_chunk)
                feat.append(feat_chunk.cpu())

        return torch.cat(feat, 1).squeeze(0)

    def predict_file(
        self,
        path: str | Path,
        chunk_length_sec: float = 10.0,
        output_layer: int | float | None = None,
        causal: bool = False,
        *,
        reader: Reader = "sf",
    ) -> Tensor:
        """Gets the hidden states for the given audio file, in chunks.

        Args:
            path: The path to the audio file to process.
            chunk_length_sec: The length of each chunk to process, in seconds.
            output_layer: The layer to get hidden states from. If `None`, will
                return the hidden states from the last layer. If an `int`, will
                return the hidden states from that layer. If a `float`, will
                return the hidden states from the layer at that percentage of
                the model. For example, `0.5` will return the hidden states
                from the middle layer. Negative values will wrap around.
            causal: If set, use a causal attention mask.
            reader: The reader to use for reading the audio file.

        Returns:
            The hidden states for the given waveform, with shape (B, T, D)
        """
        props = get_audio_props(path, reader=reader)
        effects: list[tuple[str, str]] = [("gain", "-n"), ("channels", "1")]
        if props.sample_rate != self.sample_rate:
            effects.append(("rate", str(self.sample_rate)))

        chunk_length = round(chunk_length_sec * self.sample_rate)
        with torch.inference_mode(), self.device.autocast_context():
            feat = []
            for waveform_chunk in read_audio(
                path,
                chunk_length=chunk_length,
                sampling_rate=self.sample_rate,
                reader=reader,
            ):
                waveform_tensor = torch.from_numpy(waveform_chunk).to(torch.float32)
                waveform_tensor, _ = ta_sox.apply_effects_tensor(waveform_tensor, props.sample_rate, effects)
                chans, _ = waveform_tensor.shape
                assert chans == 1, f"Expected mono-channel audio, got {chans} channels"
                x = self.device.tensor_to(waveform_tensor)

                if self.model.pre_normalize:
                    x = F.layer_norm(x, x.shape)

                feat_chunk = self.model.forward(x, causal=causal, output_layer=output_layer)
                if self.kmeans is not None:
                    feat_chunk = self.kmeans.forward(feat_chunk)
                feat.append(feat_chunk.cpu())

        return torch.cat(feat, 1).squeeze(0)


EXCLUDE_KEYS = {"masked_spec_embed", ".weight", ".bias"}


def _load_pretrained_hubert(
    size: PretrainedHubertSize,
    ckpt_url: str,
    sha256: str,
    config: HubertConfig,
    remove_prefix: str | None = None,
    load_weights: bool = True,
) -> Hubert:
    with Timer("building empty model", spinner=True):
        model = Hubert(config)

    # Loads the model weights.
    if load_weights:
        model_fname = f"{size}.bin"

        with Timer("downloading checkpoint"):
            model_path = ensure_downloaded(ckpt_url, "hubert", model_fname, sha256=sha256)

        with Timer("loading checkpoint", spinner=True):
            ckpt = torch.load(model_path, map_location="cpu")
            if remove_prefix:
                ckpt = {k[len(remove_prefix) :]: v for k, v in ckpt.items()}
            ckpt = {k: v for k, v in ckpt.items() if k not in EXCLUDE_KEYS}
            model.load_state_dict(ckpt)

    return model


def pretrained_hubert(size: PretrainedHubertSize, load_weights: bool = True) -> Hubert:
    match size:
        case "base":
            return _load_pretrained_hubert(
                size,
                ckpt_url="https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin",
                sha256="062249fffb353eab67547a2fbc129f7c31a2f459faf641b19e8fb007cc5c48ad",
                config=HubertConfig(
                    vocab_size=32,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    hidden_act="gelu",
                    hidden_dropout=0.1,
                    activation_dropout=0.1,
                    feat_proj_layer_norm=True,
                    feat_proj_dropout=0.0,
                    layer_norm_eps=1e-5,
                    feat_extract_norm="group",
                    feat_extract_dropout=0.0,
                    feat_extract_activation="gelu",
                    conv_dim=(512, 512, 512, 512, 512, 512, 512),
                    conv_stride=(5, 2, 2, 2, 2, 2, 2),
                    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
                    num_conv_pos_embeddings=128,
                    num_conv_pos_embedding_groups=16,
                    conv_bias=False,
                    do_stable_layer_norm=False,
                    pre_normalize=False,
                ),
                load_weights=load_weights,
            )

        case "large":
            return _load_pretrained_hubert(
                size,
                ckpt_url="https://huggingface.co/facebook/hubert-large-ls960-ft/resolve/main/pytorch_model.bin",
                sha256="9cf43abec3f0410ad6854afa4d376c69ccb364b48ddddfd25c4c5aa16398eab0",
                remove_prefix="hubert.",
                config=HubertConfig(
                    vocab_size=32,
                    hidden_size=1024,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    intermediate_size=4096,
                    hidden_act="gelu",
                    hidden_dropout=0.1,
                    activation_dropout=0.1,
                    feat_proj_layer_norm=True,
                    feat_proj_dropout=0.0,
                    layer_norm_eps=1e-5,
                    feat_extract_norm="layer",
                    feat_extract_dropout=0.0,
                    feat_extract_activation="gelu",
                    conv_dim=(512, 512, 512, 512, 512, 512, 512),
                    conv_stride=(5, 2, 2, 2, 2, 2, 2),
                    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
                    num_conv_pos_embeddings=128,
                    num_conv_pos_embedding_groups=16,
                    conv_bias=True,
                    do_stable_layer_norm=True,
                    pre_normalize=True,
                ),
                load_weights=load_weights,
            )

        case "extra_large":
            return _load_pretrained_hubert(
                size,
                ckpt_url="https://huggingface.co/facebook/hubert-xlarge-ll60k/resolve/main/pytorch_model.bin",
                sha256="6131dc27f4508595daa1a13fec4aa1f6b4a579b5d93550bae26c13a83221f8a7",
                config=HubertConfig(
                    vocab_size=32,
                    hidden_size=1280,
                    num_hidden_layers=48,
                    num_attention_heads=16,
                    intermediate_size=5120,
                    hidden_act="gelu",
                    hidden_dropout=0.1,
                    activation_dropout=0.1,
                    feat_proj_layer_norm=True,
                    feat_proj_dropout=0.0,
                    layer_norm_eps=1e-5,
                    feat_extract_norm="layer",
                    feat_extract_dropout=0.0,
                    feat_extract_activation="gelu",
                    conv_dim=(512, 512, 512, 512, 512, 512, 512),
                    conv_stride=(5, 2, 2, 2, 2, 2, 2),
                    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
                    num_conv_pos_embeddings=128,
                    num_conv_pos_embedding_groups=16,
                    conv_bias=True,
                    do_stable_layer_norm=True,
                    pre_normalize=True,
                ),
                load_weights=load_weights,
            )

        case _:
            raise NotImplementedError(f"Invalid size: {size}")


def test_hubert_adhoc() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=str, choices=get_args(PretrainedHubertSize))
    parser.add_argument("-t", "--tsz", type=int, default=22400)
    parser.add_argument("-n", "--no-load-weights", default=False, action="store_true")
    parser.add_argument("-c", "--causal", default=False, action="store_true")
    args = parser.parse_args()

    configure_logging()

    # Loads the model and moves to the right device.
    model = pretrained_hubert(size=cast(PretrainedHubertSize, args.size), load_weights=not args.no_load_weights)
    predictor = model.predictor()

    # Test the model on a random waveform.
    y = predictor.predict(torch.randn(1, args.tsz), causal=args.causal)
    assert (args.tsz // 320) == y.shape[1] + 1


if __name__ == "__main__":
    # python -m pretrained.hubert
    test_hubert_adhoc()
