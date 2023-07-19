# mypy: disable-error-code="import"
"""Defines a pre-trained Tacotron2 model.

This combines a Tacotron2 model with a HiFiGAN vocoder to produce an
end-to-end TTS model, adapted to be fine-tunable.

.. highlight:: python
.. code-block:: python

    from pretrained.tacotron2 import pretrained_tacotron2_tts

    tts = pretrained_tacotron2_tts()
    audio, states = tts.generate("Hello, world!")
    write_audio([audio])

You can also interact with this model directly through the command line:

.. highlight:: python
.. code-block:: python

    python -m pretrained.tacotron2 'Hello, world!'

The two parts of the model can be trained separately, including using LoRA
fine-tuning.

Using this model requires the following additional dependencies:

- ``inflect``
- ``ftfy``

Additionally, to generate STFTs for training the model, you will need
to install ``librosa``. If you want to play audio for the demo, you should
also install ``sounddevice``.
"""

import argparse
import functools
import html
import logging
import re
from dataclasses import dataclass
from math import sqrt
from numbers import Number
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from ml.core.config import conf_field
from ml.models.base import BaseModel, BaseModelConfig
from ml.models.lora import freeze_non_lora_, maybe_lora, reset_lora_weights_
from ml.utils.audio import write_audio
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.device.auto import detect_device
from ml.utils.device.base import base_device
from ml.utils.large_models import init_empty_weights, meta_to_empty_func
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from pretrained.vocoder import Vocoder, VocoderType, pretrained_vocoder

logger = logging.getLogger(__name__)

TACOTRON_CKPT_URL = "https://drive.google.com/open?id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA"


class Normalizer:
    def __init__(self) -> None:
        super().__init__()

        try:
            import inflect
        except ImportError:
            raise ImportError("Number normalization requires the inflect package; pip install inflect")

        self.inflect_engine = inflect.engine()
        self.comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
        self.decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
        self.pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
        self.dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
        self.ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
        self.number_re = re.compile(r"[0-9]+")

        self.abbr_re = [
            (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
            for x in [
                ("mrs", "misess"),
                ("mr", "mister"),
                ("dr", "doctor"),
                ("st", "saint"),
                ("co", "company"),
                ("jr", "junior"),
                ("maj", "major"),
                ("gen", "general"),
                ("drs", "doctors"),
                ("rev", "reverend"),
                ("lt", "lieutenant"),
                ("hon", "honorable"),
                ("sgt", "sergeant"),
                ("capt", "captain"),
                ("esq", "esquire"),
                ("ltd", "limited"),
                ("col", "colonel"),
                ("ft", "fort"),
            ]
        ]

        self.whitespace_re = re.compile(r"\s+")

    def _remove_commas(self, m: re.Match) -> str:
        return m.group(1).replace(",", "")

    def _expand_decimal_point(self, m: re.Match) -> str:
        return m.group(1).replace(".", " point ")

    def _expand_dollars(self, m: re.Match) -> str:
        match = m.group(1)
        parts = match.split(".")
        if len(parts) > 2:
            return match + " dollars"  # Unexpected format
        dollars = int(parts[0]) if parts[0] else 0
        cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if dollars and cents:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            cent_unit = "cent" if cents == 1 else "cents"
            return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
        elif dollars:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            return "%s %s" % (dollars, dollar_unit)
        elif cents:
            cent_unit = "cent" if cents == 1 else "cents"
            return "%s %s" % (cents, cent_unit)
        else:
            return "zero dollars"

    def _expand_ordinal(self, m: re.Match) -> str:
        return cast(str, self.inflect_engine.number_to_words(m.group(0)))

    def _expand_number(self, m: re.Match) -> str:
        num = int(m.group(0))
        if num > 1000 and num < 3000:
            if num == 2000:
                return "two thousand"
            elif num > 2000 and num < 2010:
                return "two thousand " + cast(str, self.inflect_engine.number_to_words(cast(Number, num % 100)))
            elif num % 100 == 0:
                return cast(str, self.inflect_engine.number_to_words(cast(Number, num // 100))) + " hundred"
            else:
                out = cast(str, self.inflect_engine.number_to_words(cast(Number, num), andword="", zero="oh", group=2))
                return out.replace(", ", " ")
        return cast(str, self.inflect_engine.number_to_words(cast(Number, num), andword=""))

    def __call__(self, text: str) -> str:
        text = re.sub(self.comma_number_re, self._remove_commas, text)
        text = re.sub(self.pounds_re, r"\1 pounds", text)
        text = re.sub(self.dollars_re, self._expand_dollars, text)
        text = re.sub(self.decimal_number_re, self._expand_decimal_point, text)
        text = re.sub(self.ordinal_re, self._expand_ordinal, text)
        text = re.sub(self.number_re, self._expand_number, text)
        for regex, replacement in self.abbr_re:
            text = re.sub(regex, replacement, text)
        text = re.sub(self.whitespace_re, " ", text)
        return text


@functools.lru_cache()
def text_clean_func(lower: bool = True) -> Callable[[str], str]:
    try:
        import ftfy

        ftfy_fix: Callable[[str], str] = ftfy.fix_text
    except ImportError:
        logger.warning("Please install ftfy: pip install ftfy")

        def ftfy_fix(x: str) -> str:
            return x

    try:
        normalizer: Callable[[str], str] = Normalizer()
    except ImportError:
        logger.warning("Please install inflect and make sure it can be imported: pip install inflect")

        def normalizer(x: str) -> str:
            return x

    def _clean(text: str) -> str:
        text = ftfy_fix(text)
        text = html.unescape(html.unescape(text))
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        if lower:
            text = text.lower()
        text = normalizer(text)
        return text

    return _clean


def get_mask_from_lengths(lengths: Tensor) -> Tensor:
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, dtype=torch.int32, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = mask <= 0
    return mask


class LinearNorm(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        w_init_gain: str = "linear",
        lora_rank: int | None = None,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))
        self.linear_layer = maybe_lora(linear_layer, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain: str = "linear",
        lora_rank: int | None = None,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.conv = maybe_lora(conv, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal: Tensor) -> Tensor:
        return self.conv(signal)


class LocationLayer(nn.Module):
    def __init__(
        self,
        attention_n_filters: int,
        attention_kernel_size: int,
        attention_dim: int,
        lora_rank: int | None = None,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2,
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.location_dense = LinearNorm(
            attention_n_filters,
            attention_dim,
            bias=False,
            w_init_gain="tanh",
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    def forward(self, attention_weights_cat: Tensor) -> Tensor:
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(
        self,
        attention_rnn_dim: int,
        embedding_dim: int,
        attention_dim: int,
        attention_location_n_filters: int,
        attention_location_kernel_size: int,
        lora_rank: int | None = None,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.query_layer = LinearNorm(
            attention_rnn_dim,
            attention_dim,
            bias=False,
            w_init_gain="tanh",
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.memory_layer = LinearNorm(
            embedding_dim,
            attention_dim,
            bias=False,
            w_init_gain="tanh",
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.v = LinearNorm(
            attention_dim,
            1,
            bias=False,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query: Tensor, processed_memory: Tensor, attention_weights_cat: Tensor) -> Tensor:
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))
        energies = energies.squeeze(-1)
        return energies

    def forward(
        self,
        attn_hid_state: Tensor,
        memory: Tensor,
        proc_memory: Tensor,
        attn_weights_cat: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        alignment = self.get_alignment_energies(attn_hid_state, proc_memory, attn_weights_cat)

        if mask is not None:
            alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    __constants__ = ["dropout", "dropout_always_on"]

    def __init__(
        self,
        in_dim: int = 80,
        sizes: list[int] = [256, 256],
        dropout: float = 0.5,
        lora_rank: int | None = None,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        dropout_always_on: bool = True,
    ) -> None:
        super().__init__()

        in_sizes = [in_dim] + sizes[:-1]
        layers = [
            LinearNorm(
                in_size,
                out_size,
                bias=False,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            for (in_size, out_size) in zip(in_sizes, sizes)
        ]
        self.layers = nn.ModuleList(layers)
        self.dropout = dropout
        self.dropout_always_on = dropout_always_on

    def forward(self, x: Tensor) -> Tensor:
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=self.dropout, training=self.training or self.dropout_always_on)
        return x


@dataclass
class PostnetConfig:
    n_mel_channels: int = conf_field(80, help="Number of mel channels")
    emb_dim: int = conf_field(512, help="Postnet embedding dimension")
    kernel_size: int = conf_field(5, help="Postnet kernel size")
    n_convolutions: int = conf_field(5, help="Number of postnet convolutions")
    lora_rank: int | None = conf_field(None, help="LoRA rank")
    lora_alpha: float = conf_field(1.0, help="LoRA alpha")
    lora_dropout: float = conf_field(0.0, help="LoRA dropout")


class Postnet(nn.Module):
    def __init__(self, config: PostnetConfig) -> None:
        super().__init__()

        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    config.n_mel_channels,
                    config.emb_dim,
                    kernel_size=config.kernel_size,
                    stride=1,
                    padding=(config.kernel_size - 1) // 2,
                    dilation=1,
                    w_init_gain="tanh",
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                ),
                nn.BatchNorm1d(config.emb_dim),
            )
        )

        for _ in range(1, config.n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        config.emb_dim,
                        config.emb_dim,
                        kernel_size=config.kernel_size,
                        stride=1,
                        padding=int((config.kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                        lora_rank=config.lora_rank,
                        lora_alpha=config.lora_alpha,
                        lora_dropout=config.lora_dropout,
                    ),
                    nn.BatchNorm1d(config.emb_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    config.emb_dim,
                    config.n_mel_channels,
                    kernel_size=config.kernel_size,
                    stride=1,
                    padding=int((config.kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                ),
                nn.BatchNorm1d(config.n_mel_channels),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, conv in enumerate(self.convolutions):
            if i < len(self.convolutions) - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)
        return x


@dataclass
class EncoderConfig:
    emb_dim: int = conf_field(512, help="Encoder embedding dimension")
    kernel_size: int = conf_field(5, help="Encoder kernel size")
    n_convolutions: int = conf_field(3, help="Number of encoder convolutions")
    lora_rank: int | None = conf_field(None, help="LoRA rank")
    lora_alpha: float = conf_field(1.0, help="LoRA alpha")
    lora_dropout: float = conf_field(0.0, help="LoRA dropout")
    freeze_bn: bool = conf_field(False, help="Freeze batch normalization")
    speaker_emb_dim: int | None = conf_field(None, help="Speaker embedding dimension")


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()

        convolutions = []
        for _ in range(config.n_convolutions):
            conv_norm = ConvNorm(
                config.emb_dim,
                config.emb_dim,
                kernel_size=config.kernel_size,
                stride=1,
                padding=int((config.kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="relu",
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )
            batch_norm = nn.BatchNorm1d(config.emb_dim)
            if config.freeze_bn:
                batch_norm.requires_grad_(False)
            conv_layer = nn.Sequential(conv_norm, batch_norm)
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        lstm = nn.LSTM(
            config.emb_dim,
            int(config.emb_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm = maybe_lora(lstm, r=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout)

    def forward(self, x: Tensor, input_lengths: Tensor, speaker_emb: Tensor | None = None) -> Tensor:
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        if speaker_emb is not None:
            outputs += speaker_emb.unsqueeze(1)

        return outputs

    def infer(self, x: Tensor, input_lengths: Tensor, speaker_emb: Tensor | None = None) -> Tensor:
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        if speaker_emb is not None:
            outputs += speaker_emb.unsqueeze(1)

        return outputs


@dataclass
class DecoderConfig:
    n_mel_channels: int = conf_field(80, help="Number of mel channels")
    n_frames_per_step: int = conf_field(1, help="Number of frames processed per step")
    encoder_emb_dim: int = conf_field(512, help="Encoder embedding dimension")
    attention_dim: int = conf_field(128, help="Attention dimension")
    attention_location_n_filters: int = conf_field(32, help="Number of filters for location-sensitive attention")
    attention_location_kernel_size: int = conf_field(31, help="Kernel size for location-sensitive attention")
    attention_rnn_dim: int = conf_field(1024, help="Attention RNN dimension")
    decoder_rnn_dim: int = conf_field(1024, help="Decoder RNN dimension")
    prenet_dim: int = conf_field(256, help="Prenet dimension")
    prenet_dropout: bool = conf_field(0.5, help="Whether to use dropout in prenet layers")
    max_decoder_steps: int = conf_field(1000, help="Maximum decoder steps during inference")
    gate_threshold: float = conf_field(0.5, help="Probability threshold for stop token")
    p_attention_dropout: float = conf_field(0.1, help="Dropout probability for attention LSTM")
    p_decoder_dropout: float = conf_field(0.1, help="Dropout probability for decoder LSTM")
    prenet_dropout_always_on: bool = conf_field(True, help="If set, prenet dropout is always on")
    lora_rank: int | None = conf_field(None, help="LoRA rank")
    lora_alpha: float = conf_field(1.0, help="LoRA alpha")
    lora_dropout: float = conf_field(0.0, help="LoRA dropout")


class DecoderStates(NamedTuple):
    attn_h: Tensor
    attn_c: Tensor
    dec_h: Tensor
    dec_c: Tensor
    attn_weights: Tensor
    attn_weights_cum: Tensor
    attn_ctx: Tensor
    memory: Tensor
    processed_memory: Tensor
    mask: Tensor | None


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()

        self.n_mel_channels = config.n_mel_channels
        self.n_frames_per_step = config.n_frames_per_step
        self.encoder_embedding_dim = config.encoder_emb_dim
        self.attention_rnn_dim = config.attention_rnn_dim
        self.decoder_rnn_dim = config.decoder_rnn_dim
        self.prenet_dim = config.prenet_dim
        self.max_decoder_steps = config.max_decoder_steps
        self.gate_threshold = config.gate_threshold
        self.p_attention_dropout = config.p_attention_dropout
        self.p_decoder_dropout = config.p_decoder_dropout

        self.prenet = Prenet(
            config.n_mel_channels * config.n_frames_per_step,
            [config.prenet_dim, config.prenet_dim],
            config.prenet_dropout,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            dropout_always_on=config.prenet_dropout_always_on,
        )

        attention_rnn = nn.LSTMCell(
            config.prenet_dim + config.encoder_emb_dim,
            config.attention_rnn_dim,
        )
        self.attention_rnn = maybe_lora(
            attention_rnn,
            r=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )

        self.attention_layer = Attention(
            config.attention_rnn_dim,
            config.encoder_emb_dim,
            config.attention_dim,
            config.attention_location_n_filters,
            config.attention_location_kernel_size,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )

        decoder_rnn = nn.LSTMCell(
            config.attention_rnn_dim + config.encoder_emb_dim,
            config.decoder_rnn_dim,
            bias=True,
        )
        self.decoder_rnn = maybe_lora(
            decoder_rnn,
            r=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )

        self.linear_projection = LinearNorm(
            config.decoder_rnn_dim + config.encoder_emb_dim,
            config.n_mel_channels * config.n_frames_per_step,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )

        self.gate_layer = LinearNorm(
            config.decoder_rnn_dim + config.encoder_emb_dim,
            1,
            bias=True,
            w_init_gain="sigmoid",
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )

    def get_go_frame(self, memory: Tensor) -> Tensor:
        return memory.new_zeros(memory.shape[0], self.n_mel_channels * self.n_frames_per_step)

    def initialize_decoder_states(self, memory: Tensor, mask: Tensor | None) -> DecoderStates:
        bsz, max_tsz, *_ = memory.size()

        attn_h = memory.new_zeros(bsz, self.attention_rnn_dim)
        attn_c = memory.new_zeros(bsz, self.attention_rnn_dim)

        dec_hid = memory.new_zeros(bsz, self.decoder_rnn_dim)
        dec_cell = memory.new_zeros(bsz, self.decoder_rnn_dim)

        attn_weights = memory.new_zeros(bsz, max_tsz)
        attn_weights_cum = memory.new_zeros(bsz, max_tsz)
        attn_ctx = memory.new_zeros(bsz, self.encoder_embedding_dim)

        processed_memory = self.attention_layer.memory_layer(memory)

        return DecoderStates(
            attn_h=attn_h,
            attn_c=attn_c,
            dec_h=dec_hid,
            dec_c=dec_cell,
            attn_weights=attn_weights,
            attn_weights_cum=attn_weights_cum,
            attn_ctx=attn_ctx,
            memory=memory,
            processed_memory=processed_memory,
            mask=mask,
        )

    def parse_decoder_inputs(self, decoder_inputs: Tensor) -> Tensor:
        # (bsz, n_mel_channels, tsz_out) -> (bsz, tsz_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step),
            -1,
        )
        # (bsz, tsz_out, n_mel_channels) -> (tsz_out, bsz, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(
        self,
        mel_outputs: list[Tensor],
        gate_outputs: list[Tensor],
        alignments: list[Tensor],
        states: DecoderStates,
    ) -> tuple[Tensor, Tensor, Tensor, DecoderStates]:
        alignments = torch.stack(alignments, dim=1)
        gate_outputs = torch.stack(gate_outputs, dim=1)
        mel_outputs = torch.stack(mel_outputs, dim=-1)
        return mel_outputs, gate_outputs, alignments, states

    def decode(self, decoder_input: Tensor, states: DecoderStates) -> tuple[Tensor, Tensor, Tensor, DecoderStates]:
        attn_h, attn_c, dec_h, dec_c, attn_weights, attn_weights_cum, attn_ctx, memory, processed_memory, mask = states

        cell_input = torch.cat((decoder_input, attn_ctx), -1)
        attn_h, attn_c = self.attention_rnn(cell_input, (attn_h, attn_c))
        attn_h = F.dropout(attn_h, self.p_attention_dropout, self.training)

        attn_weights_cat = torch.cat((attn_weights.unsqueeze(1), attn_weights_cum.unsqueeze(1)), dim=1)
        attn_ctx, attn_weights = self.attention_layer(attn_h, memory, processed_memory, attn_weights_cat, mask)

        attn_weights_cum = attn_weights_cum + attn_weights
        decoder_input = torch.cat((attn_h, attn_ctx), -1)
        dec_h, dec_c = self.decoder_rnn(decoder_input, (dec_h, dec_c))
        dec_h = F.dropout(dec_h, self.p_decoder_dropout, self.training)

        dec_h_attn_ctx = torch.cat((dec_h, attn_ctx), dim=1)
        dec_out = self.linear_projection(dec_h_attn_ctx)

        gate_pred = self.gate_layer(dec_h_attn_ctx)

        new_states = DecoderStates(
            attn_h=attn_h,
            attn_c=attn_c,
            dec_h=dec_h,
            dec_c=dec_c,
            attn_weights=attn_weights,
            attn_weights_cum=attn_weights_cum,
            attn_ctx=attn_ctx,
            memory=memory,
            processed_memory=processed_memory,
            mask=mask,
        )

        return dec_out, gate_pred, attn_weights, new_states

    def forward(
        self,
        memory: Tensor,
        dec_ins: Tensor,
        memory_lengths: Tensor,
        states: DecoderStates | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, DecoderStates]:
        dec_in = self.get_go_frame(memory).unsqueeze(0)
        dec_ins = self.parse_decoder_inputs(dec_ins)
        dec_ins = torch.cat((dec_in, dec_ins), dim=0)
        prenet_ins = self.prenet(dec_ins)

        states = self.initialize_decoder_states(memory, mask=get_mask_from_lengths(memory_lengths))

        mel_outs: list[Tensor] = []
        gate_outs: list[Tensor] = []
        alignments: list[Tensor] = []
        while len(mel_outs) < prenet_ins.size(0) - 1:
            prenet_in = prenet_ins[len(mel_outs)]
            mel_out, gate_out, attn_weights, states = self.decode(prenet_in, states)
            mel_outs += [mel_out.squeeze(1)]
            gate_outs += [gate_out.squeeze(1)]
            alignments += [attn_weights]

        return self.parse_decoder_outputs(mel_outs, gate_outs, alignments, states)

    def infer(
        self,
        memory: Tensor,
        memory_lengths: Tensor,
        states: DecoderStates | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, DecoderStates]:
        dec_in = self.get_go_frame(memory)
        if states is None:
            states = self.initialize_decoder_states(memory, mask=get_mask_from_lengths(memory_lengths))

        mel_outs: list[Tensor] = []
        gate_outs: list[Tensor] = []
        alignments: list[Tensor] = []
        while True:
            prenet_in = self.prenet(dec_in)
            mel_out, gate_out, alignment, states = self.decode(prenet_in, states)
            mel_outs += [mel_out.squeeze(1)]
            gate_outs += [gate_out]
            alignments += [alignment]
            if (torch.sigmoid(gate_out) > self.gate_threshold).all():
                break
            elif len(mel_outs) == self.max_decoder_steps:
                logger.warning("Warning! Reached max decoder steps %d", self.max_decoder_steps)
                break
            dec_in = mel_out

        return self.parse_decoder_outputs(mel_outs, gate_outs, alignments, states)


def window_sumsquare(
    window: str | float,
    n_frames: int,
    hop_length: int = 200,
    win_length: int = 800,
    n_fft: int = 800,
    dtype: type = np.float32,
    norm: float | None = None,
) -> np.ndarray:
    try:
        from scipy.signal import get_window
    except ImportError:
        raise ImportError("Please install scipy to use this module: pip install scipy")

    try:
        import librosa.util
    except ImportError:
        raise ImportError("Please install librosa to use this module: pip install librosa")

    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x: np.ndarray = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa.util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa.util.pad_center(win_sq, size=n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]

    return x


def griffin_lim(magnitudes: Tensor, stft_fn: "STFT", n_iters: int = 30) -> Tensor:
    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.from_numpy(angles)
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for _ in range(n_iters):
        _, angles_tensor = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles_tensor).squeeze(1)
    return signal


def dynamic_range_compression(x: Tensor, c: int | float = 1, clip_val: float = 1e-5) -> Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * c)


def dynamic_range_decompression(x: Tensor, c: int | float = 1) -> Tensor:
    return torch.exp(x) / c


class STFT(nn.Module):
    forward_basis: Tensor
    inverse_basis: Tensor

    def __init__(
        self,
        filter_length: int = 800,
        hop_length: int = 200,
        win_length: int = 800,
        window: str = "hann",
    ) -> None:
        try:
            from scipy.signal import get_window
        except ImportError:
            raise ImportError("Please install scipy to use this module: pip install scipy")

        try:
            import librosa.util
        except ImportError:
            raise ImportError("Please install librosa to use this module: pip install librosa")

        super().__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length

        with Timer("getting fourier basis"):
            fourier_basis = np.fft.fft(np.eye(self.filter_length))
            cutoff = int((self.filter_length / 2 + 1))
            fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])

        with Timer("getting forward and inverse basis"):
            forward_basis: Tensor = torch.FloatTensor(fourier_basis[:, None, :])
            inverse_basis: Tensor = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            with Timer("applying window"):
                assert filter_length >= win_length

                # get window and zero center pad it to filter_length
                fft_window = get_window(window, win_length, fftbins=True)
                fft_window = librosa.util.pad_center(fft_window, size=filter_length)
                fft_window = torch.from_numpy(fft_window).float()

                # window the bases
                forward_basis *= fft_window[None, None, :]
                inverse_basis *= fft_window[None, None, :]

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data: Tensor) -> tuple[Tensor, Tensor]:
        num_batches, num_samples = input_data.shape

        self.num_samples = num_samples

        # Similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        ).squeeze(1)

        forward_transform = F.conv1d(input_data, self.forward_basis, stride=self.hop_length, padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part.data, real_part.data)

        return magnitude, phase

    def inverse(self, magnitude: Tensor, phase: Tensor) -> Tensor:
        try:
            import librosa.util
        except ImportError:
            raise ImportError("Please install librosa to use this module: pip install librosa")

        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum_np = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )

            # Remove modulation effects.
            approx_nonzero_indices = torch.from_numpy(np.where(window_sum_np > librosa.util.tiny(window_sum_np))[0])
            window_sum = torch.from_numpy(window_sum_np)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # Scale by hop ratio.
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    def forward(self, input_data: Tensor) -> Tensor:
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TacotronSTFT(nn.Module):
    mel_basis: Tensor

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 80,
        sampling_rate: int = 16000,
        mel_fmin: float = 0.0,
        mel_fmax: float = 8000.0,
    ) -> None:
        super().__init__()

        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)

        try:
            from librosa.filters import mel as librosa_mel_fn
        except ImportError:
            raise ImportError("Please install librosa: pip install librosa")

        mel_basis_np = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        mel_basis = torch.from_numpy(mel_basis_np).float()
        self.register_buffer("mel_basis", mel_basis)

    @torch.no_grad()
    def spectral_normalize(self, magnitudes: Tensor) -> Tensor:
        output = dynamic_range_compression(magnitudes)
        return output

    @torch.no_grad()
    def spectral_de_normalize(self, magnitudes: Tensor) -> Tensor:
        output = dynamic_range_decompression(magnitudes)
        return output

    @torch.no_grad()
    def mel_spectrogram(self, y: Tensor) -> Tensor:
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        magnitudes, _ = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


@dataclass
class TacotronConfig(BaseModelConfig):
    mask_padding: bool = conf_field(False, help="Mask padding in loss computation")
    n_mel_channels: int = conf_field(80, help="Number of bins in mel-spectrograms")
    n_symbols: int = conf_field(148, help="Number of symbols in dictionary")
    symbols_emb_dim: int = conf_field(512, help="Input embedding dimension")
    n_frames_per_step: int = conf_field(1, help="Number of frames processed per step")
    symbols_emb_dropout: float = conf_field(0.1, help="Dropout rate for symbol embeddings")
    encoder: EncoderConfig = conf_field(EncoderConfig(), help="Encoder configuration")
    decoder: DecoderConfig = conf_field(DecoderConfig(), help="Decoder configuration")
    postnet: PostnetConfig = conf_field(PostnetConfig(), help="Postnet configuration")


class Tacotron(BaseModel):
    def __init__(self, config: TacotronConfig) -> None:
        super().__init__(config)

        self.mask_padding = config.mask_padding
        self.n_mel_channels = config.n_mel_channels
        self.n_frames_per_step = config.n_frames_per_step
        self.embedding = nn.Embedding(config.n_symbols, config.symbols_emb_dim)
        std = sqrt(2.0 / (config.n_symbols + config.symbols_emb_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)
        self.postnet = Postnet(config.postnet)

    def parse_output(
        self,
        outputs: tuple[Tensor, Tensor, Tensor, Tensor, DecoderStates],
        output_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, DecoderStates]:
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(
        self,
        inputs: tuple[Tensor, Tensor, Tensor, Tensor],
        states: DecoderStates | None = None,
        speaker_emb: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, DecoderStates]:
        text_inputs, text_lengths, mels, output_lengths = inputs
        embedded_inputs = self.embedding.forward(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder.forward(embedded_inputs, text_lengths, speaker_emb)
        mel_outputs, gate_outputs, alignments, states = self.decoder(encoder_outputs, mels, text_lengths, states)
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)
        return self.parse_output((mel_outputs, mel_outputs_postnet, gate_outputs, alignments, states), output_lengths)

    def infer(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
        states: DecoderStates | None = None,
        speaker_emb: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, DecoderStates]:
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.infer(embedded_inputs, input_lengths, speaker_emb)
        mel_outputs, gate_outputs, alignments, states = self.decoder.infer(encoder_outputs, input_lengths, states)
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)
        return self.parse_output((mel_outputs, mel_outputs_postnet, gate_outputs, alignments, states))


class Tokenizer:
    def __init__(self) -> None:
        super().__init__()

        pad = "_"
        punctuation = "!'(),.:;? "
        specials = "-"
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

        # Prepend "@" to ARPAbet symbols to ensure uniqueness.
        valid_symbols = (
            "AA:AA0:AA1:AA2:AE:AE0:AE1:AE2:AH:AH0:AH1:AH2:AO:AO0:AO1:AO2:AW"
            ":AW0:AW1:AW2:AY:AY0:AY1:AY2:B:CH:D:DH:EH:EH0:EH1:EH2:ER:ER0:ER1"
            ":ER2:EY:EY0:EY1:EY2:F:G:HH:IH:IH0:IH1:IH2:IY:IY0:IY1:IY2:JH:K:L"
            ":M:N:NG:OW:OW0:OW1:OW2:OY:OY0:OY1:OY2:P:R:S:SH:T:TH:UH:UH0:UH1"
            ":UH2:UW:UW0:UW1:UW2:V:W:Y:Z:ZH"
        )
        arpabet = ["@" + s for s in valid_symbols.split(":")]

        # Gets the symbol conversion dictionary.
        self.symbols = [pad] + list(specials) + list(punctuation) + list(letters) + arpabet
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {v: k for k, v in self.symbol_to_id.items()}

        self.curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

    def __call__(self, text: str) -> Tensor:
        clean_func = text_clean_func()

        sequence: list[int] = []

        def _should_keep_symbol(s: str) -> bool:
            return s in self.symbol_to_id and s != "_" and s != "~"

        def _symbols_to_sequence(symbols: str | list[str]) -> list[int]:
            return [self.symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]

        def _arpabet_to_sequence(text: str) -> list[int]:
            return _symbols_to_sequence(["@" + s for s in text.split()])

        while len(text):
            m = self.curly_re.match(text)
            if not m:
                sequence += _symbols_to_sequence(clean_func(text))
                break
            sequence += _symbols_to_sequence(clean_func(m.group(1)))
            sequence += _arpabet_to_sequence(m.group(2))
            text = m.group(3)

        return torch.tensor(sequence, dtype=torch.int32)


def ensure_tacotron_downloaded() -> Path:
    with Timer("downloading checkpoint"):
        return ensure_downloaded(TACOTRON_CKPT_URL, "tacotron2", "weights_tacotron.pth")


def pretrained_tacotron2(
    *,
    pretrained: bool = True,
    lora_rank: int | None = None,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0,
    lora_encoder: bool = True,
    lora_decoder: bool = True,
    lora_postnet: bool = True,
    device: torch.device | None = None,
    prenet_dropout: bool = True,
    num_tokens: int | None = None,
) -> Tacotron:
    """Loads the pretrained Tacotron2 model.

    Args:
        pretrained: Whether to load the pretrained weights.
        lora_rank: The LoRA rank to use, if LoRA is desired.
        lora_alpha: The LoRA alpha to use, if LoRA is desired.
        lora_dropout: The LoRA dropout to use, if LoRA is desired.
        lora_encoder: Whether to use LoRA in the encoder.
        lora_decoder: Whether to use LoRA in the decoder.
        lora_postnet: Whether to use LoRA in the postnet.
        device: The device to load the weights onto.
        prenet_dropout: Whether to use always apply dropout in the PreNet.
        num_tokens: The number of tokens in the vocabulary.

    Returns:
        The pretrained Tacotron model.
    """
    config = TacotronConfig()
    if lora_encoder:
        config.encoder.lora_rank = lora_rank
        config.encoder.lora_alpha = lora_alpha
        config.encoder.lora_dropout = lora_dropout
    if lora_decoder:
        config.decoder.lora_rank = lora_rank
        config.decoder.lora_alpha = lora_alpha
        config.decoder.lora_dropout = lora_dropout
    if lora_postnet:
        config.postnet.lora_rank = lora_rank
        config.postnet.lora_alpha = lora_alpha
        config.postnet.lora_dropout = lora_dropout
    config.decoder.prenet_dropout_always_on = prenet_dropout
    if num_tokens is not None:
        config.n_symbols = num_tokens

    if not pretrained:
        return Tacotron(config)

    with Timer("initializing model", spinner=True), init_empty_weights():
        model = Tacotron(config)

    with Timer("downloading checkpoint", spinner=True):
        filepath = ensure_tacotron_downloaded()

    with Timer("loading checkpoint", spinner=True):
        if device is None:
            device = torch.device("cpu")
        ckpt = torch.load(filepath, map_location=device)
        model._apply(meta_to_empty_func(device))
        model.load_state_dict({k: v for k, v in ckpt["state_dict"].items()})
        reset_lora_weights_(model)

    with Timer("freezing weights", spinner=True):
        if lora_rank is not None:
            freeze_non_lora_(model)
        if num_tokens is not None:
            model.embedding.requires_grad_(True)

    return model


def tacotron_stft(
    filter_length: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mel_channels: int = 80,
    sampling_rate: int = 16000,
    mel_fmin: float = 0.0,
    mel_fmax: float = 8000.0,
) -> TacotronSTFT:
    """Returns an STFT module for training the Tacotron model.

    Args:
        filter_length: The length of the filters used for the STFT.
        hop_length: The hop length of the STFT.
        win_length: The window length of the STFT.
        n_mel_channels: The number of mel channels.
        sampling_rate: The sampling rate of the audio.
        mel_fmin: The minimum frequency of the mel filterbank.
        mel_fmax: The maximum frequency of the mel filterbank.

    Returns:
        The STFT module.
    """
    return TacotronSTFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        sampling_rate=sampling_rate,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax,
    )


def tacotron_tokenizer() -> Tokenizer:
    return Tokenizer()


class TTS:
    def __init__(
        self,
        tacotron: Tacotron,
        vocoder: Vocoder,
        *,
        device: base_device | None = None,
    ) -> None:
        """Provides an API for doing text-to-speech.

        Note that this module is not an `nn.Module`, so you can use it in your
        module without worrying aobut storing all the weights on accident.

        Args:
            tacotron: The Tacotron model.
            vocoder: The vocoder model.
            device: The device to load the weights onto.
        """
        super().__init__()

        self.device = detect_device() if device is None else device
        self.tacotron = tacotron.eval()
        self.vocoder = vocoder.eval()
        self.vocoder.remove_weight_norm()
        self.sampling_rate = self.vocoder.sampling_rate
        self.device.module_to(self.tacotron)
        self.device.module_to(self.vocoder)
        self.tokenizer = Tokenizer()

    @torch.inference_mode()
    def generate_mels(
        self,
        text: str | list[str],
        postnet: bool = True,
        states: DecoderStates | None = None,
    ) -> tuple[Tensor, DecoderStates]:
        if isinstance(text, str):
            tokens = self.tokenizer(text).unsqueeze(0)
            token_lengths = tokens.new_full((1,), tokens.shape[1], dtype=torch.int32)
        else:
            token_list = [self.tokenizer(t) for t in text]
            tokens = pad_sequence(token_list, batch_first=True, padding_value=0)
            token_lengths = tokens.new_empty((tokens.shape[0],), dtype=torch.int32)
            for i, t in enumerate(token_list):
                token_lengths[i] = t.shape[0]
        tokens, token_lengths = self.device.tensor_to(tokens), self.device.tensor_to(token_lengths)
        mel_outputs, mel_outputs_postnet, _, _, states = self.tacotron.infer(tokens, token_lengths, states)
        return mel_outputs_postnet if postnet else mel_outputs, states

    @torch.inference_mode()
    def generate_wave(self, mels: Tensor) -> Tensor:
        return self.vocoder.infer(mels)

    @torch.inference_mode()
    def generate(
        self,
        text: str | list[str],
        postnet: bool = True,
        states: DecoderStates | None = None,
    ) -> tuple[Tensor, DecoderStates]:
        mels, states = self.generate_mels(text, postnet=postnet, states=states)
        audio = self.generate_wave(mels).squeeze(0)
        return audio, states


def pretrained_tacotron2_tts(vocoder_type: VocoderType = "hifigan", *, device: base_device | None = None) -> TTS:
    tacotron = pretrained_tacotron2()
    vocoder = pretrained_vocoder(vocoder_type)
    tts = TTS(tacotron, vocoder, device=device)
    return tts


def test_tacotron_adhoc() -> None:
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, nargs="?", help="The text to synthesize.")
    parser.add_argument("-o", "--out-file", type=str, default=None, help="The output file.")
    parser.add_argument("-p", "--prompt", default="Text: ", help="The prompt to use.")
    args = parser.parse_args()

    tts = pretrained_tacotron2_tts()

    def generate_for_text(texts: Iterable[str]) -> None:
        for text in texts:
            logger.info("Generating audio for '%s'", text)
            audio, _ = tts.generate(text, postnet=True)

            if args.out_file is None:
                try:
                    import sounddevice as sd
                except ImportError:
                    raise ImportError("Please install sounddevice to play audio: pip install sounddevice")

                audio = audio.cpu().numpy().T
                sd.play(audio, tts.sampling_rate, blocking=True)

            else:
                out_path = Path(args.out_file)
                out_path.parent.mkdir(exist_ok=True)
                write_audio(iter([audio]), out_path, tts.sampling_rate)

    if args.text:
        generate_for_text([args.text])

    else:

        def gen_texts() -> Iterable[str]:
            try:
                text = input(args.prompt)
                while text:
                    yield text
                    text = input(args.prompt)
            except EOFError:
                pass

        generate_for_text(gen_texts())


if __name__ == "__main__":
    # python -m pretrained.tacotron2
    test_tacotron_adhoc()
