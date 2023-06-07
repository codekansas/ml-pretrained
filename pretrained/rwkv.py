# mypy: disable-error-code="import, override"
"""Defines a simple API for using the RWKV model.

This code is adapted from the minimimal implementation
`here <https://johanwind.github.io/2023/03/23/rwkv_details.html>`_, adapted
to be fine-tunable.

.. highlight:: python
.. code-block:: python

    from pretrained.rwkv import pretrained_rwkv

    model = pretrained_rwkv("7B")
    predictor = model.predictor()

    for token in predictor.generate("The quick brown fox jumped over the"):
        print(token)

Using the tokenizer requires installing the ``tokenizers`` library:

.. code-block:: bash

    pip install tokenizers

The choices for the model key are:

- ``"169m"``
- ``"430m"``
- ``"1.5b"``
- ``"3b"``
- ``"7b"``
- ``"14b"``
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Sequence, get_args

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ml.models.lora import maybe_lora
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.device.auto import AutoDevice
from ml.utils.device.base import BaseDevice
from ml.utils.large_models import init_empty_weights, meta_to_empty_func
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer

logger = logging.getLogger(__name__)

PretrainedRwkvKey = Literal["169m", "430m", "1.5b", "3b", "7b", "14b"]

AttentionState = tuple[Tensor, Tensor, Tensor]
FeedForwardState = Tensor
State = tuple[AttentionState, FeedForwardState]


@dataclass
class ModelArgs:
    url: str
    sha256: str
    emb_dim: int
    num_layers: int


PRETRAINED_MODEL_SIZES: dict[PretrainedRwkvKey, ModelArgs] = {
    "169m": ModelArgs(
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth",
        sha256="713c6f6137a08d3a86ab57df4f09ea03563329beb3bbabc23509d6c57aa0f9e2",
        emb_dim=768,
        num_layers=12,
    ),
    "430m": ModelArgs(
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth",
        sha256="261e6b8fef1c7c9e08a4dde31bf5caf8e79c4da38126d77977a4707de82a7f64",
        emb_dim=1024,
        num_layers=24,
    ),
    "1.5b": ModelArgs(
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-1b5/resolve/main/RWKV-4-Pile-1B5-20220929-ctx4096.pth",
        sha256="6c97043e1bb0867368249290c97a2fe8ffc5ec12ceb1b5251f4ee911f9982c23",
        emb_dim=2048,
        num_layers=24,
    ),
    "3b": ModelArgs(
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-20221110-ctx4096.pth",
        sha256="9500633f23d86fbae3cb3cbe7908b97b971e9561edf583c2c5c60b10b02bcc27",
        emb_dim=2560,
        num_layers=32,
    ),
    "7b": ModelArgs(
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-7b/resolve/main/RWKV-4-Pile-7B-20230109-ctx4096.pth",
        sha256="9ea1271b25deb6c72bd29f629147d5013cc7d7c69f9715192f6b6b92fca08f64",
        emb_dim=4096,
        num_layers=32,
    ),
    "14b": ModelArgs(
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-14b/resolve/main/RWKV-4-Pile-14B-20230313-ctx8192-test1050.pth",
        sha256="9e1b9b44f2a98124d86fe35e298f230e3a4fa7b60431962da282817ae1b0bf32",
        emb_dim=5120,
        num_layers=40,
    ),
}

TOKENIZER_URL = "https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/20B_tokenizer.json"


def get_mask(tsz: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
    """Returns the forward mask, used for training.

    Args:
        tsz: The number of timesteps in the mask
        device: The mask device
        dtype: The mask dtype

    Returns:
        The forward mask, with shape (T, T)
    """
    mask = torch.empty(tsz, tsz, device=device, dtype=dtype)
    mask.fill_(float("-inf"))
    # mask.triu_(1)
    mask.tril_(-1)
    return mask


def run_wkv(
    tsz: int,
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    last_num: Tensor,
    last_den: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Runs the core WKV computation.

    Args;
        tsz: The number of timesteps
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        last_num: The last numerator, with shape (B, 1, D)
        last_den: The last denominator, with shape (B, 1, D)
        mask: The attention mask, with shape (T, T)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next numerator and
        denominator tensors, each with shape (B, T, D)
    """
    assert w.dim() == u.dim() == 1
    assert mask is None or mask.dim() == 2
    assert k.dim() == v.dim() == last_num.dim() == last_den.dim() == 3

    t = torch.arange(tsz + 1, device=w.device)[None, :, None]
    wt = t[:, None, :-1, :] - t[:, :-1, None, :]
    w = -torch.exp(w)
    tw = w * t[:, 1:]
    twt = w * wt
    ktw = twt + k[:, :, None]
    if mask is not None:
        ktw = ktw + mask[None, :tsz, :tsz, None]

    etw, ektw = torch.exp(tw), torch.exp(ktw)
    num = etw * last_num + (ektw * v[:, :, None]).sum(1)
    den = etw * last_den + ektw.sum(1)

    last_num = torch.cat((last_num, num[..., :-1, :]), dim=-2)
    last_den = torch.cat((last_den, den[..., :-1, :]), dim=-2)

    out = (last_num + torch.exp(u + k) * v) / (last_den + torch.exp(u + k))

    return out, num, den


class Attention(nn.Module):
    init_x: Tensor
    init_num: Tensor
    init_den: Tensor
    mask: Tensor

    def __init__(self, emb_dim: int, max_tsz: int = 1024, lora_rank: int | None = None) -> None:
        super().__init__()

        self.time_decay = nn.Parameter(torch.empty(emb_dim))
        self.time_first = nn.Parameter(torch.empty(emb_dim))

        self.time_mix_k = nn.Parameter(torch.empty(1, 1, emb_dim))
        self.time_mix_v = nn.Parameter(torch.empty(1, 1, emb_dim))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, emb_dim))

        # Disables updating the time parameters if using LoRA.
        if lora_rank is not None:
            self.time_decay.requires_grad_(False)
            self.time_first.requires_grad_(False)
            self.time_mix_k.requires_grad_(False)
            self.time_mix_v.requires_grad_(False)
            self.time_mix_r.requires_grad_(False)

        self.key = maybe_lora(nn.Linear(emb_dim, emb_dim, bias=False), lora_rank)
        self.value = maybe_lora(nn.Linear(emb_dim, emb_dim, bias=False), lora_rank)
        self.receptance = maybe_lora(nn.Linear(emb_dim, emb_dim, bias=False), lora_rank)
        self.output = maybe_lora(nn.Linear(emb_dim, emb_dim, bias=False), lora_rank)

        self.register_buffer("init_x", torch.zeros(1, 1, emb_dim), persistent=False)
        self.register_buffer("init_num", torch.zeros(1, 1, emb_dim), persistent=False)
        self.register_buffer("init_den", torch.zeros(1, 1, emb_dim), persistent=False)
        self.register_buffer("mask", get_mask(max_tsz), persistent=False)

    def time_shift(self, last_x: Tensor, x: Tensor) -> Tensor:
        _, tsz, _ = x.shape
        if tsz > 1:
            last_x = torch.cat((last_x, x[..., :-1, :]), dim=-2)
        return last_x

    def forward(self, x: Tensor, state: AttentionState) -> tuple[Tensor, AttentionState]:
        _, tsz, _ = x.shape

        last_x, last_num, last_den = (self.init_x, self.init_num, self.init_den) if state is None else state
        last_x = self.time_shift(last_x, x)

        k = self.key(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        v = self.value(x * self.time_mix_v + last_x * (1 - self.time_mix_v))
        r = self.receptance(x * self.time_mix_r + last_x * (1 - self.time_mix_r))
        sr = torch.sigmoid(r)

        w, u = self.time_decay, self.time_first
        wkv, num, den = run_wkv(tsz, w, u, k, v, last_num, last_den, self.mask)
        rwkv = wkv * sr

        return self.output(rwkv), (x[..., -1:, :], num[..., -1:, :], den[..., -1:, :])


class FeedForward(nn.Module):
    init_state: Tensor

    def __init__(self, emb_dim: int, ffn_dim: int, lora_rank: int | None = None) -> None:
        super().__init__()

        self.time_mix_k = nn.Parameter(torch.empty(1, 1, emb_dim))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, emb_dim))

        # Disables updating the time parameters if using LoRA.
        if lora_rank is not None:
            self.time_mix_k.requires_grad_(False)
            self.time_mix_r.requires_grad_(False)

        self.key = maybe_lora(nn.Linear(emb_dim, ffn_dim, bias=False), lora_rank)
        self.receptance = maybe_lora(nn.Linear(emb_dim, emb_dim, bias=False), lora_rank)
        self.value = maybe_lora(nn.Linear(ffn_dim, emb_dim, bias=False), lora_rank)

        self.register_buffer("init_state", torch.zeros(1, 1, emb_dim), persistent=False)

    def time_shift(self, last_x: Tensor, x: Tensor) -> Tensor:
        _, tsz, _ = x.shape
        if tsz > 1:
            last_x = torch.cat((last_x, x[..., :-1, :]), dim=-2)
        return last_x

    def forward(self, x: Tensor, state: FeedForwardState | None = None) -> tuple[Tensor, FeedForwardState]:
        last_x = self.time_shift(self.init_state if state is None else state, x)

        k = self.key(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        r = self.receptance(x * self.time_mix_r + last_x * (1 - self.time_mix_r))
        vk = self.value(F.relu(k) ** 2)

        return torch.sigmoid(r) * vk, x[..., -1:, :]


class Block(nn.Module):
    def __init__(self, emb_dim: int, pre_norm: bool, lora_rank: int | None = None) -> None:
        super().__init__()

        self.ln0 = nn.LayerNorm(emb_dim) if pre_norm else None
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

        if lora_rank is not None:
            if self.ln0 is not None:
                self.ln0.requires_grad_(False)
            self.ln1.requires_grad_(False)
            self.ln2.requires_grad_(False)

        self.att = Attention(emb_dim, lora_rank=lora_rank)
        self.ffn = FeedForward(emb_dim, emb_dim * 4, lora_rank=lora_rank)

    def forward(self, x: Tensor, state: State | None = None) -> tuple[Tensor, State]:
        if self.ln0 is not None:
            x = self.ln0(x)
        dx, att_state_out = self.att(self.ln1(x), None if state is None else state[0])
        x = x + dx
        dx, ffn_state_out = self.ffn(self.ln2(x), None if state is None else state[1])
        x = x + dx
        return x, (att_state_out, ffn_state_out)


class Rwkv(nn.Module):
    def __init__(self, emb_dim: int, num_tokens: int, num_layers: int, lora_rank: int | None = None) -> None:
        super().__init__()

        self.emb = maybe_lora(nn.Embedding(num_tokens, emb_dim), lora_rank)
        self.blocks = nn.ModuleList([Block(emb_dim, i == 0, lora_rank=lora_rank) for i in range(num_layers)])
        self.ln_out = nn.LayerNorm(emb_dim)
        self.head = maybe_lora(nn.Linear(emb_dim, num_tokens, bias=False), lora_rank)

        # Disables updating the layer norm parameters if using LoRA.
        if lora_rank is not None:
            self.ln_out.requires_grad_(False)

    def tensor_to(self, x: Tensor) -> Tensor:
        ref_tensor = self.head.weight
        if x.is_floating_point():
            return x.to(ref_tensor)
        return x.to(ref_tensor.device)

    def forward(self, tokens: Tensor, states_in: list[State] | None = None) -> tuple[Tensor, list[State]]:
        x = self.emb(tokens)
        states_out: list[State] = []
        for i, block in enumerate(self.blocks):
            x, state_out = block(x, None if states_in is None else states_in[i])
            states_out.append(state_out)
        x = self.head(self.ln_out(x))
        e_x = torch.exp(x - torch.max(x))
        probs = e_x / e_x.sum()
        return probs, states_out

    def predictor(self) -> "RwkvPredictor":
        return RwkvPredictor(self)


def get_tokenizer() -> Any:
    try:
        from tokenizers import Tokenizer

    except ImportError:
        raise ImportError("Please install tokenizers with: `pip install tokenizers`")

    tokenizer_path = ensure_downloaded(TOKENIZER_URL, "rwkv", "tokenizer.json")
    return Tokenizer.from_file(str(tokenizer_path))


class RwkvPredictor:
    def __init__(self, rwkv_model: Rwkv) -> None:
        """Provides an API for sampling from the RWKV model.

        Args:
            rwkv_model: The RWKV model to use for sampling.
        """
        super().__init__()

        self.tokenizer = get_tokenizer()
        self.model = rwkv_model

    def sample_probs(self, probs: Tensor, temperature: float = 1.0, top_p: float = 0.85) -> Tensor:
        probs = probs ** (1 / temperature)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort.squeeze(-3), num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token[..., None, :, :]).squeeze(-1)
        return next_token

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_len: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.85,
        end_toks: Sequence[int] | None = None,
        end_strs: Sequence[str] | None = None,
    ) -> Iterator[str]:
        tokens = self.tokenizer.encode(prompt).ids

        probs, state = self.model(self.model.tensor_to(torch.tensor([tokens])))
        probs = probs[:, -1:]

        end_toks_set = set() if end_toks is None else set(end_toks)
        end_strs_set = [] if end_strs is None else list(end_strs)

        for i in range(max_len):
            token = self.sample_probs(probs, temperature=temperature, top_p=top_p)
            if token in end_toks_set:
                break
            token_str = self.tokenizer.decode([token.item()])
            yield token_str
            if any(e in token_str for e in end_strs_set):
                break
            if i < max_len - 1:
                probs, state = self.model(self.model.tensor_to(torch.tensor([[token]])), state)


def pretrained_rwkv(key: PretrainedRwkvKey, *, device: BaseDevice | None = None, lora_rank: int | None = None) -> Rwkv:
    device = AutoDevice.detect_device() if device is None else device
    model_args = PRETRAINED_MODEL_SIZES[key]
    ckpt_path = ensure_downloaded(model_args.url, "rwkv", f"{key}.pth", sha256=model_args.sha256)

    with Timer("loading model checkpoint", spinner=True):
        ckpt = torch.load(ckpt_path, map_location="cpu")

    with Timer("building model skeleton", spinner=True), init_empty_weights():
        model = Rwkv(model_args.emb_dim, 50277, model_args.num_layers, lora_rank=lora_rank)

    # Build the transformer and loads the checkpoint.
    with Timer("loading state dict", spinner=True):
        model._apply(meta_to_empty_func(device.get_device(), torch.half))
        model.load_state_dict(ckpt)

    return model


def test_rwkv_adhoc() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=str, choices=get_args(PretrainedRwkvKey))
    parser.add_argument("prompt", type=str, nargs="?")
    parser.add_argument("-t", "--tsz", type=int, default=128)
    parser.add_argument("-m", "--temperature", type=float, default=1.0)
    parser.add_argument("-p", "--top-p", type=float, default=0.85)
    parser.add_argument("-e", "--end-tok", nargs="+", default=[])
    args = parser.parse_args()

    configure_logging()

    model = pretrained_rwkv(args.size)
    predictor = model.predictor()

    def generate_for_prompt(prompt: str) -> None:
        print(prompt, end="")
        for token in predictor.generate(
            prompt,
            max_len=args.tsz,
            temperature=args.temperature,
            top_p=args.top_p,
            end_strs=args.end_tok,
        ):
            print(token, end="", flush=True)
        print()

    if args.prompt:
        generate_for_prompt(args.prompt)

    else:
        prompt = input("Prompt: ")
        while prompt:
            generate_for_prompt(prompt)
            prompt = input("Prompt: ")


if __name__ == "__main__":
    # python -m pretrained.rwkv
    test_rwkv_adhoc()
