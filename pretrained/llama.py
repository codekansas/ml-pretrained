"""Defines a simple API for using Meta's pretrained LLaMa model.

This code is adapted from the original implementation
`here <https://github.com/facebookresearch/llama>`_, adapted to use
the parallelism primitives in this codebase.

.. highlight:: python
.. code-block:: python

    from pretrained.llama import pretrained_llama

    model = pretrained_llama("7B")
    predictor = model.predictor()

    predictor.predict("The quick brown fox jumps over the lazy dog.")

Using the tokenizer requires installing the ``sentencepiece`` library:

.. code-block:: bash

    pip install sentencepiece

The choices for the model key are:

- ``"7B"``
- ``"13B"``
- ``"30B"``
- ``"65B"``
"""

import argparse
import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, cast, get_args

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from ml.core.config import conf_field
from ml.core.env import get_model_dir
from ml.models.lora import maybe_lora
from ml.models.parallel import ColumnParallelLinear, ParallelEmbedding, RowParallelLinear
from ml.utils.device.auto import detect_device
from ml.utils.device.base import base_device
from ml.utils.large_models import init_empty_weights, meta_to_empty_func
from ml.utils.logging import configure_logging
from ml.utils.parallel import parallel_group_info
from ml.utils.timer import Timer
from ml.utils.torch_distributed import MultiprocessConfig, launch_subprocesses
from omegaconf import MISSING
from torch import Tensor, nn

logger = logging.getLogger(__name__)

PretrainedLlamaKey = Literal["7B", "13B", "30B", "65B"]


def cast_pretrained_llama_key(s: str) -> PretrainedLlamaKey:
    if s not in get_args(PretrainedLlamaKey):
        raise KeyError(f"Invalid LLaMa key: {s} Expected one of: {get_args(PretrainedLlamaKey)}")
    return cast(PretrainedLlamaKey, s)


@dataclass
class ModelArgs:
    dim: int = conf_field(MISSING, help="The model inner dimension size")
    n_layers: int = conf_field(MISSING, help="The number of layers in the model")
    n_heads: int = conf_field(MISSING, help="The number of transformer heads")
    mp_size: int = conf_field(MISSING, help="The expected model parallelism size")
    vocab_size: int = conf_field(MISSING, help="The vocabulary size")
    multiple_of: int = conf_field(256, help="Make SwiGLU hidden layer size a multiple of large power of two")
    norm_eps: float = conf_field(1e-4, help="The normalization epsilon value")
    max_seq_len: int = conf_field(2048, help="The maximum sequence length")
    use_checkpointing: bool = conf_field(True, help="Whether to use checkpointing")


PRETRAINED_MODEL_SIZES: dict[PretrainedLlamaKey, ModelArgs] = {
    "7B": ModelArgs(dim=4096, n_heads=32, n_layers=32, mp_size=1),
    "13B": ModelArgs(dim=5120, n_heads=40, n_layers=40, mp_size=2),
    "30B": ModelArgs(dim=6656, n_heads=52, n_layers=60, mp_size=4),
    "65B": ModelArgs(dim=8192, n_heads=64, n_layers=80, mp_size=8),
}


class RMSNorm(nn.Module):
    __constants__ = ["eps"]

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not self.weight.data.is_meta:
            nn.init.ones_(self.weight.data)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor) -> Tensor:
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i in (1, ndim - 1) else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, lora_rank: int | None = None) -> None:
        super().__init__()

        self.n_local_heads = args.n_heads // parallel_group_info().mp.world_size
        self.head_dim = args.dim // args.n_heads

        wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False, gather_output=False)
        wk = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False, gather_output=False)
        wv = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False, gather_output=False)
        wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim, bias=False, input_is_parallel=True)
        self.wq = maybe_lora(wq, lora_rank)
        self.wk = maybe_lora(wk, lora_rank)
        self.wv = maybe_lora(wv, lora_rank)
        self.wo = maybe_lora(wo, lora_rank)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        is_causal: bool,
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Gets the cached keys.
        if cache is None:
            keys, values = xk, xv
        else:
            cache_k, cache_v = cache
            keys, values = torch.cat((cache_k, xk), dim=1), torch.cat((cache_v, xv), dim=1)

        # (B, T, H, D) -> (B, H, T, D)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Attention.
        output = F.scaled_dot_product_attention(xq, keys, values, is_causal=is_causal).transpose(1, 2).flatten(2)

        return self.wo(output), (keys.transpose(1, 2), values.transpose(1, 2))


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        lora_rank: int | None = None,
    ) -> None:
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False)
        w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True)
        w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False)
        self.w1 = maybe_lora(w1, lora_rank)
        self.w2 = maybe_lora(w2, lora_rank)
        self.w3 = maybe_lora(w3, lora_rank)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, lora_rank: int | None = None) -> None:
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, lora_rank=lora_rank)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            lora_rank=lora_rank,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.use_checkpointing = args.use_checkpointing

        if lora_rank is not None:
            self.attention_norm.requires_grad_(False)
            self.ffn_norm.requires_grad_(False)

    def run_attn(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        is_causal: bool,
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        return self.attention.forward(self.attention_norm(x), freqs_cis, is_causal, cache)

    def run_ffn(self, x: Tensor) -> Tensor:
        return self.feed_forward.forward(self.ffn_norm(x))

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        is_causal: bool,
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if self.use_checkpointing:
            h, cache_out = torch.utils.checkpoint.checkpoint(self.run_attn, x, freqs_cis, is_causal, cache)
            x = x + h
            out = x + torch.utils.checkpoint.checkpoint(self.run_ffn, x)
        else:
            h, cache_out = self.run_attn(x, freqs_cis, is_causal, cache)
            x = x + h
            out = x + self.run_ffn(x)
        return out, cache_out


class Tokenizer:
    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path).resolve()
        assert model_path.is_file(), f"Tokenizer model file not found at {model_path}"

        try:
            from sentencepiece import SentencePieceProcessor  # type: ignore[import]

        except ImportError:
            raise ImportError("Please install sentencepiece with: `pip install sentencepiece`")

        self.sp_model = SentencePieceProcessor(model_file=str(model_path))
        logger.info("Loaded sentence piece model from %s", model_path)

        # Gets the sentence statistics.
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
        logger.info("Number of words: %d, BOS ID: %d, EOS ID: %d", self.n_words, self.bos_id, self.eos_id)

    def encode(self, s: str, bos: bool, eos: bool) -> list[int]:
        assert isinstance(s, str)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: list[int]) -> str:
        return self.sp_model.decode(t)


class Llama(nn.Module):
    freqs_cis: Tensor

    def __init__(self, params: ModelArgs, tokenizer: Tokenizer | None = None, lora_rank: int | None = None) -> None:
        super().__init__()

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tokenizer = tokenizer

        tok_embeddings = ParallelEmbedding(params.vocab_size, params.dim)
        self.tok_embeddings = maybe_lora(tok_embeddings, lora_rank)

        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, lora_rank=lora_rank))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        output = ColumnParallelLinear(params.dim, params.vocab_size, bias=False)
        self.output = maybe_lora(output, lora_rank)

        freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        if lora_rank is not None:
            self.tok_embeddings.requires_grad_(False)
            self.norm.requires_grad_(False)

    @torch.no_grad()
    def get_mask(self, seqlen: int, ref_tensor: Tensor) -> Tensor | None:
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=ref_tensor.device)
            mask = torch.triu(mask, diagonal=1).type_as(ref_tensor)
        return mask

    def forward(self, tokens: Tensor) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        _, seqlen = tokens.shape
        x = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seqlen]

        # Runs the transformer.
        caches: list[tuple[Tensor, Tensor]] = []
        for layer in self.layers:
            x, cache = layer(x, freqs_cis, seqlen > 1)
            caches.append(cache)
        x = self.norm(x)
        logits = self.output(x)

        return logits, caches

    @torch.no_grad()
    def infer(
        self,
        tokens: Tensor,
        max_gen_len: int,
        temperature: float,
        top_p: float,
        eos_id: int | None = None,
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Runs model inference for a token sequence.

        Args:
            tokens: The input tokens, with shape (T).
            max_gen_len: The maximum number of tokens to generate.
            temperature: The softmax temperature.
            top_p: The top-p sampling threshold.
            eos_id: The EOS token ID; if not provided, generate as many tokens
                as possible.

        Yields:
            The generated token sequence, with shape (T + N), along with the
            associated logits, with shape (N, V).
        """
        tokens = tokens[None]  # (T) -> (B, T)
        _, seqlen = tokens.shape
        x = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seqlen]

        # Runs the first step of the transformer.
        caches: list[tuple[Tensor, Tensor]] = []

        for layer in self.layers:
            x, cache = layer(x, freqs_cis, seqlen > 1)
            caches.append(cache)
        x = self.norm(x)
        logits = self.output(x[:, -1:])

        for i in range(max_gen_len):
            # Samples the next token from the sequence.
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs.flatten(), top_p)[None]
            else:
                next_token = torch.argmax(logits, dim=-1)
            tokens = torch.cat((tokens, next_token), dim=1)

            if (eos_id is not None and (next_token == eos_id).item()) or i == max_gen_len - 1:
                break

            # Runs the next step of the transformer.
            x = self.tok_embeddings(next_token)
            freqs_cis = self.freqs_cis[seqlen + i : seqlen + i + 1]
            next_caches: list[tuple[Tensor, Tensor]] = []
            for layer, cache in zip(self.layers, caches):
                x, cache = layer(x, freqs_cis, False, cache)
                next_caches.append(cache)
            x = self.norm(x)
            logits = self.output(x[:, -1:, :])
            caches = next_caches

            yield next_token, logits

    def predictor(self) -> "LlamaPredictor":
        return LlamaPredictor(self)


class LlamaPredictor:
    def __init__(self, llama_model: Llama, *, device: base_device | None = None) -> None:
        """Provides an API for sampling from the LLaMa model.

        Args:
            llama_model: The LLaMa model.
            device: The device to use for sampling. If None, the device will be
                automatically detected.

        Raises:
            ValueError: If the tokenizer is not set.
        """
        super().__init__()

        self.device = detect_device() if device is None else device
        self.device.module_to(llama_model)
        tokenizer = llama_model.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer must be set to use predictor")
        self.tokenizer = tokenizer
        self.model = llama_model

    def tokenize(self, prompt: str | None) -> Tensor:
        if prompt is None:
            prompt_tokens = torch.full((1, 1), self.tokenizer.bos_id, dtype=torch.long)
        else:
            prompt_tokens = torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False))
        prompt_tokens = self.device.tensor_to(prompt_tokens)
        return prompt_tokens

    @torch.inference_mode()
    def generate_for_tokens(
        self,
        prompt_tokens: Tensor,
        max_gen_len: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Iterator[str]:
        for pred_token, _ in self.model.infer(
            prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            eos_id=self.tokenizer.eos_id,
        ):
            yield self.tokenizer.decode(pred_token.tolist())[0]

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | None = None,
        max_gen_len: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Iterator[str]:
        prompt_tokens = self.tokenize(prompt)
        yield from self.generate_for_tokens(
            prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

    @torch.no_grad()
    def unit_test_forward_matches_infer(self, prompt: str) -> bool:
        """Ensures that the forward pass matches the inference pass.

        This is a simple unit test which does argmax decoding for the inference
        pass to get a sequence, then passes the sequence to the forward pass.
        The output of the forward pass should match.

        Args:
            prompt: The prompt to use for the unit test.

        Returns:
            Whether the forward pass matches the inference pass.
        """
        test_tokens = torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False))
        test_tokens = self.device.tensor_to(test_tokens)
        seqlen = test_tokens.shape[0]

        inferred_tokens = self.model.infer(
            test_tokens,
            max_gen_len=16,
            temperature=0.0,
            top_p=0.0,
            eos_id=None,
        )

        pred_tokens_list, pred_logits_list = zip(*inferred_tokens)
        pred_tokens, pred_logits = torch.cat(pred_tokens_list, dim=1), torch.cat(pred_logits_list, dim=1)

        fwd_logits, _ = self.model.forward(pred_tokens.unsqueeze(0))

        # Need to check with a high atol because models are in FP16.
        return torch.allclose(pred_logits[:, 1:], fwd_logits[:, seqlen:-1], atol=1e-2)


def sample_top_p(probs: Tensor, p: float) -> Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def get_ckpt_and_tokenizer_path(key: PretrainedLlamaKey) -> tuple[Path, Path]:
    root_dir = get_model_dir() / "llama"
    ckpt_dir = root_dir / key
    tokenizer_path = root_dir / "tokenizer.model"
    return ckpt_dir, tokenizer_path


def empty_llama(key: PretrainedLlamaKey) -> Llama:
    _, tokenizer_path = get_ckpt_and_tokenizer_path(key)
    if not tokenizer_path.exists():
        raise ValueError(f"LLaMa tokenizer not found at {tokenizer_path}; download it first")

    model_args = PRETRAINED_MODEL_SIZES[key]

    with Timer("loading tokenizer", spinner=True):
        tokenizer = Tokenizer(tokenizer_path)
        model_args.vocab_size = tokenizer.n_words

    with Timer("building empty model", spinner=True), init_empty_weights():
        model = Llama(model_args, tokenizer)

    with Timer("moving model to device", spinner=True):
        device = detect_device().get_device()
        model._apply(meta_to_empty_func(device, torch.half))

    def reset_params(module: nn.Module) -> None:
        if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
            module.reset_parameters()

    with Timer("resetting parameters", spinner=True):
        model.apply(reset_params)

    return model


def pretrained_llama(key: PretrainedLlamaKey, *, lora_rank: int | None = None) -> Llama:
    rank, world_size = parallel_group_info().mp.rank, parallel_group_info().mp.world_size

    ckpt_dir, tokenizer_path = get_ckpt_and_tokenizer_path(key)
    if not ckpt_dir.exists():
        raise ValueError(f"LLaMa model {key} not found at {ckpt_dir}; download it first")
    if not tokenizer_path.exists():
        raise ValueError(f"LLaMa tokenizer not found at {tokenizer_path}; download it first")

    # Loads the checkpoint for the current rank.
    with Timer("loading checkpoint", spinner=True):
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        if world_size != (num_ckpts := len(checkpoints)):
            raise ValueError(f"Loading a checkpoint for {num_ckpts=} but {world_size=}")
        ckpt_path = checkpoints[rank]
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Loads the checkpoint parameters from the JSON file.
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args = ModelArgs(**params)

    # Builds the tokenizer and updates the vocab size.
    with Timer("loading tokenizer", spinner=True):
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words

    # Builds the model with empty weights.
    with Timer("building model skeleton", spinner=True), init_empty_weights():
        model = Llama(model_args, tokenizer, lora_rank=lora_rank)

    # Logs model summary.
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model %s has %s parameters", key, f"{total_params:,}")

    # Build the transformer and loads the checkpoint.
    with Timer("loading state dict", spinner=True):
        model._apply(meta_to_empty_func(torch.device("cuda"), torch.half))
        model.load_state_dict(checkpoint)

    return model


def test_worker(
    key: PretrainedLlamaKey,
    prompt: str | None,
    max_gen_len: int,
    temperature: float,
    top_p: float,
    pretrained: bool,
) -> None:
    model = pretrained_llama(key) if pretrained else empty_llama(key)
    predictor = model.predictor()

    # Setting the seed across all processes to make sure that the weights
    # initialize to the same values (needed to make the test pass).
    torch.manual_seed(1337)

    def generate_for_prompt(prompt: str) -> None:
        print(prompt, end="")
        for token in predictor.generate(
            prompt=prompt,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        ):
            print(token, end="", flush=True)
        print()

    if prompt:
        generate_for_prompt(prompt)

    else:
        prompt = input("Prompt: ")
        while prompt:
            generate_for_prompt(prompt)
            prompt = input("Prompt: ")


def setup() -> None:
    # Hides some nuisance logs.
    logging.getLogger("torch.distributed").setLevel(logging.ERROR)
    logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.ERROR)
    logging.getLogger("ml.utils.torch_distributed").setLevel(logging.ERROR)

    # Setting the seed across all processes to make sure that the weights
    # initialize to the same values (needed to make the test pass).
    torch.manual_seed(1337)


def test_pretrained_model() -> None:
    parser = argparse.ArgumentParser(description="Tests a pretrained LLaMA model")
    parser.add_argument("key", type=str, choices=get_args(PretrainedLlamaKey))
    parser.add_argument("prompt", type=str)
    parser.add_argument("-m", "--max-gen-len", type=int, default=256)
    parser.add_argument("-t", "--temperature", type=float, default=0.8)
    parser.add_argument("-p", "--top-p", type=float, default=0.95)
    parser.add_argument("-e", "--empty", default=False, action="store_true")
    args = parser.parse_args()

    configure_logging()

    key = args.key
    all_args = args.prompt, args.max_gen_len, args.temperature, args.top_p, not args.empty
    world_size = PRETRAINED_MODEL_SIZES[key].mp_size

    launch_subprocesses(
        functools.partial(test_worker, key, *all_args),
        MultiprocessConfig(
            world_size=world_size,
            model_parallelism=world_size,
        ),
        setup=setup,
    )


if __name__ == "__main__":
    # python -m pretrained.llama 7B 'The meaning of life is'
    test_pretrained_model()
