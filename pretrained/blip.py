"""Defines a simple API for using the BLIP model.

This code provides a simple API for interacting with BLIP, referencing the
original repository `here <https://github.com/salesforce/BLIP>`_.

.. highlight:: python
.. code-block:: python

    from pretrained.blip import pretrained_blip

    model = pretrained_blip("ViT-B")
    predictor = model.predictor()

    image = PIL.Image.open(image_path)
    embedding = predictor.predict(image)

The choices for the model key are:

- ``ViT-B``
- ``ViT-B-CapFilt``
- ``ViT-L``
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Literal, Sequence, Union, cast, get_args

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as V
from ml.core.config import conf_field
from ml.models.activations import ActivationType, get_activation
from ml.models.init import init_
from ml.models.norms import NormType, get_norm_linear
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.device.auto import detect_device
from ml.utils.device.base import base_device
from ml.utils.large_models import init_empty_weights, meta_to_empty_func
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer
from omegaconf import MISSING
from torch import Tensor, nn

logger = logging.getLogger(__name__)

RawImage = Union[PIL.Image.Image, Tensor, np.ndarray]

PretrainedBlipKey = Literal[
    # ViT-B models
    "ViT-B",
    "ViT-B-Coco",
    "ViT-B-Flickr30k",
    "ViT-B-VQA",
    "ViT-B-NLVR2",
    # ViT-B-CapFilt models
    "ViT-B-CapFilt",
    "ViT-B-CapFilt-Coco",
    "ViT-B-CapFilt-VQA",
    # ViT-L models
    "ViT-L",
    "ViT-L-Coco-Retrieval",
    "ViT-L-Flickr30k",
    "ViT-L-Coco-Captioning",
]


def cast_pretrained_blip_key(s: str) -> PretrainedBlipKey:
    if s not in get_args(PretrainedBlipKey):
        raise KeyError(f"Invalid BLIP size: {s} Expected one of: {get_args(PretrainedBlipKey)}")
    return cast(PretrainedBlipKey, s)


@dataclass
class ViTParams:
    img_size: int | tuple[int, int] = conf_field(224, help="Total image size")
    patch_size: int = conf_field(16, help="Size of image patches")
    in_chans: int = conf_field(3, help="Number of input channels")
    embed_dim: int = conf_field(MISSING, help="Number of embedding dimensions")
    depth: int = conf_field(MISSING, help="Number of transformer layers")
    num_heads: int = conf_field(MISSING, help="Number of attention heads")
    mlp_ratio: float = conf_field(4.0, help="Ratio of MLP hidden dim to embedding dim")
    qkv_bias: bool = conf_field(True, help="If True, use bias for qkv projection")
    drop_rate: float = conf_field(0.0, help="Dropout rate")
    attn_drop_rate: float = conf_field(0.0, help="Attention dropout rate")
    norm_type: NormType = conf_field("layer_affine", help="Normalization type")


@dataclass
class TextParams:
    num_tokens: int = conf_field(MISSING, help="Number of tokens")


@dataclass
class ModelParams:
    url: str = conf_field(MISSING, help="URL to pre-trained weights")
    vit: ViTParams = conf_field(MISSING, help="ViT parameters")


VIT_BASE = ViTParams(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
)

VIT_BASE_CAPFILT = ViTParams(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
)

VIT_LARGE = ViTParams(
    img_size=224,
    patch_size=16,
    embed_dim=1024,
    depth=16,
    num_heads=24,
)


PRETRAINED_MODEL_SIZES: dict[PretrainedBlipKey, ModelParams] = {
    # ViT-B models
    "ViT-B": ModelParams(
        url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth",
        vit=VIT_BASE,
    ),
    "ViT-B-Coco": ModelParams(
        url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth",
        vit=VIT_BASE,
    ),
    "ViT-B-Flickr30k": ModelParams(
        url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth",
        vit=VIT_BASE,
    ),
    "ViT-B-VQA": ModelParams(
        url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth",
        vit=VIT_BASE,
    ),
    "ViT-B-NLVR2": ModelParams(
        url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_nlvr.pth",
        vit=VIT_BASE,
    ),
    # ViT-B-CapFilt models
    "ViT-B-CapFilt": ModelParams(
        url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth",
        vit=VIT_BASE_CAPFILT,
    ),
    "ViT-B-CapFilt-Coco": ModelParams(
        url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth",
        vit=VIT_BASE_CAPFILT,
    ),
    "ViT-B-CapFilt-VQA": ModelParams(
        url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth",
        vit=VIT_BASE_CAPFILT,
    ),
    # ViT-L models
    "ViT-L": ModelParams(
        url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth",
        vit=VIT_LARGE,
    ),
    "ViT-L-Coco-Retrieval": ModelParams(
        url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth",
        vit=VIT_LARGE,
    ),
    "ViT-L-Flickr30k": ModelParams(
        url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_flickr.pth",
        vit=VIT_LARGE,
    ),
    "ViT-L-Coco-Captioning": ModelParams(
        url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth",
        vit=VIT_LARGE,
    ),
}


class PatchEmbed(nn.Module):
    """Gets patch embeddings for an input image.

    Parameters:
        img_size: The size of the input image.
        patch_size: The size of the patches to extract.
        flatten: Whether to flatten the output.
        in_chans: The number of input channels.
        embed_dim: The embedding dimension.
        norm_type: The type of normalization to use.
        bias: Whether to use a bias term.

    Inputs:
        x: The input image, of shape ``(B, C, H, W)``.

    Outputs:
        The patch embeddings, of shape ``(B, num_patches, embed_dim)``
        if ``flatten=True``, or ``(B, embed_dim, H // patch_size, W // patch_size)``
        otherwise.
    """

    __constants__ = ["patch_size", "img_size", "num_patches", "flatten", "in_chans"]

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        flatten: bool = True,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_type: NormType = "no_norm",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        assert self.img_size[0] % self.patch_size[0] == 0 and self.img_size[1] % self.patch_size[1] == 0
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        self.flatten = flatten
        self.in_chans = in_chans
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = get_norm_linear(norm_type, dim=embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        _, _, hei, wid = x.shape
        assert hei == self.img_size[0] and wid == self.img_size[1]

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, H * W, C)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_type: ActivationType = "gelu",
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = get_activation(act_type)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def forward(self, x: Tensor) -> Tensor:
        bsz, tsz, chans = x.shape
        qkv = self.qkv(x).reshape(bsz, tsz, 3, self.num_heads, chans // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(1, 2).reshape(bsz, tsz, chans)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_type: ActivationType = "gelu",
        norm_type: NormType = "layer_affine",
    ) -> None:
        super().__init__()
        self.norm1 = get_norm_linear(norm_type, dim=dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = get_norm_linear(norm_type, dim=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_type=act_type, drop=drop)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Defines a vision transformer implementation.

    Implementation of `this paper <https://arxiv.org/abs/2010.11929>`_.
    """

    def __init__(self, model_args: ViTParams) -> None:
        super().__init__()

        self.num_features = self.embed_dim = model_args.embed_dim
        self.patch_embed = PatchEmbed(
            img_size=model_args.img_size,
            patch_size=model_args.patch_size,
            in_chans=model_args.in_chans,
            embed_dim=model_args.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_args.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, model_args.embed_dim))
        self.pos_drop = nn.Dropout(p=model_args.drop_rate)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=model_args.embed_dim,
                    num_heads=model_args.num_heads,
                    mlp_ratio=model_args.mlp_ratio,
                    qkv_bias=model_args.qkv_bias,
                    drop=model_args.drop_rate,
                    attn_drop=model_args.attn_drop_rate,
                    norm_type=model_args.norm_type,
                )
                for i in range(model_args.depth)
            ]
        )
        self.norm = get_norm_linear(model_args.norm_type, dim=model_args.embed_dim)

        init_(self.pos_embed, None, "trunc_normal", std=0.02)
        init_(self.cls_token, None, "trunc_normal", std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            init_(m.weight, m.bias, "trunc_normal", std=0.02)
        elif isinstance(m, nn.LayerNorm):
            init_(m.weight, m.bias, "ones")

    def forward(self, x: Tensor) -> Tensor:
        bsz = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x


class Blip(nn.Module):
    def __init__(self, model_args: ModelParams) -> None:
        super().__init__()

        self.visual_encoder = VisionTransformer(model_args.vit)

    def forward(self, img: Tensor) -> Tensor:
        return self.visual_encoder(img)

    def predictor(self, device: base_device | None = None) -> "BlipPredictor":
        return BlipPredictor(self, device=device)


class BlipPredictor:
    def __init__(self, blip_model: Blip, *, device: base_device | None = None) -> None:
        """Provides an API for sampling from the BLIP model.

        Args:
            blip_model: The BLIP model to use for sampling.
            device: The device to use for sampling. If None, the device will be
                automatically detected.
        """
        super().__init__()

        self.device = detect_device() if device is None else device
        self.device.module_to(blip_model)
        self.model = blip_model

    @torch.no_grad()
    def get_input_image(self, image: RawImage) -> Tensor:
        if isinstance(image, PIL.Image.Image):
            image = image.convert("RGB")
        elif isinstance(image, (np.ndarray, Tensor)):
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            assert image.dim() == 3 and image.shape[0] == self.model.visual_encoder.patch_embed.in_chans
        image = V.resize(image, self.model.visual_encoder.patch_embed.img_size, PIL.Image.BICUBIC)
        x = self.device.tensor_to(image if isinstance(image, Tensor) else V.pil_to_tensor(image))
        x = V.convert_image_dtype(x, self.device.get_floating_point_type())
        x = V.normalize(x, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        return x

    @torch.no_grad()
    def predict(self, image: RawImage | Sequence[RawImage]) -> Tensor:
        """Gets the embedding for a given image.

        Args:
            image: The image to get an embedding for.

        Returns:
            The embedding tensor, with shape ``(embed_dim)``,
            or ``(bsz, embed_dim)`` if ``image`` is a sequence.
        """
        if isinstance(image, Sequence):
            x = torch.stack([self.get_input_image(img) for img in image])
            x = self.model(x)[:, 0]
        else:
            x = self.get_input_image(image)
            x = self.model(x.unsqueeze(0)).squeeze(0)[0]
        return x


def pretrained_blip(key: PretrainedBlipKey, *, device: base_device | None = None) -> Blip:
    device = detect_device() if device is None else device
    model_args = PRETRAINED_MODEL_SIZES[key]

    with Timer("downloading checkpoint"):
        ckpt_path = ensure_downloaded(model_args.url, "blip", f"{key}.pth")

    with Timer("loading model checkpoint", spinner=True):
        ckpt = torch.load(ckpt_path, map_location="cpu")["model"]
        # Filters jut the visual encoder weights.
        ckpt = {k: v for k, v in ckpt.items() if k.startswith("visual_encoder.")}

    with Timer("building model skeleton", spinner=True), init_empty_weights():
        model = Blip(model_args)

    # Logs model summary.
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model %s has %s parameters", key, f"{total_params:,}")

    # Build the transformer and loads the checkpoint.
    with Timer("loading state dict", spinner=True):
        model._apply(meta_to_empty_func(device.get_device(), torch.half))
        model.load_state_dict(ckpt, strict=True)

    return model


def test_blip_adhoc() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("key", type=str, choices=get_args(PretrainedBlipKey))
    args = parser.parse_args()

    configure_logging()

    # Gets an image of a peach from Wikipedia.
    peach_url = "https://upload.wikimedia.org/wikipedia/commons/9/9e/Autumn_Red_peaches.jpg"
    img_path = ensure_downloaded(peach_url, "peach.jpg", is_tmp=True)
    image = PIL.Image.open(img_path)

    model = pretrained_blip(args.key)
    predictor = model.predictor()

    # Outputs the embedding for a given image.
    embedding = predictor.predict(image)
    logger.info("Embedding shape: %s", embedding.shape)


if __name__ == "__main__":
    # python -m pretrained.blip
    test_blip_adhoc()
