"""Defines a simple API for using Meta's pretrained Segment Anything model.

.. highlight:: python
.. code-block:: python

    from pretrained.sam import pretrained_sam

    model = pretrained_sam("ViT-B")
    predictor = model.predictor()

    image = PIL.Image.open(img_path)
    predictor.set_image(np.array(image))

    predictions, _, _ = predictor.predict()
    single_mask = predictions[0]  # Same shape as the original image.

Alternatively, you can run the script directly on an image:

.. code-block:: bash

    python -m pretrained.sam ViT-B /path/to/image.jpg

The choices for the model key are:

- ``ViT-H``: ViT with 32 layers and 16 attention heads.
- ``ViT-L``: ViT with 24 layers and 16 attention heads.
- ``ViT-B``: ViT with 12 layers and 12 attention heads.
"""

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Type, cast, get_args

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from ml.utils.checkpoint import ensure_downloaded
from ml.utils.device.auto import detect_device
from ml.utils.device.base import base_device
from ml.utils.logging import configure_logging
from ml.utils.timer import Timer
from torch import Tensor, nn
from torchvision.transforms.functional import resize, to_pil_image

PretrainedSamSize = Literal["ViT-H", "ViT-L", "ViT-B"]

ImageFormat = Literal["RGB", "BGR"]

DEFAULT_PIXEL_MEAN = (123.675, 116.28, 103.53)
DEFAULT_PIXEL_STD = (58.395, 57.12, 57.375)


def cast_pretrained_sam_size(s: str) -> PretrainedSamSize:
    if s not in get_args(PretrainedSamSize):
        raise KeyError(f"Invalid SAM size: {s} Expected one of: {get_args(PretrainedSamSize)}")
    return cast(PretrainedSamSize, s)


@dataclass
class PretrainedModelConfig:
    url: str
    encoder_embed_dim: int
    encoder_depth: int
    encoder_num_heads: int
    encoder_global_attn_indices: tuple[int, int, int, int]


PRETRAINED_MODELS: dict[PretrainedSamSize, PretrainedModelConfig] = {
    "ViT-H": PretrainedModelConfig(
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indices=(7, 15, 23, 31),
    ),
    "ViT-L": PretrainedModelConfig(
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indices=(5, 11, 17, 23),
    ),
    "ViT-B": PretrainedModelConfig(
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indices=(2, 5, 8, 11),
    ),
}


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()

        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerNormHigherEps(nn.LayerNorm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["eps"] = kwargs.pop("eps", 1e-6)
        super().__init__(*args, **kwargs)


class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 0,
        global_attn_indexes: tuple[int, ...] = (),
    ) -> None:
        """Image encoder based on Vision Transformer.

        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of input image channels.
            embed_dim: Patch embedding dimension.
            depth: Depth of ViT.
            num_heads: Number of attention heads in each ViT block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            out_chans: Number of output channels.
            qkv_bias: If True, add a learnable bias to query, key, value.
            norm_layer: Normalization layer.
            act_layer: Activation layer.
            use_abs_pos: If True, use absolute positional embeddings.
            use_rel_pos: If True, add relative positional embeddings to the attention map.
            window_size: Window size for window attention blocks.
            global_attn_indexes: Indexes for blocks using global attention.
        """
        super().__init__()

        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: nn.Parameter | None = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        window_size: int = 0,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """Transformer blocks, which support window attention and residual propagation.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads in each ViT block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            norm_layer: Normalization layer.
            act_layer: Activation layer.
            use_rel_pos: If True, add relative positional embeddings to the
                attention map.
            window_size: Window size for window attention blocks. If it equals
                0, then use global attention.
            input_size: Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)

        # Window partition
        if self.window_size > 0:
            hei, wid = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (hei, wid))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """Multi-head attention block with relative position embeddings.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            qkv_bias:  If True, add a learnable bias to query, key, value.
            use_rel_pos: If True, add relative positional embeddings to the
                attention map.
            input_size: Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."

            # Initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: Tensor) -> Tensor:
        bsz, hei, wid, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, hei * wid, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, bsz * self.num_heads, hei * wid, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (hei, wid), (hei, wid))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(bsz, self.num_heads, hei, wid, -1).permute(0, 2, 3, 1, 4).reshape(bsz, hei, wid, -1)
        x = self.proj(x)

        return x


def window_partition(x: Tensor, window_size: int) -> tuple[Tensor, tuple[int, int]]:
    """Partition into non-overlapping windows with padding if needed.

    Args:
        x: Input tokens with shape (B, H, W, C).
        window_size: Window size.

    Returns:
        Windows after partition with shape (B * n_win, win_size, win_size, C),
        and the shape.
    """
    bsz, hei, wid, chans = x.shape

    pad_h = (window_size - hei % window_size) % window_size
    pad_w = (window_size - wid % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hei_p, wid_p = hei + pad_h, wid + pad_w

    x = x.view(bsz, hei_p // window_size, window_size, wid_p // window_size, window_size, chans)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, chans)
    return windows, (hei_p, wid_p)


def window_unpartition(windows: Tensor, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]) -> Tensor:
    """Window unpartition into original sequences and removing padding.

    Args:
        windows: Input tokens with (B * n_win, win_size, win_size, C).
        window_size: Window size.
        pad_hw: Padded height and width (Hp, Wp).
        hw: Original height and width (H, W) before padding.

    Returns:
        The unpartitioned sequences with shape (B, H, W, C).
    """
    hei_p, wid_p = pad_hw
    hei, wid = hw
    bsz = windows.shape[0] // (hei_p * wid_p // window_size // window_size)
    x = windows.view(bsz, hei_p // window_size, wid_p // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(bsz, hei_p, wid_p, -1)

    if hei_p > hei or wid_p > wid:
        x = x[:, :hei, :wid, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: Tensor) -> Tensor:
    """Get relative positional embeddings.

    Args:
        q_size: Size of query q.
        k_size: Size of key k.
        rel_pos: Relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)

    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: Tensor,
    q: Tensor,
    rel_pos_h: Tensor,
    rel_pos_w: Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> Tensor:
    """Calculate decomposed Relative Positional Embeddings.

    https://github.com/facebookresearch/mvit/mvit/models/attention.py

    Args:
        attn: Attention map.
        q: Query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h: Relative position embeddings (Lh, C) for height axis.
        rel_pos_w: Relative position embeddings (Lw, C) for width axis.
        q_size: Spatial sequence size of query q with (q_h, q_w).
        k_size: Spatial sequence size of key k with (k_h, k_w).

    Returns:
        Attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    rel_h = get_rel_pos(q_h, k_h, rel_pos_h)
    rel_w = get_rel_pos(q_w, k_w, rel_pos_w)

    bsz, _, dim = q.shape
    r_q = q.reshape(bsz, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, rel_h)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, rel_w)

    attn = attn.view(bsz, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    attn = attn.view(bsz, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """Image to Patch Embedding.

        Args:
            kernel_size: Kernel size of the projection layer.
            stride: Stride of the projection layer.
            padding: Padding size of the projection layer.
            in_chans: Number of input image channels.
            embed_dim: Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        return x


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """Predicts masks given an image and prompt embeddings.

        Args:
            transformer_dim: The channel dimension of the transformer
            transformer: The transformer used to predict masks
            num_multimask_outputs: The number of masks to predict when
                disambiguating masks
            activation: The type of activation to use when upscaling masks
            iou_head_depth: The depth of the MLP used to predict mask quality
            iou_head_hidden_dim: The hidden dimension of the MLP used to
                predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for i in range(self.num_mask_tokens)]
        )

        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        multimask_output: bool,
    ) -> tuple[Tensor, Tensor]:
        """Predict masks given image and prompt embeddings.

        Args:
            image_embeddings: The embeddings from the image encoder
            image_pe: Positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings: The embeddings of the points and boxes
            dense_prompt_embeddings: The embeddings of the mask inputs
            multimask_output: Whether to return multiple masks or a single
                mask.

        Returns:
            The batched predicted masks and batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: list[Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MLP(nn.Module):
    __constants__ = ["num_layers", "sigmoid_output"]

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class PromptEncoder(nn.Module):
    __constants__ = ["embed_dim", "input_image_size", "input_image_size"]

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: tuple[int, int],
        input_image_size: tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """Encodes prompts for input to SAM's mask decoder.

        Args:
            embed_dim: The prompts' embedding dimension
            image_embedding_size: The spatial size of the image embedding,
                as (H, W).
            input_image_size: The padded size of the image as input to the
                image encoder, as (H, W).
            mask_in_chans: The number of hidden channels used for encoding
                input masks.
            activation: The activation to use when encoding input masks.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> Tensor:
        """Returns the positional encoding used to encode point prompts.

        The embedding is applied to a dense set of points the shape of the
        image encoding.

        Returns:
            Positional encoding with shape (1, emb_dim, emb_h, emb_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: Tensor,
        labels: Tensor,
        pad: bool,
    ) -> Tensor:
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: Tensor) -> Tensor:
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: Tensor) -> Tensor:
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: tuple[Tensor, Tensor] | None,
        boxes: Tensor | None,
        masks: Tensor | None,
    ) -> int:
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: tuple[Tensor, Tensor] | None,
        boxes: Tensor | None,
        masks: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Embeds different types of prompts.

        This function returns both sparse and dense embeddings.

        Args:
            points: Point coordinates and labels to embed.
            boxes: Boxes to embed
            masks: Masks to embed

        Returns:
            Sparse embeddings for the points and boxes, with shape
            (B, N, embed_dim), where N is determined by the number of
            input points and boxes, and the dense embeddings for the masks,
            with shape (B, emb_dim, emb_h, emb_w)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    positional_encoding_gaussian_matrix: Tensor

    def __init__(self, num_pos_feats: int = 64, scale: float | None = None) -> None:
        super().__init__()

        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        # Assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)  # (d_1, ..., d_n, C)

    def forward(self, size: tuple[int, int]) -> Tensor:
        h, w = size
        grid = self.positional_encoding_gaussian_matrix.new_ones((h, w))
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # (C, H, W)

    def forward_with_coords(self, coords_input: Tensor, image_size: tuple[int, int]) -> Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # (B, N, C)


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """Transformer which does cross-attention in two directions.

        This transformer decoder attends to an input image using queries whose
        positional embedding is supplied.

        Args:
            depth: Number of layers in the transformer
            embedding_dim: The channel dimension for the input embeddings
            num_heads: The number of heads for multihead attention. Must
                divide embedding_dim
            mlp_dim: The channel dimension internal to the MLP block
            activation: The activation to use in the MLP block
            attention_downsample_rate: The downsample rate for the attention
                blocks. The attention blocks will downsample the input image
                by this factor.
        """
        super().__init__()

        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = TwoWayAttentionFunction(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Runs the transformer forward pass.

        Args:
            image_embedding: Image to attend to. Should be shape
                (B, embedding_dim, h, w).
            image_pe: The positional encoding to add to the image. Must
                have the same shape as image_embedding.
            point_embedding: The embedding to add to the query points. Must
                have shape (B, N_points, embedding_dim).

        Returns:
            The processed point and image embeddings.
        """
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """Defines a mutual cross attention block.

        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to
        sparse inputs.

        Args:
            embedding_dim: The channel dimension of the embeddings
            num_heads: The number of heads in the attention layers
            mlp_dim: The hidden dimension of the mlp block
            activation: The activation of the mlp block
            attention_downsample_rate: The downsample rate for the attention
                blocks. The attention blocks will downsample the input image
                by this factor.
            skip_first_layer_pe: Skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = TwoWayAttentionFunction(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = TwoWayAttentionFunction(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = TwoWayAttentionFunction(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayAttentionFunction(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # (B, N_heads, N_tokens, C_per_head)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # (B, N_tokens, C)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections.
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads.
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Applies attention.
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # (B, N_heads, N_tokens, N_tokens)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Gets the output.
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    pixel_mean: Tensor
    pixel_std: Tensor

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: tuple[float, float, float] = DEFAULT_PIXEL_MEAN,
        pixel_std: tuple[float, float, float] = DEFAULT_PIXEL_STD,
    ) -> None:
        """SAM predicts object masks from an image and input prompts.

        Args:
            image_encoder: The backbone used to encode the image.
            prompt_encoder: Encodes various types of input prompts.
            mask_decoder: Predicts masks from the image embeddings
                and encoded prompts.
            pixel_mean: Mean values for normalizing pixels in the input image.
            pixel_std: Std values for normalizing pixels in the input image.
        """
        super().__init__()

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: list[dict[str, Any]],
        multimask_output: bool,
    ) -> list[dict[str, Tensor]]:
        """Predicts masks end-to-end from provided images and prompts.

        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Args:
            batched_input: A list over input images, each a dictionary with
                the following keys. A prompt key can be excluded if it is not
                present. The 'image' key expects a tensor with shape (3, H, W),
                already transformed for input to the model. The 'original_size'
                key expects a tuple of (H, W), the original size of the image
                before transformation. The 'point_coords' key expects a tensor
                with shape (N, 2), the coordinates of N point prompts in the
                image. The 'point_labels' key expects a tensor with shape (N,),
                the labels of the N point prompts. The 'boxes' key expects a
                tensor with shape (4,), the coordinates of a box prompt in
                the image. The 'mask_inputs' key expects a tensor with shape
                (1, H, W), the mask input to the model.
            multimask_output: Whether the model should predict multiple
                disambiguating masks, or return a single mask.

        Returns:
            A list over input images, where each element is as dictionary with
            the following keys. The 'masks' key is the batched binary mask
            predictions with shape (B, C, H, W), where B is the number of input
            prompts, C is determined by multimask_output, and (H, W) is the
            original size of the image. The 'iou_predictions' key is the
            model's predictions of mask quality, with shape (B, C). The
            'low_res_logits' key is the low resolution logits with shape
            (B, C, 256, 256). This can be passed as mask input to subsequent
            iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: Tensor,
        input_size: tuple[int, ...],
        original_size: tuple[int, ...],
    ) -> Tensor:
        """Removes padding and upscale masks to the original image size.

        Args:
            masks: Batched masks from the mask_decoder, with shape (B, C, H, W)
            input_size: The size of the image input to the model, in (H, W)
                format. Used to remove padding.
            original_size: The original size of the image before resizing for
                input to the model, in (H, W) format.

        Returns:
            Batched masks with shape (B, C, H, W), where (H, W) matches
            the original size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: Tensor) -> Tensor:
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))

        return x

    def predictor(self) -> "SamPredictor":
        return SamPredictor(self)


class ResizeLongestSide:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: tuple[int, ...]) -> np.ndarray:
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        coords = copy.deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: tuple[int, ...]) -> np.ndarray:
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: Tensor) -> Tensor:
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(image, target_size, mode="bilinear", align_corners=False, antialias=True)

    def apply_coords_torch(self, coords: Tensor, original_size: tuple[int, ...]) -> Tensor:
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        coords = copy.deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(self, boxes: Tensor, original_size: tuple[int, ...]) -> Tensor:
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class SamPredictor:
    def __init__(self, sam_model: Sam, *, device: base_device | None = None) -> None:
        """Provides an API to do repeated mask predictions on an image.

        This predictor uses SAM to calculate the image embedding for an image,
        and then allow repeated, efficient mask prediction given prompts.

        Args:
            sam_model: The model to use for mask prediction.
            device: The device to use for prediction. If None, will use the
                device returned by detect_device().
        """
        super().__init__()

        self.device = detect_device() if device is None else device
        self.model = sam_model.eval()
        self.device.module_to(self.model)
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def set_image(self, image: np.ndarray, image_format: ImageFormat = "RGB") -> None:
        """Sets a given image for mask prediction.

        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Args:
            image: The image for calculating masks. Expects an image in HWC
                uint8 format, with pixel values in [0, 255].
            image_format: The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in get_args(ImageFormat)

        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = self.device.tensor_to(torch.as_tensor(input_image))
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: Tensor,
        original_image_size: tuple[int, ...],
    ) -> None:
        """Sets a given image for mask prediction.

        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Args:
            transformed_image: The input image, with shape (1, 3, H, W), which
                has been transformed with ResizeLongestSide.
            original_image_size: The size of the image before transformation,
                in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
        self,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks for the given input prompts for the current image.

        Args:
            point_coords: A (N, 2) array of point prompts to the model. Each
                point is in (X, Y) in pixels.
            point_labels: A length N array of labels for the point prompts. 1
                indicates a foreground point and 0 indicates a background point.
            box: A length 4 array given a box prompt to the model, in
                XYXY format.
            mask_input: A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form
                (1, H, W), where for SAM, H=W=256.
            multimask_output: If true, the model will return three masks.
                For ambiguous input prompts (such as a single click), this will
                often produce better masks than a single prediction. If only a
                single mask is needed, the model's predicted quality score can
                be used to select the best mask. For non-ambiguous prompts, such
                as multiple input prompts, multimask_output=False can give
                better results.
            return_logits: If true, returns un-thresholded masks logits instead
                of a binary mask.

        Returns:
            The output masks with shape (C, H, W), where C is the number of
            masks, and (H, W) is the original image size; an array of length C
            containing the model's predictions for the quality of each mask;
            and an array of shape (C, H, W), where C is the number of masks and
            H=W=256. These low resolution logits can be passed to a subsequent
            iteration as mask input.

        Raises:
            RuntimeError: If an image has not been set yet.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None, "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = self.device.tensor_to(torch.as_tensor(point_coords, dtype=torch.float))
            labels_torch = self.device.tensor_to(torch.as_tensor(point_labels, dtype=torch.int))
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = self.device.tensor_to(torch.as_tensor(box, dtype=torch.float))
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = self.device.tensor_to(torch.as_tensor(mask_input, dtype=torch.float))
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Tensor | None,
        point_labels: Tensor | None,
        boxes: Tensor | None = None,
        mask_input: Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predicts masks for the given input prompts.

        Predicts masks for the given input prompts, using the currently set
        image. Input prompts are batched Tensors and are expected to
        already be transformed to the input frame using ResizeLongestSide.

        Args:
            point_coords: A (B, N, 2) array of point prompts to the
                model. Each point is in (X, Y) in pixels.
            point_labels: A (B, N) array of labels for the point prompts. 1
                indicates a foreground point and 0 indicates a background
                point.
            boxes: A (B, 4) array given a box prompt to the model, in
                XYXY format.
            mask_input: A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form
                (B, 1, H, W), where for SAM, H=W=256. Masks returned by a
                previous iteration of the predict method do not need further
                transformation.
            multimask_output: If true, the model will return three masks.
                For ambiguous input prompts (such as a single click), this will
                often produce better masks than a single prediction. If only a
                single mask is needed, the model's predicted quality score can
                be used to select the best mask. For non-ambiguous prompts,
                such as multiple input prompts, multimask_output=False can give
                better results.
            return_logits: If true, returns un-thresholded masks logits
                instead of a binary mask.

        Returns:
            The output masks with shape (C, H, W), where C is the number of
            masks, and (H, W) is the original image size; an array of length C
            containing the model's predictions for the quality of each mask;
            and an array of shape (C, H, W), where C is the number of masks and
            H=W=256. These low resolution logits can be passed to a subsequent
            iteration as mask input.

        Raises:
            RuntimeError: If an image has not been set yet.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> Tensor:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) to generate an embedding.")
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None


def get_pretrained_path(key: PretrainedSamSize) -> Path:
    if key not in PRETRAINED_MODELS:
        raise KeyError(f"Invalid CLIP model key {key}; choices are {list(PRETRAINED_MODELS.keys())}")
    model_url = PRETRAINED_MODELS[key].url
    with Timer("downloading checkpoint"):
        filepath = ensure_downloaded(model_url, "sam", f"{key}_ckpt.pt")
    return filepath


def pretrained_sam(key: PretrainedSamSize, *, skip_weights: bool = False) -> Sam:
    config = PRETRAINED_MODELS[key]

    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    model = Sam(
        image_encoder=ImageEncoderViT(
            depth=config.encoder_depth,
            embed_dim=config.encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=LayerNormHigherEps,
            num_heads=config.encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=config.encoder_global_attn_indices,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=DEFAULT_PIXEL_MEAN,
        pixel_std=DEFAULT_PIXEL_STD,
    )

    if not skip_weights:
        with open(get_pretrained_path(key), "rb") as f:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict)

    return model


def test_pretrained_model() -> None:
    parser = argparse.ArgumentParser(description="Tests a pretrained SAM model")
    parser.add_argument("key", type=str, choices=get_args(PretrainedSamSize))
    parser.add_argument("image_path", type=str, nargs="?", default=None)
    parser.add_argument("-o", "--output-path", type=str, default=None)
    args = parser.parse_args()

    configure_logging()

    # Gets an image of a peach from Wikipedia.
    if args.image_path is None:
        peach_url = "https://upload.wikimedia.org/wikipedia/commons/9/9e/Autumn_Red_peaches.jpg"
        img_path = ensure_downloaded(peach_url, "peach.jpg", is_tmp=True)
    else:
        img_path = Path(args.image_path)

    model = pretrained_sam(cast(PretrainedSamSize, args.key))
    predictor = model.predictor()

    peach_img = PIL.Image.open(img_path)
    predictor.set_image(np.array(peach_img))

    predictions, _, _ = predictor.predict()
    single_mask = predictions[0]
    mask = PIL.Image.fromarray(single_mask.astype(np.uint8) * 255)

    # Overlays the mask on the original image.
    mask.putalpha(128)
    peach_img.paste(mask, (0, 0), mask)

    if args.output_path is None:
        peach_img.show()
    else:
        peach_img.save(args.output_path)


if __name__ == "__main__":
    # python -m pretrained.sam
    test_pretrained_model()
