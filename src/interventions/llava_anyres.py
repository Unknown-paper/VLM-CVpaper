from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Iterator

import numpy as np
import torch


def _permuted(patches, tile_permutation):
    if tile_permutation is None:
        return patches
    permutation = tuple(int(index) for index in tile_permutation)
    if sorted(permutation) != list(range(len(patches))):
        raise ValueError(
            f"tile_permutation must contain every patch index exactly once; "
            f"got {permutation} for {len(patches)} patches"
        )
    return [patches[index] for index in permutation]


def _transform_thumbnail_chw_numpy(thumbnail: np.ndarray, background: np.ndarray, mode: str) -> np.ndarray:
    """Apply information-preserving/removing thumbnail controls without changing its shape."""
    if mode == "normal":
        return thumbnail
    if mode in {"blank", "objects_removed"}:
        return np.broadcast_to(background.reshape(-1, 1, 1), thumbnail.shape).copy()
    if mode == "blurred":
        from scipy.ndimage import gaussian_filter

        blurred = gaussian_filter(thumbnail.astype(np.float32), sigma=(0, 12, 12), mode="nearest")
        if np.issubdtype(thumbnail.dtype, np.integer):
            limits = np.iinfo(thumbnail.dtype)
            blurred = np.clip(np.rint(blurred), limits.min, limits.max)
        return blurred.astype(thumbnail.dtype)
    if mode == "layout_only":
        from scipy.ndimage import label

        work = thumbnail.astype(np.float32)
        scale = 255.0 if np.issubdtype(thumbnail.dtype, np.integer) else max(1.0, float(np.max(np.abs(work))))
        mask = np.max(np.abs(work - background.reshape(-1, 1, 1)), axis=0) > 0.03 * scale
        components, count = label(mask)
        output = np.broadcast_to(background.reshape(-1, 1, 1), thumbnail.shape).copy()
        neutral = np.full((thumbnail.shape[0],), 96 if scale > 2 else 0.38, dtype=thumbnail.dtype)
        yy, xx = np.ogrid[: thumbnail.shape[1], : thumbnail.shape[2]]
        for component_id in range(1, count + 1):
            ys, xs = np.where(components == component_id)
            if len(xs) < 10:
                continue
            center_x, center_y = int(np.rint(xs.mean())), int(np.rint(ys.mean()))
            marker = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= 9 ** 2
            output[:, marker] = neutral[:, None]
        return output
    raise ValueError(f"Unknown global thumbnail mode: {mode}")


def _constant_pad_crop(image: torch.Tensor, top: int, left: int, size: int) -> torch.Tensor:
    """Extract a fixed crop, filling out-of-canvas pixels with the source corner color."""
    channels, height, width = image.shape
    fill = image[:, 0, 0].reshape(channels, 1, 1)
    out = fill.expand(channels, size, size).clone()
    src_y0, src_x0 = max(0, top), max(0, left)
    src_y1, src_x1 = min(height, top + size), min(width, left + size)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    dst_y0, dst_x0 = src_y0 - top, src_x0 - left
    out[:, dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = image[
        :, src_y0:src_y1, src_x0:src_x1
    ]
    return out


def _constant_pad_crop_numpy(image: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    """NumPy CHW equivalent used by Transformers' legacy slow processor."""
    channels, height, width = image.shape
    fill = image[:, 0, 0].reshape(channels, 1, 1)
    out = np.broadcast_to(fill, (channels, size, size)).copy()
    src_y0, src_x0 = max(0, top), max(0, left)
    src_y1, src_x1 = min(height, top + size), min(width, left + size)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    dst_y0, dst_x0 = src_y0 - top, src_x0 - left
    out[:, dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = image[
        :, src_y0:src_y1, src_x0:src_x1
    ]
    return out


@contextmanager
def phased_anyres_grid(
    image_processor,
    phase_x: int,
    phase_y: int,
    overlap_x: int = 0,
    overlap_y: int = 0,
    global_thumbnail_mode: str = "normal",
    tile_permutation: tuple[int, ...] | None = None,
) -> Iterator[None]:
    """Temporarily move only LLaVA-OneVision's high-resolution AnyRes crop origin.

    The global thumbnail produced by the original method remains unchanged. The
    number, size, and row-major order of high-resolution crops also remain fixed.
    """
    method_name = "_get_image_patches" if hasattr(image_processor, "_get_image_patches") else "get_image_patches"
    original = getattr(image_processor, method_name)

    def shifted(self, image, grid_pinpoints, size, patch_size, resample):
        from transformers.image_processing_utils import select_best_resolution
        from transformers.image_utils import ChannelDimension, SizeDict, get_image_size

        image_size = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        best_resolution = select_best_resolution(image_size, grid_pinpoints)
        resized = self._resize_for_patching(
            image, best_resolution, resample=resample, input_data_format=ChannelDimension.FIRST
        )
        padded = self._pad_for_patching(resized, best_resolution)
        _, height, width = padded.shape
        patches = []
        if not 0 <= overlap_x < patch_size or not 0 <= overlap_y < patch_size:
            raise ValueError(
                f"overlap must be in [0, {patch_size}); got x={overlap_x}, y={overlap_y}"
            )
        stride_x = patch_size - int(overlap_x)
        stride_y = patch_size - int(overlap_y)
        n_rows = height // patch_size
        n_cols = width // patch_size
        for row in range(n_rows):
            for col in range(n_cols):
                patches.append(
                    _constant_pad_crop(
                        padded,
                        top=row * stride_y + int(phase_y),
                        left=col * stride_x + int(phase_x),
                        size=patch_size,
                    )
                )
        size_height, size_width = size
        global_thumbnail = self.resize(
            image=image,
            size=SizeDict(height=size_height, width=size_width),
            resample=resample,
        )
        transformed = _transform_thumbnail_chw_numpy(
            global_thumbnail.detach().cpu().numpy(),
            image[:, 0, 0].detach().cpu().numpy(),
            global_thumbnail_mode,
        )
        global_thumbnail = torch.as_tensor(transformed, device=global_thumbnail.device, dtype=global_thumbnail.dtype)
        return [global_thumbnail] + _permuted(patches, tile_permutation)

    def shifted_numpy(
        self,
        image,
        grid_pinpoints,
        size,
        patch_size,
        resample,
        data_format,
        input_data_format,
    ):
        from transformers.image_processing_utils import select_best_resolution
        from transformers.image_transforms import resize, to_channel_dimension_format
        from transformers.image_utils import ChannelDimension, get_image_size

        image_size = get_image_size(image, channel_dim=input_data_format)
        best_resolution = select_best_resolution(image_size, grid_pinpoints)
        resized = self._resize_for_patching(
            image, best_resolution, resample=resample, input_data_format=input_data_format
        )
        padded = self._pad_for_patching(
            resized, best_resolution, input_data_format=input_data_format
        )
        padded_chw = to_channel_dimension_format(
            padded, channel_dim=ChannelDimension.FIRST, input_channel_dim=input_data_format
        )
        _, height, width = padded_chw.shape
        if not 0 <= overlap_x < patch_size or not 0 <= overlap_y < patch_size:
            raise ValueError(
                f"overlap must be in [0, {patch_size}); got x={overlap_x}, y={overlap_y}"
            )
        stride_x = patch_size - int(overlap_x)
        stride_y = patch_size - int(overlap_y)
        patches = []
        for row in range(height // patch_size):
            for col in range(width // patch_size):
                patch_chw = _constant_pad_crop_numpy(
                    padded_chw,
                    top=row * stride_y + int(phase_y),
                    left=col * stride_x + int(phase_x),
                    size=patch_size,
                )
                patches.append(
                    to_channel_dimension_format(
                        patch_chw,
                        channel_dim=data_format,
                        input_channel_dim=ChannelDimension.FIRST,
                    )
                )
        global_thumbnail = resize(
            image,
            size=size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        thumbnail_chw = to_channel_dimension_format(
            global_thumbnail,
            channel_dim=ChannelDimension.FIRST,
            input_channel_dim=data_format,
        )
        thumbnail_chw = _transform_thumbnail_chw_numpy(
            thumbnail_chw,
            padded_chw[:, 0, 0],
            global_thumbnail_mode,
        )
        global_thumbnail = to_channel_dimension_format(
            thumbnail_chw,
            channel_dim=data_format,
            input_channel_dim=ChannelDimension.FIRST,
        )
        return [global_thumbnail] + _permuted(patches, tile_permutation)

    replacement = shifted if method_name.startswith("_") else shifted_numpy
    setattr(image_processor, method_name, MethodType(replacement, image_processor))
    try:
        yield
    finally:
        setattr(image_processor, method_name, original)
