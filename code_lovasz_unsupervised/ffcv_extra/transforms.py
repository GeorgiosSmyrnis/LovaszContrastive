"""FFCV extra transforms.

Extra transforms to be used with FFCV. The aim is to implement PyTorch transforms
in a way to achieve speedups with FFCV.
"""

from dataclasses import replace
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch as ch

import numpy as np
from numpy.random import rand

from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State

from .utils import get_colorjitter_params, rgb_to_grayscale, change_brightness, change_contrast, change_hue, change_saturation


class RandomGrayscale(Operation):
    """FFCV transform that randomly converts the channels of an image to grayscale.

    This operation is the same as that of the Pytorch version, with the difference
    that the image format expected is [N, H, W, C], so the channels should be the
    last element.

    Attributes:
        gray_prob: Probability to convert an image to grayscale. Default is 0.1.
    """

    def __init__(self, p: float = 0.1, multi_view: bool = False):
        super().__init__()
        self.gray_prob = p
        self.multi_view = multi_view

    def declare_state_and_memory(
        self, previous_state: State)-> Tuple[State, Optional[AllocationQuery]]:

        return (replace(previous_state, jit_mode=True),
            AllocationQuery(previous_state.shape, previous_state.dtype)
        )

    def generate_code(self) -> Callable:
        parallel_range = Compiler.get_iterator()
        gray_prob = self.gray_prob

        def random_grayscale_single_view(images, dst):
            should_gray = rand(images.shape[0]) < gray_prob
            for i in parallel_range(images.shape[0]):
                if should_gray[i]:
                    dst[i] = (rgb_to_grayscale(images[i].astype(np.float32) / 255.0) * 255.0).astype(np.uint8)
                else:
                    dst[i] = images[i]
            return dst

        def random_grayscale_multi_view(images, dst):
            should_gray = rand(images.shape[0], images.shape[1]) < gray_prob
            for i in parallel_range(images.shape[0]):
                for j in range(images.shape[1]):
                    if should_gray[i,j]:
                        dst[i,j] = (rgb_to_grayscale(images[i, j].astype(np.float32) / 255.0) * 255.0).astype(np.uint8)
                    else:
                        dst[i,j] = images[i,j]
            return dst

        random_grayscale_single_view.is_parallel = True
        random_grayscale_multi_view.is_parallel = True

        if self.multi_view:
            return random_grayscale_multi_view
        else:
            return random_grayscale_single_view


class ColorJitter(Operation):
    """Apply ColorJitter.

    This class automatically converts to float16.
    """

    def __init__(
        self,
        brightness: Union[float, Sequence[float]] = 0.0,
        contrast: Union[float, Sequence[float]] = 0.0,
        saturation: Union[float, Sequence[float]] = 0.0,
        hue: Union[float, Sequence[float]] = 0.0,
        apply_prob: float = 1.0,
        multi_view: bool = False,
        modify_hue: bool = True
    ):
        super().__init__()
        self.brightness = self._check_input(brightness)
        self.contrast = self._check_input(contrast)
        self.saturation = self._check_input(saturation)
        self.hue = self._check_input(hue, center=0, bound=(-0.5,0.5), clip_first_on_zero=False)
        self.apply_prob = apply_prob
        self.multi_view = multi_view
        self.modify_hue = modify_hue

    # This function taken almost as-is from the PyTorch implementation
    def _check_input(
        self,
        value: Union[float, Sequence[float]],
        center: float = 1.0,
        bound: Sequence[float] = (0.0, float("inf")),
        clip_first_on_zero: bool = True
    ) -> Optional[List[float]]:
        if isinstance(value, float):  # Only this condition is changed trivially.
            if value < 0:
                raise ValueError("If input is a single number, it must be non negative.")
            result = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                result[0] = max(result[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"Input values should be between {bound}")
            result = list(value)
        else:
            raise TypeError("Input should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if result[0] == result[1] == center:
            result = None
        return result

    def declare_state_and_memory(
        self, previous_state: State)-> Tuple[State, Optional[AllocationQuery]]:

        return (replace(previous_state, jit_mode=True),
            AllocationQuery(previous_state.shape, previous_state.dtype)
        )

    def generate_code(self) -> Callable:
        parallel_range = Compiler.get_iterator()
        brightness = None if self.brightness is None else np.array(self.brightness)
        contrast = None if self.contrast is None else np.array(self.contrast)
        saturation = None if self.saturation is None else np.array(self.saturation)
        hue = None if self.hue is None else np.array(self.hue)
        apply_prob = self.apply_prob
        modify_hue = self.modify_hue
        def apply_colorjitter_single_view(images, dst):
            should_apply = rand(images.shape[0]) < apply_prob
            for i in parallel_range(images.shape[0]):
                if should_apply[i]:
                    out = images[i].astype(np.float32) / 255.0
                    params = get_colorjitter_params(brightness, contrast, saturation, hue)
                    order = params[0]
                    brightness_factor = params[1]
                    contrast_factor = params[2]
                    saturation_factor = params[3]
                    hue_factor = params[4]
                    for j in range(4):
                        idx = order[j]
                        if idx == 0:
                            out = change_brightness(out, brightness_factor)
                        elif idx == 1:
                            out = change_contrast(out, contrast_factor)
                        elif idx == 2:
                            out = change_saturation(out, saturation_factor)
                        elif idx == 3:
                            if modify_hue:
                                out = change_hue(out, hue_factor)
                    dst[i] = (out * 255.0).astype(np.uint8)
                else:
                    dst[i] = images[i]

            return dst

        def apply_colorjitter_multi_view(images, dst):
            should_apply = rand(images.shape[0], images.shape[1]) < apply_prob
            for i in parallel_range(images.shape[0]):
                for v in range(images.shape[1]):
                    if should_apply[i,v]:
                        out = images[i,v].astype(np.float32) / 255.0
                        params = get_colorjitter_params(brightness, contrast, saturation, hue)
                        order = params[0]
                        brightness_factor = params[1]
                        contrast_factor = params[2]
                        saturation_factor = params[3]
                        hue_factor = params[4]
                        for j in range(4):
                            idx = order[j]
                            if idx == 0:
                                out = change_brightness(out, brightness_factor)
                            elif idx == 1:
                                out = change_contrast(out, contrast_factor)
                            elif idx == 2:
                                out = change_saturation(out, saturation_factor)
                            elif idx == 3:
                                if modify_hue:
                                    out = change_hue(out, hue_factor)
                        dst[i,v] = (out * 255.0).astype(np.uint8)
                    else:
                        dst[i,v] = images[i,v]
            return dst

        apply_colorjitter_single_view.is_parallel = True
        apply_colorjitter_multi_view.is_parallel = True

        if self.multi_view:
            return apply_colorjitter_multi_view
        else:
            return apply_colorjitter_single_view


class MultiViewRandomHorizontalFlip(Operation):
    """Randomly flip an image horizontally.

    This supports single view (N x H x W x C) and multiple view (N x V x H x W x C).
    """
    def __init__(self, flip_prob: float = 0.5):
        super().__init__()
        self.flip_prob = flip_prob

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        flip_prob = self.flip_prob

        def flip(images, dst):
            if len(images.shape) == 4:
                should_flip = rand(images.shape[0]) < flip_prob
                for i in my_range(images.shape[0]):
                    if should_flip[i]:
                        dst[i] = images[i, :, ::-1]
                    else:
                        dst[i] = images[i]

            elif len(images.shape) == 5:
                should_flip = rand(images.shape[0], images.shape[1]) < flip_prob
                for i in my_range(images.shape[0]):
                    for v in range(images.shape[1]):
                        if should_flip[i, v]:
                            dst[i, v] = images[i, v, :, ::-1]
                        else:
                            dst[i, v] = images[i, v]

            return dst

        flip.is_parallel = True
        return flip

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))



class MultiViewToTorchImage(Operation):
    """Change tensor to PyTorch format for images (B x V x C x H x W)

    ASSUMES INPUT IS LIKE (B x V x H x W x C)
    Parameters
    ----------
    convert_back_int16 : bool
        Convert to float16.
    """
    def __init__(self, channels_last=True, convert_back_int16=True):
        super().__init__()
        self.convert_int16 = convert_back_int16
        self.channels_last = channels_last
        self.enable_int16conv = False

    def generate_code(self) -> Callable:
        do_conv = self.enable_int16conv
        def to_torch_image(inp: ch.Tensor, dst):
            # Returns a permuted view of the same tensor
            if do_conv:
                inp = inp.view(dtype=ch.float16)
                pass

            inp = inp.permute(0, 1, 4, 2, 3)
            shape = inp.shape
            if self.channels_last:
                inp = inp.view((shape[0] * shape[1],) + shape[2:])
                assert inp.is_contiguous(memory_format=ch.channels_last)
                return inp.view(shape)

            dst[:inp.shape[0]] = inp.contiguous()
            return dst[:inp.shape[0]]

        return to_torch_image

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        alloc = None
        V, H, W, C = previous_state.shape
        new_shape = (V, C, H, W)

        new_type = previous_state.dtype
        if new_type is ch.int16 and self.convert_int16:
            new_type = ch.float16
            self.enable_int16conv = True

        if not self.channels_last:
            alloc = AllocationQuery((V, C, H, W), dtype=new_type)
        return replace(previous_state, shape=(V, C, H, W), dtype=new_type), alloc