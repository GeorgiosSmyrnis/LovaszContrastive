from dataclasses import replace
from typing import Callable, Tuple, TYPE_CHECKING, Dict

import numpy as np

from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.fields.rgb_image import get_random_crop
from ffcv.libffcv import imdecode, resize_crop
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.state import State

if TYPE_CHECKING:
    from ffcv.memory_managers.base import MemoryManager
    from ffcv.reader import Reader

IMAGE_MODES : Dict[str, int] = {}
IMAGE_MODES['jpg'] = 0
IMAGE_MODES['raw'] = 1

class MultiViewRandomResizedCropRGBImageDecoder(SimpleRGBImageDecoder):
    """Random resized crop decoder with multiple views per image.

    This decoder works like RandomResizedCropRGBImageDecoder, while also adding support
    for multiple views per image.
    """
    def __init__(self, output_size, num_views, scale=(0.08, 1.0), ratio=(0.75, 4/3)):
        super().__init__()
        self.output_size = output_size
        self.num_views = num_views
        self.scale = scale
        self.ratio = ratio

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        widths = self.metadata['width']
        heights = self.metadata['height']
        # We convert to uint64 to avoid overflows
        self.max_width = np.uint64(widths.max())
        self.max_height = np.uint64(heights.max())
        output_shape = (self.num_views, self.output_size[0], self.output_size[1], 3)
        my_dtype = np.dtype('<u1')

        return (
            replace(previous_state, jit_mode=True,
                    shape=output_shape, dtype=my_dtype),
            (AllocationQuery(output_shape, my_dtype),
            AllocationQuery((self.max_height * self.max_width * np.uint64(3),), my_dtype),  # type: ignore
            )
        )

    def generate_code(self) -> Callable:

        jpg = IMAGE_MODES['jpg']

        mem_read = self.memory_read
        my_range = Compiler.get_iterator()
        imdecode_c = Compiler.compile(imdecode)
        resize_crop_c = Compiler.compile(resize_crop)
        get_crop_c = Compiler.compile(self.get_crop_generator)

        scale = self.scale
        ratio = self.ratio
        num_views = self.num_views
        if isinstance(scale, tuple):
            scale = np.array(scale)
        if isinstance(ratio, tuple):
            ratio = np.array(ratio)

        def decode(batch_indices, my_storage, metadata, storage_state):
            destination, temp_storage = my_storage
            for dst_ix in my_range(len(batch_indices)):
                source_ix = batch_indices[dst_ix]
                field = metadata[source_ix]
                image_data = mem_read(field['data_ptr'], storage_state)  # type: ignore
                height = np.uint32(field['height'])
                width = np.uint32(field['width'])

                if field['mode'] == jpg:
                    temp_buffer = temp_storage[dst_ix]
                    imdecode_c(image_data, temp_buffer,
                               height, width, height, width, 0, 0, 1, 1, False, False)  # type: ignore
                    selected_size = 3 * height * width
                    temp_buffer = temp_buffer.reshape(-1)[:selected_size]
                    temp_buffer = temp_buffer.reshape(height, width, 3)

                else:
                    temp_buffer = image_data.reshape(height, width, 3)

                for view_idx in range(num_views):
                    i, j, h, w = get_crop_c(height, width, scale, ratio)

                    resize_crop_c(temp_buffer, i, i + h, j, j + w,
                                  destination[dst_ix, view_idx])

            return destination[:len(batch_indices)]

        decode.is_parallel = True
        return decode

    @property
    def get_crop_generator(self):
        return get_random_crop
