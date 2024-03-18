import random

from torchvision.transforms.functional import crop

class CustomCrop(object):
    """
    Crop the given image at specified location and output size.
    Inspired by: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, output_size, offset_range=(-128, 10)):
        # Output size
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        # Offset range
        assert isinstance(offset_range, (int, tuple))
        if isinstance(offset_range, int):
            self.offset_range = (offset_range, offset_range)
        else:
            assert len(offset_range) == 2
            self.offset_range = offset_range        

    def __call__(self, rgb, depth, mask, loc_x, loc_y):
        
        # Random offsets
        x = loc_x
        y = loc_y
        offset_x = -1
        offset_y = -1
        while True:
            offset_x = random.randint(self.offset_range[0], self.offset_range[1])
            offset_y = random.randint(self.offset_range[0], self.offset_range[1])
            x = loc_x + offset_x
            y = loc_y + offset_y
            if (x >= 0 and x < self.output_size[0] and
                y >= 0 and y < self.output_size[1]):
                break

        # Crop image
        if not isinstance(rgb, int):
            rgb = crop(rgb, y, x, self.output_size[0], self.output_size[1])
        if not isinstance(depth, int):
            depth = crop(depth, y, x, self.output_size[0], self.output_size[1])
        if not isinstance(mask, int):
            mask = crop(mask, y, x, self.output_size[0], self.output_size[1])

        return rgb, depth, mask, -offset_x, -offset_y