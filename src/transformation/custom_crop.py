import random

import torch
import torchvision.transforms as transforms

class RandomCrop(object):

    def __init__(self, output_size, offset_range=(-128, 10)):
        """
        Crop the given image at specified location and output size with a random offset.
        Inspired by: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

        Parameters
        ----------
        output_size : int | tuple[int, int]
            Size of the output image
        offset_range : int | tuple[int, int], optional
            Random offset to apply to the specified location, by default (-128, 10)
        """
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
        """
        Perform cropping.

        Parameters
        ----------
        rgb : torch.Tensor
            RGB modality
        depth : torch.Tensor
            Depth modality
        mask : torch.Tensor
            Mask modality
        loc_x : int
            Position of the object in the image (x coordinate)
        loc_y : int
            Position of the object in the image (y coordinate)

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor, int, int
            Cropped RGB, depth and mask modalities with position of the object in the cropped image
        """

        # Check inputs and inputs shapes
        assert loc_x != -1 # See RGBDObjectDataset
        assert loc_y != -1 # See RGBDObjectDataset

        # Retrieve input shape
        _, input_height, input_width = rgb.shape
        
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
            if (x >= 0 and x + self.output_size[0] < input_width and
                y >= 0 and y + self.output_size[1] < input_height ):
                break

        # Crop images
        if not isinstance(rgb, int):
            rgb = transforms.functional.crop(rgb, y, x, self.output_size[0], self.output_size[1])
        if not isinstance(depth, int):
            depth = transforms.functional.crop(depth, y, x, self.output_size[0], self.output_size[1])
        if not isinstance(mask, int):
            mask = transforms.functional.crop(mask, y, x, self.output_size[0], self.output_size[1])

        return rgb, depth, mask, -offset_x, -offset_y
    

class ObjectCrop(object):

    def __init__(self, output_size=None, padding=(20,20), offset_range=(-10, 10)):
        """
        Crop the object from the image (using location data from mask modality) with optional resizing, padding and offset.
        Inspired by: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

        Parameters
        ----------
        output_size : int | tuple[int, int], optional
            Size of the ouptut image, by default None
        padding : int | tuple[int, int], optional
            Size of the padding to apply in each direction, by default (20,20)
        offset_range : int | tuple[int, int], optional
            Random offset to apply to the specified location, by default (-10, 10)
        """
        # Output size
        assert isinstance(output_size, (int, tuple)) or output_size is None
        if output_size is None:
            self.output_size = None
            self.resize = None
        elif isinstance(output_size, int):
            self.output_size = (output_size, output_size)
            self.resize = transforms.Resize(size=self.output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            self.resize = transforms.Resize(size=self.output_size)

        # Padding
        assert isinstance(padding, (int, tuple))
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            assert len(padding) == 2
            self.padding = padding

        # Offset range
        assert isinstance(offset_range, (int, tuple))
        if isinstance(offset_range, int):
            self.offset_range = (offset_range, offset_range)
        else:
            assert len(offset_range) == 2
            self.offset_range = offset_range        

    def __call__(self, rgb, depth, mask, loc_x, loc_y):
        """
        Perform cropping.
    
        Parameters
        ----------
        rgb : torch.Tensor
            RGB modality
        depth : torch.Tensor
            Depth modality
        mask : torch.Tensor
            Mask modality
        loc_x : int
            Position of the object in the image (x coordinate)
        loc_y : int
            Position of the object in the image (y coordinate)

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor, int, int
            Cropped RGB, depth and mask modalities with position of the object in the cropped image
        """

        # Check inputs and inputs shapes
        assert isinstance(mask, torch.Tensor)

        # Retrieve input shape
        _, input_height, input_width = rgb.shape
        
        # Retrive min and max coordinates of object
        mask_coord = (mask[0,:,:] == 1).nonzero()
        min_x = torch.min(mask_coord[:,1])
        max_x = torch.max(mask_coord[:,1])
        min_y = torch.min(mask_coord[:,0])
        max_y = torch.max(mask_coord[:,0])

        # Add padding
        min_x -= max(0, self.padding[0])
        max_x += min(input_width, self.padding[0])
        min_y -= max(0, self.padding[1])
        max_y += min(input_height, self.padding[1])

        # Compute crop width and height
        crop_width = max_x - min_x
        crop_height = max_y - min_y
        
        # Random offsets
        x = min_x
        y = min_y
        offset_x = -1
        offset_y = -1
        while True:
            offset_x = random.randint(self.offset_range[0], self.offset_range[1])
            offset_y = random.randint(self.offset_range[0], self.offset_range[1])
            x = min_x + offset_x
            y = min_y + offset_y
            if (x >= 0 and x + crop_width < input_width and
                y >= 0 and y + crop_height < input_height):
                break

        # Crop image
        if not isinstance(rgb, int):
            rgb = transforms.functional.crop(rgb, y, x, crop_height, crop_width)
        if not isinstance(depth, int):
            depth = transforms.functional.crop(depth, y, x, crop_height, crop_width)
        if not isinstance(mask, int):
            mask = transforms.functional.crop(mask, y, x, crop_height, crop_width)

        # Resize image
        if self.output_size is not None:
            if not isinstance(rgb, int):
                rgb = self.resize(rgb)
            if not isinstance(depth, int):
                depth = self.resize(depth)
            if not isinstance(mask, int):
                mask = self.resize(mask)

        return rgb, depth, mask, max(0, -offset_x), max(0, -offset_y)