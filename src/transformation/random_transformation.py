import random

import torch
import torchvision.transforms as transforms

class RandomTransformation(object):

    def __init__(self, output_size):
        """
        Apply random transformation to an image.

        Parameters
        ----------
        output_size : int | tuple[int, int]
            Size of the output image
        """
        # Output size
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.h_flip = transforms.RandomHorizontalFlip(p=1)
        self.v_flip = transforms.RandomVerticalFlip(p=1)

    def __call__(self, rgb, depth, mask, loc_x, loc_y, label):
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

        if not isinstance(rgb, int):

            # Horizontal flip
            if random.randint(0, 1) == 1:
                rgb = self.h_flip(rgb)
                if not isinstance(depth, int):
                    depth = self.h_flip(depth)
                if not isinstance(mask, int):
                    mask = self.h_flip(mask)
            
            # Vertical flip
            if random.randint(0, 1) == 1:
                rgb = self.v_flip(rgb)
                if not isinstance(depth, int):
                    depth = self.v_flip(depth)
                if not isinstance(mask, int):
                    mask = self.v_flip(mask)

            # Color jitter
            if random.randint(0, 1) == 1:
                rgb = transforms.ColorJitter(brightness=random.random(),
                                             contrast=random.random(),
                                             saturation=random.random(),
                                             hue=random.uniform(0, 0.5))(rgb)
                
            # Gaussian blur
            # if random.randint(0, 1) == 1:
            #     rgb = transforms.GaussianBlur(kernel_size=(5, 5),
            #                                   sigma=(0.1, 5))(rgb)
                
            return rgb, depth, mask, loc_x, loc_y, label
                
        