from typing import Tuple
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageEnhance, ImageFilter


class ElasticTransform:
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, random_state=None):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.random_state = random_state

    def __call__(self, img):
        if self.random_state is None:
            random_state = np.random.RandomState(None)
        else:
            random_state = self.random_state

        shape = img.size
        shape_size = shape[:2]

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-self.alpha_affine, self.alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), self.sigma) * self.alpha

        x, y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        return cv2.remap(img, indices[1].astype(np.float32), indices[0].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


class RandomNoise:
    """Add random noise to the image"""
    def __init__(self, noise_factor=0.1):
        self.noise_factor = noise_factor

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        noise = np.random.normal(0, self.noise_factor * 255, img.shape).astype(np.uint8)
        noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_img)


class RandomBrightnessContrast:
    """Randomly adjust brightness and contrast"""
    def __init__(self, brightness_range=0.2, contrast_range=0.2):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, img):
        # Brightness
        brightness_factor = 1 + random.uniform(-self.brightness_range, self.brightness_range)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        
        # Contrast
        contrast_factor = 1 + random.uniform(-self.contrast_range, self.contrast_range)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
        
        return img


class RandomPerspective:
    """Random perspective transformation"""
    def __init__(self, distortion_scale=0.2, p=0.5):
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            width, height = img.size
            half_height = height // 2
            half_width = width // 2
            
            # Define perspective points
            topleft = (random.randint(0, int(self.distortion_scale * half_width)), 
                      random.randint(0, int(self.distortion_scale * half_height)))
            topright = (width - random.randint(0, int(self.distortion_scale * half_width)), 
                       random.randint(0, int(self.distortion_scale * half_height)))
            bottomleft = (random.randint(0, int(self.distortion_scale * half_width)), 
                         height - random.randint(0, int(self.distortion_scale * half_height)))
            bottomright = (width - random.randint(0, int(self.distortion_scale * half_width)), 
                          height - random.randint(0, int(self.distortion_scale * half_height)))
            
            startpoints = [(0, 0), (width, 0), (0, height), (width, height)]
            endpoints = [topleft, topright, bottomleft, bottomright]
            
            return F.perspective(img, startpoints, endpoints)
        return img


def build_transforms(image_size: int = 64, grayscale: bool = True) -> Tuple[T.Compose, T.Compose]:
    """Build enhanced transforms with improved preprocessing and augmentation"""
    channels = 1 if grayscale else 3
    
    # Enhanced normalization - using ImageNet stats for better convergence
    if grayscale:
        normalize = T.Normalize(mean=[0.485], std=[0.229])  # Grayscale ImageNet stats
    else:
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Enhanced training transforms with comprehensive augmentation
    train_tfms = T.Compose([
        # Convert to grayscale first if needed
        T.Grayscale(num_output_channels=1) if grayscale else T.Lambda(lambda x: x),
        
        # Resize with better interpolation
        T.Resize((image_size + 8, image_size + 8), interpolation=T.InterpolationMode.BICUBIC),
        
        # Center crop to exact size
        T.CenterCrop(image_size),
        
        # Random crop with padding for better augmentation
        T.RandomCrop(image_size, padding=4, padding_mode='reflect'),
        
        # Geometric augmentations
        T.RandomApply([
            T.RandomAffine(
                degrees=15,  # Increased rotation range
                translate=(0.1, 0.1), 
                scale=(0.85, 1.15),  # Increased scale range
                shear=8,  # Increased shear
                fill=0,
                interpolation=T.InterpolationMode.BICUBIC
            ),
        ], p=0.8),
        
        # Perspective transformation
        T.RandomApply([RandomPerspective(distortion_scale=0.3, p=1.0)], p=0.4),
        
        # Brightness and contrast augmentation
        T.RandomApply([RandomBrightnessContrast(brightness_range=0.3, contrast_range=0.3)], p=0.6),
        
        # Noise augmentation
        T.RandomApply([RandomNoise(noise_factor=0.05)], p=0.3),
        
        # Blur augmentation
        T.RandomApply([
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        ], p=0.3),
        
        # Sharpening
        T.RandomApply([
            T.Lambda(lambda x: x.filter(ImageFilter.SHARPEN) if hasattr(x, 'filter') else x),
        ], p=0.2),
        
        # Random erasing (cutout)
        T.RandomApply([
            T.RandomErasing(p=1.0, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        ], p=0.3),
        
        # Convert to tensor
        T.ToTensor(),
        
        # Enhanced normalization
        normalize,
    ])

    # Enhanced validation transforms
    val_tfms = T.Compose([
        # Convert to grayscale first if needed
        T.Grayscale(num_output_channels=1) if grayscale else T.Lambda(lambda x: x),
        
        # Resize with better interpolation
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        
        # Convert to tensor
        T.ToTensor(),
        
        # Normalization
        normalize,
    ])

    return train_tfms, val_tfms


