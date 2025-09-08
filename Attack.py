#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 20:34:14 2025

@author: asandhir
"""

import torch
from torchvision import transforms 

from PIL import Image
import requests
import io
import warnings

import matplotlib.pyplot as plt
import numpy as np

class Attack:
    
    def __init__(self, image, label):
        """
        Initializes the necessary inputs image, ground truth label, model, and 
        list of classes and performs a projected gradient decent attack to fool
        the model with a perturbed image

        Parameters
        ----------
        image : PIL.Image
            Image to be perturbed 
        label : string
            Label of the image

        Returns
        -------
        None.

        """
        self.image = image
        self.image_tensor = self.prepare_image(self.image)
        self.label = label
        
        # model specific attributes
        self.mobilenetv2 = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=True).eval()
        model_categories_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
        self.model_categories = requests.get(model_categories_url).text.split('\n')

        self.original_prediction = self.make_prediction(self.image_tensor)
        self.least_noisy_fooling_image = None
        self.least_noisy_top5_image = None
        
        self.grad_cam_gif = []

    def make_prediction(self, image_tensor):
        """
        Runs inferes upon the loaded model with the given image_tensor
        
        Parameters
        ----------
        image_tensor : torch.Tensor 
            Input image batch [n x c x h x w]

        Returns
        -------
        prediction : torch.Tensor
            The first image's full probability vector as output by the model [1 x k]
            
        """
        with torch.no_grad():
            model_output = self.mobilenetv2(image_tensor)
        prediction = torch.nn.functional.softmax(model_output[0], dim = 0)
        
        return prediction 

    def prepare_image(self, image):
        """
        Transforms the input image into a fromat the model can process. 
        These transformations include resizing the image, casting it into a 
        torch.Tensor, normalizing the RGB values to align with  model's 
        training set, and reshaping it into a batch of 1

        Parameters
        ----------
        image : PIL.Image
            Image to be transformed

        Returns
        -------
        mini_batch : torch.Tensor
             Batch of transformed images [n x c x h x w]

        """
        preprocessing = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        image = preprocessing(image)
        mini_batch = image.unsqueeze(0)
        
        return mini_batch
        
    def get_predicted_category(self, model_output):
        """        
        Finds most likely category from the given model_output 
        
        Parameters
        ----------
        model_output : torch.Tensor
            Image's full probability vector as output by the model [1 x k]

        Returns
        -------
        predicted_category : string
            Name of the most likely class

        """
        highest_probability_index = model_output.argmax()
        predicted_category = self.model_categories[highest_probability_index]
        
        return predicted_category
    
    def get_category_probability(self, category, model_output):
        """
        Finds the probablity of the given category in the given model_output

        Parameters
        ----------
        category : string
            Category of interest
        model_output : torch.Tensor
            Image's full probability vector as output by the model [1 x k]

        Returns
        -------
        category_probability : float
            The probability of the given category

        """
        category_probability = 0
        try:
            category_index = self.model_categories.index(category)
        except ValueError:
            return category_probability
        
        category_probability = model_output[category_index]
        
        return category_probability
    
    def set_least_noisy_fooling_image(self, image):
        
        self.least_noisy_fooling_image = image
        
    def set_least_noisy_top5_image(self, image):
        
        self.least_noisy_top5_image = image
        
    def grad_CAM(self, input_image):
        
        cam_image = input_image.clone().detach()
        activations = []
        gradients = []
        target_layer = self.mobilenetv2.features[-1]
        
        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        # Register hooks within the model's final layer
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        output = self.mobilenetv2(cam_image)
        predicted_class_index = torch.argmax(output)
        
        self.mobilenetv2.zero_grad()
        output[0, predicted_class_index].backward()
        
        # Get hooked activations and gradients
        act = activations[0].squeeze(0)     # Shape: [C, H, W]
        grad = gradients[0].squeeze(0)      # Shape: [C, H, W]
        
        # Compute weights: average gradient for each channel
        weights = grad.mean(dim=(1, 2))     # Shape: [C]
        
        # Weighted sum of activations
        cam = torch.zeros(act.shape[1:], dtype=torch.float32)  # Shape: [H, W]
        for i, w in enumerate(weights):
            cam += w * act[i]
        
        # ReLU and scale
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max()
        
        # Create an Image from the Class Activation Map 
        cam = cam.detach().numpy()
        cam = np.uint8(255 * cam)
        cam = Image.fromarray(cam).resize(cam_image.shape[-2:], resample=Image.BILINEAR)
        cam = np.array(cam)
        
        # Plot the CAM over the original Image
        display_image = input_image[0].detach().clamp(0, 1).permute(1,2,0).numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.imshow(display_image)
        plt.imshow(cam, cmap='jet', alpha=.50)  # Alpha for transparency
        plt.axis('off')
        
        # COnvert the plot into an Image object
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        cam = Image.open(buf)

        return cam
    
    def plot_image(self, display_image, title = None):
        """
        Plots the given display_image

        Parameters
        ----------
        display_image : PIL.Image or torch.Tensor
            Image to be displayed
        title : string, optional
            Text to be displayed above the image. The default is None.

        Returns
        -------
        None.

        """
        if (type(display_image) == torch.Tensor):
            if (display_image.ndim == 4):
                display_image = display_image[0]    
            
            display_image = display_image.permute(1,2,0).numpy()
        
        plt.imshow(display_image)
        plt.title(title)
        plt.show()
        