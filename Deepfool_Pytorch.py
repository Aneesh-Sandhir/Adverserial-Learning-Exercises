# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 17:02:23 2025

@author: anees
"""

import torch
from torchvision import transforms 

from PIL import Image
import requests
import copy

import matplotlib.pyplot as plt

class DeepFool_pytorch:
    
    def __init__(self, image, label, epsilon = .2, itterations = 200, overshoot = .02):
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
        self.itterations = itterations
        self.overshoot = overshoot
        
        # model specific attributes
        self.pytorch_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=True).eval()
        model_categories_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
        self.model_categories = requests.get(model_categories_url).text.split('\n')

        self.original_prediction = self.make_prediction(self.image_tensor)
                
        self.least_noisy_fooling_image = None
        self.least_noisy_top5_image = None
        self.confidence_label_vector = torch.zeros(self.itterations)
        self.confidence_prediction_vector = torch.zeros(self.itterations)
        self.confidence_top5_vector = torch.zeros(self.itterations)
        self.deep_fool(itterations, overshoot)

    def deep_fool(self, itterations, overshoot, class_count = 10):    
        """
        

        Parameters
        ----------
        epsilon : TYPE
            DESCRIPTION.
        itterations : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        predictions = self.pytorch_model.forward(self.image_tensor)
        predictions = torch.nn.functional.softmax(predictions, dim = 1) 
        confidences, class_indicies = torch.sort(predictions, descending = True) 
        neighborhood = class_indicies[0, 0:class_count] # truncate class list
        
        perturbed_image = copy.deepcopy(self.image_tensor).requires_grad_()
        gradient = torch.zeros_like(self.image_tensor) 
        total_perterbations = torch.zeros_like(self.image_tensor)
        
        perturbed_predictions = self.pytorch_model.forward(perturbed_image)
        perturbed_predictions = torch.nn.functional.softmax(perturbed_predictions, dim = 1) 
        
        for itteration in range(itterations):
            minimum_perturbation = torch.inf
            perturbed_predictions[0, neighborhood[0]].backward(retain_graph = True)
            gradient_datum = perturbed_image.grad.data.detach().clone()
            
            for neighbor in range(1, class_count):
                perturbed_image.grad.zero_()
        
                perturbed_predictions[0, neighborhood[neighbor]].backward(retain_graph = True)
                current_gradient = perturbed_image.grad.data.detach().clone()
                candidate_gradient = current_gradient - gradient_datum
                prediction_delta = (perturbed_predictions[0, neighborhood[neighbor]] - perturbed_predictions[0, neighborhood[0]]).data
        
                candidate_perturbation = abs(prediction_delta)/torch.linalg.norm(candidate_gradient.flatten())
        
                if candidate_perturbation < minimum_perturbation:
                    minimum_perturbation = candidate_perturbation
                    gradient = candidate_gradient
        
            incremental_perturbation = (minimum_perturbation + 1e-4) * gradient / torch.linalg.norm(gradient)
            total_perterbations = total_perterbations + incremental_perturbation
        
            perturbed_image = self.image_tensor + (1 + overshoot) * total_perterbations
            perturbed_image.requires_grad_()
            perturbed_predictions = self.pytorch_model.forward(perturbed_image)
            perturbed_predictions = torch.nn.functional.softmax(perturbed_predictions, dim = 1) 
            confidences, class_indicies = torch.sort(perturbed_predictions, descending = True)
        
            self.confidence_prediction_vector[itteration] = confidences[0, 0]
            self.confidence_top5_vector[itteration] = confidences[0, 4]
            self.confidence_label_vector[itteration] = perturbed_predictions[0, neighborhood[0]].max()
            
        total_perterbations = (1 + overshoot) * total_perterbations

            
    def augment_image(self, epsilon):
        """
        Applies the adverserial pattern 
        
        Parameters
        ----------
        epsilon : float
            Value specifiying the strength of the perturbation

        Returns
        -------
        augmneted_image : torch.Tensor
            The perturbed image
            
        """
        augmneted_image = self.image_tensor + (epsilon * self.adverserial_pattern)
        augmneted_image = torch.clamp(augmneted_image, -1, 1)
        
        return augmneted_image
        
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
            model_output = self.pytorch_model(image_tensor)
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
        
if __name__ == "__main__":
    image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    image = Image.open(requests.get(image_url, stream = True).raw)
    image_label = 'Labrador retriever'
    
    attack = DeepFool_pytorch(image, image_label)
    
    plt.plot(range(attack.itterations), attack.confidence_prediction_vector, label = 'Prediction Confidence')
    plt.plot(range(attack.itterations), attack.confidence_top5_vector, label = 'Top 5 Confidence')
    plt.plot(range(attack.itterations), attack.confidence_label_vector, label = 'Labeled Class Confidence')