#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 01:38:51 2025

@author: asandhir
"""

from Attack import Attack

import torch

from PIL import Image
import requests
import copy
from tqdm import trange

import matplotlib.pyplot as plt

class DeepFool_Mobilenetv2(Attack):
    
    def __init__(self, image, label, itterations = 50, overshoot = .0001):
        super().__init__(image, label)
        
        # attack hyperparamters 
        self.itterations = itterations
        self.overshoot = overshoot
        
        # initialize confidence vectors
        self.itterations_vector = torch.linspace(0, self.itterations, self.itterations + 1)
        self.confidence_label_vector = torch.zeros(self.itterations + 1)
        self.confidence_prediction_vector = torch.zeros(self.itterations + 1)
        self.confidence_top5_vector = torch.zeros(self.itterations + 1)
        
        # perform attack
        self.deepfool(self.itterations, self.overshoot)
        self.grad_cam_gif[0].save("DeepFool_grad_cam.gif", save_all = True, 
                                  append_images = self.grad_cam_gif[1:], duration=100, loop=0)
        
    def deepfool(self, itterations, overshoot, class_count = 10):    
        """
        
        Parameters
        ----------
        itterations : integer
            The number of times the Deepfool algorithm will run.
        overshoot : float
            The amount the latest perturbation should exceed the distance 
            between its current state and the neares class boundary.
        class_count : integer, optional
            The number of classes the algorithm should cycle through in search
            of the nearest class boundary. The default is 10.

        Returns
        -------
        None.

        """
        
        predictions = self.mobilenetv2.forward(self.image_tensor)
        predictions = torch.nn.functional.softmax(predictions, dim = 1) 
        confidences, class_indicies = torch.sort(predictions, descending = True) 
        neighborhood = class_indicies[0, 0:class_count] # truncate class list
        
        perturbed_image = copy.deepcopy(self.image_tensor).requires_grad_()
        gradient = torch.zeros_like(self.image_tensor) 
        total_perterbations = torch.zeros_like(self.image_tensor)
        
        perturbed_predictions = self.mobilenetv2.forward(perturbed_image)
        perturbed_predictions = torch.nn.functional.softmax(perturbed_predictions, dim = 1) 
        confidences, class_indicies = torch.sort(perturbed_predictions, descending = True)

        self.confidence_prediction_vector[0] = confidences[0, 0]
        self.confidence_top5_vector[0] = confidences[0, 4]
        self.confidence_label_vector[0] = perturbed_predictions[0, neighborhood[0]]
        self.grad_cam_gif.append(self.grad_CAM(perturbed_image))
        
        for itteration in trange(itterations):
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
            perturbed_predictions = self.mobilenetv2.forward(perturbed_image)
            perturbed_predictions = torch.nn.functional.softmax(perturbed_predictions, dim = 1) 
            confidences, class_indicies = torch.sort(perturbed_predictions, descending = True)
        
            self.confidence_prediction_vector[itteration + 1] = confidences[0, 0]
            self.confidence_top5_vector[itteration + 1] = confidences[0, 4]
            self.confidence_label_vector[itteration + 1] = perturbed_predictions[0, neighborhood[0]].max()
            self.grad_cam_gif.append(self.grad_CAM(perturbed_image))
            
            if ((class_indicies[0, 0] != neighborhood[0]) & (self.least_noisy_fooling_image == None)):
                self.set_least_noisy_fooling_image(perturbed_image)
            
            if ((confidences[0, 4] > perturbed_predictions[0, neighborhood[0]]) & (self.least_noisy_top5_image == None)):
                self.set_least_noisy_top5_image(perturbed_image)
            
        total_perterbations = (1 + overshoot) * total_perterbations

if __name__ == "__main__":
    image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    image = Image.open(requests.get(image_url, stream = True).raw)
    image_label = 'Labrador retriever'
    
    attack = DeepFool_Mobilenetv2(image, image_label)
    
    fooled = (torch.Tensor(attack.confidence_label_vector - attack.confidence_prediction_vector) == 0).nonzero(as_tuple=True)[0][-1]
    not_top_5 = (torch.Tensor(attack.confidence_label_vector - attack.confidence_top5_vector) < 0).nonzero(as_tuple=True)[0][0]
    
    attack.plot_image(attack.image, 'Original Image')
    attack.plot_image(attack.image_tensor, 'Input Image')
    
    plt.ylabel('Confidence')
    plt.xlabel('Itterations')
    
    plt.plot(attack.itterations_vector, attack.confidence_prediction_vector.detach(), label = 'Prediction Confidence')
    plt.plot(attack.itterations_vector, attack.confidence_top5_vector.detach(), label = 'Top 5 Confidence')
    plt.plot(attack.itterations_vector, attack.confidence_label_vector.detach(), label = 'Labeled Class Confidence')
    plt.scatter([fooled, not_top_5], [attack.confidence_label_vector.detach()[fooled], attack.confidence_label_vector.detach()[not_top_5]], color = 'red')
    