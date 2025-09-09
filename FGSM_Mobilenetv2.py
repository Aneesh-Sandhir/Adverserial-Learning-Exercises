#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 00:31:55 2025

@author: asandhir
"""

from Attack import Attack

import torch

from PIL import Image
import requests
from tqdm import trange

import matplotlib.pyplot as plt

class FGSM_Mobilenetv2(Attack):
    
    def __init__(self, image, label, max_epsilon = .25, steps = 101):
        super().__init__(image, label)
        
        # attack hyperparamters 
        self.max_epsilon = max_epsilon
        self.steps = steps
        
        # initialize confidence vectors
        self.epsilons = torch.linspace(0, self.max_epsilon, steps = self.steps)
        self.confidence_label_vector = torch.zeros_like(self.epsilons)
        self.confidence_prediction_vector = torch.zeros_like(self.epsilons)
        self.confidence_top5_vector = torch.zeros_like(self.epsilons)
        
        # perform attack
        self.adverserial_pattern = self.create_adverserial_pattern()
        self.evaluate_epsilons()         
        self.grad_cam_gif[0].save("Animations/FGSM_grad_cam.gif", save_all = True, 
                                  append_images = self.grad_cam_gif[1:], duration=100, loop=0)
    
    def create_adverserial_pattern(self):    
        """
        Creates the perturbations to the image by calculating the gradients 
        after back-propogating the loss all the way to the input layer and 
        noting their directions
    
        Returns
        -------
        adverserial_pattern : torch.Tensor
            The adverserial pattern 
    
        """
        #create label as a vector
        label_vector = torch.zeros_like(self.original_prediction)
        category_index = self.model_categories.index(self.label)
        label_vector[category_index] = 1
        
        #run inference while recording gradients
        self.image_tensor.requires_grad_()
        model_output = self.mobilenetv2(self.image_tensor)
        prediction = torch.nn.functional.softmax(model_output[0], dim = 0)
        
        #calculate cross entropy loss on image and backpropogate it
        loss = torch.nn.functional.nll_loss(prediction, label_vector.long())
        loss.backward()
        
        #extract the sign of the resultant gradient at the input layer 
        self.image_tensor.requires_grad = False        
        adverserial_pattern = self.image_tensor.grad.data
        adverserial_pattern = adverserial_pattern/torch.abs(adverserial_pattern)
        
        return adverserial_pattern
    
    def evaluate_epsilons(self):
        
        category_index = self.model_categories.index(self.label)
        augmented_image = self.image_tensor.clone().detach()

        for index in trange(self.steps):
            self.grad_cam_gif.append(self.grad_CAM(augmented_image))
            model_output = self.mobilenetv2(augmented_image)
            prediction = torch.nn.functional.softmax(model_output[0], dim = 0)
            
            # record confidences of various classes    
            self.confidence_label_vector[index] = prediction[category_index]
            ordered_output, indicies = torch.sort(prediction, descending = True)
            self.confidence_prediction_vector[index] = ordered_output[0]
            self.confidence_top5_vector[index] = ordered_output[4]
            
            # save the least noisy images which are misclassified 
            if ((prediction.argmax() != category_index) & (self.least_noisy_fooling_image == None)):
                self.set_least_noisy_fooling_image(augmented_image)
            
            if ((ordered_output[4] > prediction[category_index]) & (self.least_noisy_top5_image == None)):
                self.set_least_noisy_top5_image(augmented_image)
            
            # apply adverserial pattern
            augmented_image = self.augment_image(self.epsilons[index])
        
        # unclear why these vectors need to be detached when their gradients are not used
        self.confidence_label_vector = self.confidence_label_vector.detach()
        self.confidence_prediction_vector = self.confidence_prediction_vector.detach()
        self.confidence_top5_vector = self.confidence_top5_vector.detach()
                
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
    
if __name__ == "__main__":
    image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    image = Image.open(requests.get(image_url, stream = True).raw)
    image_label = 'Labrador retriever'
    
    attack = FGSM_Mobilenetv2(image, image_label)
    
    fooled = (torch.Tensor(attack.confidence_label_vector - attack.confidence_prediction_vector) == 0).nonzero(as_tuple=True)[0][-1]
    not_top_5 = (torch.Tensor(attack.confidence_label_vector - attack.confidence_top5_vector) <= 0).nonzero(as_tuple=True)[0][0]
    
    attack.plot_image(attack.image, 'Original Image')
    attack.plot_image(attack.image_tensor, 'Input Image')
    
    plt.ylabel('Confidence')
    plt.xlabel('Epsilon Value')
    
    plt.plot(attack.epsilons, attack.confidence_prediction_vector, label = 'Prediction Confidence')
    plt.plot(attack.epsilons, attack.confidence_top5_vector, label = 'Top 5 Confidence')
    plt.plot(attack.epsilons, attack.confidence_label_vector, label = 'Labeled Class Confidence')
    plt.scatter([attack.epsilons[fooled], attack.epsilons[not_top_5]], [attack.confidence_label_vector[fooled], attack.confidence_label_vector[not_top_5]], color = 'red')
    plt.legend()