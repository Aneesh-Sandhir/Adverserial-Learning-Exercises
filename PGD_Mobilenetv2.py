#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 00:19:39 2025

@author: asandhir
"""

from Attack import Attack

import torch

from PIL import Image
import requests
from tqdm import trange

import matplotlib.pyplot as plt

class PGD_Mobilenetv2(Attack):
    
    def __init__(self, image, label, epsilon = .00005, itterations = 200):
        super().__init__(image, label)
        
        # attack hyperparamters 
        self.epsilon = epsilon
        self.itterations = itterations
        
        # initialize confidence vectors
        self.epsilons = torch.linspace(0, (self.itterations * self.epsilon), steps = self.itterations)
        self.confidence_label_vector = torch.zeros(self.itterations)
        self.confidence_prediction_vector = torch.zeros_like(self.epsilons)
        self.confidence_top5_vector = torch.zeros_like(self.epsilons)
        
        # perform attack
        self.projected_gradient_descent(self.epsilon, self.itterations)
        self.grad_cam_gif[0].save("PGD_grad_cam.gif", save_all = True, 
                                  append_images = self.grad_cam_gif[1:], duration=100, loop=0)
        
    def projected_gradient_descent(self, epsilon, itterations):    
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
        
        augmented_image = self.image_tensor.clone().detach()
        
        for index in trange(itterations):
            # run inference while recording gradients
            augmented_image.requires_grad_()
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
                
            # calculate cross entropy loss on image and backpropogate it
            loss = torch.nn.functional.nll_loss(prediction, label_vector.long())
            loss.backward()
            
            # extract the sign of the resultant gradient at the input layer 
            augmented_image.requires_grad = False        
            adverserial_pattern = augmented_image.grad.data
            adverserial_pattern = adverserial_pattern/torch.abs(adverserial_pattern)
            
            # augment image
            augmented_image = augmented_image + (epsilon * adverserial_pattern)
            augmented_image = torch.clamp(augmented_image, -1, 1)
            
        self.confidence_label_vector = torch.nan_to_num(self.confidence_label_vector, nan = 0.0).detach().numpy()
        self.confidence_prediction_vector = torch.nan_to_num(self.confidence_prediction_vector, nan = 0.0).detach().numpy()
        self.confidence_top5_vector = torch.nan_to_num(self.confidence_top5_vector, nan = 0.0).detach().numpy()
    
if __name__ == "__main__":
    image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    image = Image.open(requests.get(image_url, stream = True).raw)
    image_label = 'Labrador retriever'
    
    attack = PGD_Mobilenetv2(image, image_label)
    
    fooled = (torch.Tensor(attack.confidence_label_vector - attack.confidence_prediction_vector) == 0).nonzero(as_tuple=True)[0][-1]
    not_top_5 = (torch.Tensor(attack.confidence_label_vector - attack.confidence_top5_vector) < 0).nonzero(as_tuple=True)[0][0]
    
    attack.plot_image(attack.image, 'Original Image')
    attack.plot_image(attack.image_tensor, 'Input Image')
    
    plt.ylabel('Confidence')
    plt.xlabel(f'Steps of size {attack.epsilon}')
    
    plt.plot(range(attack.itterations), attack.confidence_prediction_vector, label = 'Prediction Confidence')
    plt.plot(range(attack.itterations), attack.confidence_top5_vector, label = 'Top 5 Confidence')
    plt.plot(range(attack.itterations), attack.confidence_label_vector, label = 'Labeled Class Confidence')
    plt.scatter([fooled, not_top_5], [attack.confidence_label_vector[fooled], attack.confidence_label_vector[not_top_5]], color = 'red')
    
    