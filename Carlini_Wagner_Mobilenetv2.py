#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 19:44:49 2025

@author: asandhir
"""

from Attack import Attack

import torch
import torch.optim as optim
from torchvision import transforms 

from PIL import Image
import requests
import copy
from tqdm import trange

import numpy as np
import matplotlib.pyplot as plt

class CarliniWagner_Mobilenetv2(Attack):
    
    def __init__(self, image, label, targeted = False, target = None, 
                     learning_rate = 0.01, initial_constant = 0.001,
                     binary_search_steps = 5, max_itterations = 51, confidence = 0, 
                     abort_early = False):
        super().__init__(image, label)
        
        # attack hyperparamters 
        self.targeted = targeted 
        self.target = None
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.initial_constant = initial_constant
        self.binary_search_steps = binary_search_steps
        self.max_itterations = max_itterations
        self.abort_early = abort_early
        if self.targeted:
            self.target = torch.Tensor([self.model_categories.index(target)]).int()
            
        self.label = torch.Tensor([self.model_categories.index(label)]).int()
        
        # initialize confidence vectors
        self.itterations_vector = torch.linspace(0, self.max_itterations - 1, self.max_itterations)
        self.confidence_label_vector = torch.zeros(self.max_itterations)
        self.confidence_target_vector = torch.zeros(self.max_itterations)
        self.confidence_prediction_vector = torch.zeros(self.max_itterations)
        self.confidence_top5_vector = torch.zeros(self.max_itterations)

        # perform attack
        self.adverserial_image = self.carlini_wagner(self.image_tensor, self.label, self.target)
        self.grad_cam_gif[0].save("Animations/CW_grad_cam.gif", save_all = True, 
                                  append_images = self.grad_cam_gif[1:], duration=100, loop=0)
        
    def carlini_wagner(self, inputs, labels, target):    
        """
        

        Parameters
        ----------
        inputs : torch.Tensor 
            Batch of images to be perturbed [n x c x h x w] scaled between [0, 1]
        labels : torch.Tensor
            Indicies of the label classes for each image
        target : TYPE
            Indicies of the target classes for each image

        Returns
        -------
        outer_best_adv : torch.Tensor
            Batch of perturbed images [n x c x h x w] scaled between [0, 1]

        """
        
        if self.targeted:
            original_labels = labels
            labels = target
        
        batch_size = len(inputs)
        batch_view = lambda tensor: tensor.view(batch_size,  *[1] * (inputs.dim() - 1))
        tanh_inputs = torch.atanh((1 - 1e-6) * (2 * inputs - 1))
        multiplier = -1 if self.targeted else 1
        
        c = torch.full((batch_size, ), self.initial_constant)
        lowerbound = torch.zeros_like(c)
        upperbound = torch.full_like(c, 1e10)
        
        outer_best_l2 = torch.full_like(c, torch.inf)
        outer_best_adv = inputs.clone()
        outer_adv_found = torch.zeros(batch_size, dtype = torch.bool)
        
        for outer_step in trange(self.binary_search_steps, leave = True):
            modifier = torch.zeros_like(inputs, requires_grad = True)
            optimizer = optim.Adam([modifier], lr = self.learning_rate)
            best_l2 = torch.full_like(c, torch.inf)
            adv_found = torch.zeros(batch_size, dtype = torch.bool)
        
            if (self.binary_search_steps >= 10) & (outer_step == (self.binary_search_steps - 1)):
                c = upperbound
        
            prev = torch.inf
            self.grad_cam_gif = []
            for i in trange(self.max_itterations, leave = False):
                adv_inputs = (torch.tanh(tanh_inputs + modifier) + 1)/2
                l2_squared = (adv_inputs - inputs).flatten(1).square().sum(1)
                l2 = l2_squared.detach().sqrt()
                logits = self.mobilenetv2(adv_inputs)
        
                if (outer_step == 0) & (i == 0):
                    oh_labels = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1).long(), 1)
                    infh_labels = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1).long(), torch.inf)
        
                if self.targeted: 
                    predicted_classes = (logits - (self.confidence * oh_labels)).argmax(1)
                else:
                    predicted_classes = (logits + (self.confidence * oh_labels)).argmax(1)
                
                if self.targeted:
                    adverserial = (predicted_classes == labels) 
                else:
                    adverserial = (predicted_classes != labels)
                is_smaller = l2 < best_l2
                outer_is_smaller = l2 < outer_best_l2
                is_both = adverserial & is_smaller
                outer_is_both = adverserial & outer_is_smaller
        
                best_l2 = torch.where(is_both, l2, best_l2)
                adv_found.logical_or_(is_both)
                outer_best_l2 = torch.where(outer_is_both, l2, outer_best_l2)
                outer_adv_found.logical_or_(outer_is_both)
                outer_best_adv = torch.where(batch_view(outer_is_both), adv_inputs.detach(), outer_best_adv)
                confidences, classes = torch.sort(torch.nn.functional.softmax(self.mobilenetv2(adv_inputs), dim = 1),
                                                  descending = True)
                
                self.confidence_prediction_vector[i] = confidences[0, 0]
                self.confidence_top5_vector[i] = confidences[0, 4]                
                self.grad_cam_gif.append(self.grad_CAM(adv_inputs))
                if self.targeted:
                    label_index = (classes == labels[0]).nonzero(as_tuple = True)[-1]
                    self.confidence_target_vector[i] = confidences[0, label_index]
                    original_label_index = (classes == original_labels[0]).nonzero(as_tuple = True)[-1]
                    self.confidence_label_vector[i] = confidences[0, original_label_index]
                    
                    # save the least noisy images which are misclassified 
                    if ((classes[0, 0] != original_labels[0]) & (self.least_noisy_fooling_image == None)):
                        self.set_least_noisy_fooling_image(outer_best_adv)
                    
                    if ((confidences[0, 4] > confidences[0, original_labels[0]]) & (self.least_noisy_top5_image == None)):
                        self.set_least_noisy_top5_image(outer_best_adv)
                else:
                    label_index = (classes == labels[0]).nonzero(as_tuple = True)[-1]
                    self.confidence_label_vector[i] = confidences[0, label_index]
                    
                    # save the least noisy images which are misclassified 
                    if ((classes[0, 0] != labels[0]) & (self.least_noisy_fooling_image == None)):
                        self.set_least_noisy_fooling_image(outer_best_adv)
                    
                    if ((confidences[0, 4] > confidences[0, labels[0]]) & (self.least_noisy_top5_image == None)):
                        self.set_least_noisy_top5_image(outer_best_adv)

                class_logits = logits.gather(1, labels.unsqueeze(1).long()).squeeze(1)
                other_logits = (logits - infh_labels).amax(dim = 1)
                logits_dist = multiplier * (class_logits - other_logits)
                loss = l2_squared + c * (logits_dist + self.confidence).clamp_(min = 0)
        
                if self.abort_early and (i % (self.max_itterations // 10) == 0):
                    if (loss > prev * .9999).all():
                        break
                    prev = loss.detach()
                
                optimizer.zero_grad(set_to_none = None)
                modifier.grad = torch.autograd.grad(loss.sum(), modifier, only_inputs = True)[0]
                optimizer.step()
        
            upperbound[adv_found] = torch.min(upperbound[adv_found], c[adv_found])
            lowerbound[~adv_found] = torch.max(lowerbound[~adv_found], c[~adv_found])
            is_smaller = upperbound < 1e9
            c[is_smaller] = (lowerbound[is_smaller] + upperbound[is_smaller])/2
            c[(~is_smaller) & (~adv_found)] *= 10
            
        return outer_best_adv
    
    def prepare_image(self, image):
        """
        Transforms the input image into a fromat the model can process. 
        These transformations include resizing the image, casting it into a 
        torch.Tensor, and reshaping it into a batch of 1 Carlini Wagner 
        algortim needs the pixel values to be normalized between 0 and 1

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
            ])
        image = preprocessing(image)
        mini_batch = image.unsqueeze(0)
        
        return mini_batch

if __name__ == "__main__":
    image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    image = Image.open(requests.get(image_url, stream = True).raw)
    image_label = 'Saluki' # model already misclassifies this image becasue of how CW attack needs the image scaled
    
    attack = CarliniWagner_Mobilenetv2(image, image_label)
    
    fooled = (torch.Tensor(attack.confidence_label_vector - attack.confidence_prediction_vector) == 0).nonzero(as_tuple=True)[0][0]
    not_top_5 = (torch.Tensor(attack.confidence_label_vector - attack.confidence_top5_vector) < 0).nonzero(as_tuple=True)[0][0]
    
    attack.plot_image(attack.image, 'Original Image')
    attack.plot_image(attack.image_tensor, 'Input Image')
    
    plt.ylabel('Confidence')
    plt.xlabel('Itterations')
    
    plt.plot(attack.itterations_vector, attack.confidence_prediction_vector.detach(), label = 'Prediction Confidence')
    plt.plot(attack.itterations_vector, attack.confidence_top5_vector.detach(), label = 'Top 5 Confidence')
    plt.plot(attack.itterations_vector, attack.confidence_label_vector.detach(), label = 'Labeled Class Confidence')
    plt.scatter([fooled, not_top_5], [attack.confidence_label_vector.detach()[fooled], attack.confidence_label_vector.detach()[not_top_5]], color = 'red')
    plt.legend(loc='upper right')
    plt.show()
    
    target_label = 'hare'
    image_label = 'Labrador retriever'
    hareOfTheDog = CarliniWagner_Mobilenetv2(image, image_label, targeted = True, 
                                             target = target_label, binary_search_steps = 1,
                                             learning_rate = .01, initial_constant = 0.1,)
    
    fooled = (torch.Tensor(hareOfTheDog.confidence_target_vector - hareOfTheDog.confidence_prediction_vector) == 0).nonzero(as_tuple=True)[0][0]
    
    itteration_vector = torch.linspace(1, hareOfTheDog.max_itterations, hareOfTheDog.max_itterations)
    plt.plot(itteration_vector, hareOfTheDog.confidence_prediction_vector.detach(), label = 'Prediction Confidence')
    plt.plot(itteration_vector, hareOfTheDog.confidence_top5_vector.detach(), label = 'Top 5 Confidence')
    plt.plot(itteration_vector, hareOfTheDog.confidence_label_vector.detach(), label = 'Labeled Class Confidence')
    plt.plot(itteration_vector, hareOfTheDog.confidence_target_vector.detach(), label = 'Target Class Confidence')
    plt.legend()
    plt.scatter([fooled + 1], [hareOfTheDog.confidence_target_vector.detach()[fooled]], color = 'red')
    