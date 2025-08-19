#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 17:31:22 2025

@author: asandhir
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 17:02:23 2025

@author: anees
"""

import torch
import torch.optim as optim
from torchvision import transforms 

from PIL import Image
import requests
import copy
from tqdm import trange 
import matplotlib.pyplot as plt

class CarliniWagner_pytorch:
    
    def __init__(self, image, label, targeted = False, confidence = 0,
                 learning_rate = 0.01, initial_constant = 0.001,
                 binary_search_steps = 1, max_itterations = 50, 
                 abort_early = False):
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
        
        self.targeted = targeted 
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.initial_constant = initial_constant
        self.binary_search_steps = binary_search_steps
        self.max_itterations = max_itterations
        self.abort_early = abort_early
        
        # model specific attributes
        self.pytorch_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=True).eval()
        model_categories_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
        self.model_categories = requests.get(model_categories_url).text.split('\n')
        self.label = torch.Tensor([self.model_categories.index(label)]).int()

        self.original_prediction = self.make_prediction(self.image_tensor)
                
        self.least_noisy_fooling_image = None
        self.least_noisy_top5_image = None
        
        self.confidence_label_vector = torch.zeros(self.max_itterations)
        self.confidence_prediction_vector = torch.zeros(self.max_itterations)
        self.confidence_top5_vector = torch.zeros(self.max_itterations)
        self.adverserial_image = self.carlini_wagner(self.pytorch_model, self.image_tensor, self.label)

    def carlini_wagner(self, model, inputs, labels):    
        
        batch_size = len(inputs)
        batch_view = lambda tensor: tensor.view(batch_size,  *[1] * (inputs.dim() - 1))
        tanh_inputs = torch.atanh((1 - 1e-6) * (2 * inputs - 1))
        multiplier = 1 if self.targeted else 1
        
        c = torch.full((batch_size, ), self.initial_constant)
        lowerbound = torch.zeros_like(c)
        upperbound = torch.full_like(c, 1e10)
        
        outer_best_l2 = torch.full_like(c, torch.inf)
        outer_best_adv = inputs.clone()
        outer_adv_found = torch.zeros(batch_size, dtype = torch.bool)
        
        i_total = 0
        for outer_step in range(self.binary_search_steps):
            modifier = torch.zeros_like(inputs, requires_grad = True)
            optimizer = optim.Adam([modifier], lr = self.learning_rate)
            best_l2 = torch.full_like(c, torch.inf)
            adv_found = torch.zeros(batch_size, dtype = torch.bool)
        
            if (self.binary_search_steps >= 10) & (outer_step == (self.binary_search_steps - 1)):
                c = upperbound
        
            prev = torch.inf
            for i in trange(self.max_itterations):
                adv_inputs = (torch.tanh(tanh_inputs + modifier) + 1)/2
                l2_squared = (adv_inputs - inputs).flatten(1).square().sum(1)
                l2 = l2_squared.detach().sqrt()
                logits = model(adv_inputs)
        
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
                
                confidences, classes = torch.sort(torch.nn.functional.softmax(model(adv_inputs)), descending = True)
                self.confidence_prediction_vector[i] = confidences[0, 0]
                self.confidence_top5_vector[i] = confidences[0, 5]
                label_index = (classes == labels[0]).nonzero(as_tuple = True)[-1]
                self.confidence_label_vector[i] = confidences[0, label_index]
                #print(f'{self.model_categories[classes[0, 0]]} confidence {confidences[0,0]: .4f}')
                #print(f'{self.model_categories[classes[0, label_index]]} confidence {confidences[0, label_index]: .4f}')
                
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
        training set, and reshaping it into a batch of 1 Carlini Wagner 
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
    image_label = 'Saluki'
    
    attack = CarliniWagner_pytorch(image, image_label)
    
    attack.plot_image(attack.image, 'Original Image')
    attack.plot_image(attack.image_tensor, 'Input Image')
    attack.plot_image(attack.image_tensor - attack.adverserial_image, 'Perturbations')
    attack.plot_image(attack.adverserial_image, 'Adverserial Image')
    
    plt.ylabel('Confidence')
    plt.xlabel('Itterations')
    
    plt.plot(torch.linspace(1, attack.max_itterations, attack.max_itterations), attack.confidence_prediction_vector.detach(), label = 'Prediction Confidence')
    plt.plot(torch.linspace(1, attack.max_itterations, attack.max_itterations), attack.confidence_top5_vector.detach(), label = 'Top 5 Confidence')
    plt.plot(torch.linspace(1, attack.max_itterations, attack.max_itterations), attack.confidence_label_vector.detach(), label = 'Labeled Class Confidence')
    