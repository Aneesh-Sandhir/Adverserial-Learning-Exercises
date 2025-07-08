# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 17:02:23 2025

@author: anees
"""

import torch
from torchvision import transforms 

from PIL import Image
import requests

import numpy as np
import matplotlib.pyplot as plt

class FGSM_pytorch:
    
    def __init__(self, image, label):
        self.image = image
        self.image_tensor = self.prepare_image(self.image)
        self.label = label
        
        # model specific attributes
        self.pytorch_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=True)
        model_categories_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
        self.model_categories = requests.get(model_categories_url).text.split('\n')
        
        self.original_prediction = self.make_prediction(self.image_tensor)
        self.adverserial_pattern = self.create_adverserial_pattern()

    def create_adverserial_pattern(self):       
        
        label_vector = torch.zeros_like(self.original_prediction)
        category_index = self.model_categories.index(self.label)
        label_vector[category_index] = 1
        
        #inference cant be run via a seperate function  for some reason
        self.image_tensor.requires_grad_()
        self.pytorch_model.eval()
        model_output = self.pytorch_model(self.image_tensor)
        prediction = torch.nn.functional.softmax(model_output[0], dim = 0)
        
        loss = torch.nn.functional.nll_loss(prediction, label_vector.long())
        loss.backward()
        
        self.image_tensor.requires_grad = False        
        adverserial_pattern = self.image_tensor.grad.data
        adverserial_pattern = adverserial_pattern/torch.abs(adverserial_pattern)
        
        return adverserial_pattern
    
    def augment_image(self, epsilon):
        
        augmneted_image = self.image_tensor + (epsilon * self.adverserial_pattern)
        augmneted_image = torch.clamp(augmneted_image, -1, 1)
        
        return augmneted_image
        
    def make_prediction(self, image_tensor):
        
        self.pytorch_model.eval()
        
        with torch.no_grad():
            model_output = self.pytorch_model(image_tensor)
        prediction = torch.nn.functional.softmax(model_output[0], dim = 0)
        
        return prediction 

    def prepare_image(self, image):
        
        preprocessing = transforms.Compose([
            transforms.Resize((224, 244)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        image = preprocessing(image)
        mini_batch = image.unsqueeze(0)
        
        return mini_batch
        
    def get_predicted_category(self, model_output):
        
        highest_probability_index = model_output[0].argmax()
        predicted_category = self.model_categories[highest_probability_index]
        
        return predicted_category
    
    def get_category_probability(self, category, image_predictions):
        
        #image_predictions = torch.nn.functional.softmax(model_output[0], dim = 0)
        category_probability = 0
        try:
            category_index = self.model_categories.index(category)
        except ValueError:
            return category_probability
        
        category_probability = image_predictions[category_index]
        
        return category_probability
    
    def plot_image(self, display_image):
        if (type(display_image) == torch.Tensor):
            if (display_image.ndim == 4):
                display_image = display_image[0]    
            
            display_image = display_image.permute(1,2,0).numpy()
        
        plt.imshow(display_image)
        plt.show()
        
if __name__ == "__main__":
    image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    image = Image.open(requests.get(image_url, stream = True).raw)
    image_label = 'Labrador retriever'
    
    attack = FGSM_pytorch(image, image_label)
    attack.plot_image(attack.image)
    attack.plot_image(attack.image_tensor)
    attack.plot_image(attack.adverserial_pattern)
    
    epsilons = torch.linspace(0, 1, steps = 26)
    confidence_vector = torch.zeros_like(epsilons)
    for index, epsilon in enumerate(epsilons):
        augment_image = attack.augment_image(epsilon)
        prediction = attack.make_prediction(augment_image)
        confidence_vector[index] = attack.get_category_probability(image_label, prediction)
        
    plt.plot(epsilons, confidence_vector)
        
        
    
        