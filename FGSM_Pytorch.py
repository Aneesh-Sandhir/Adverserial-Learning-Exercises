# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 17:02:23 2025

@author: anees
"""

import torch
from torchvision import transforms 

from PIL import Image
import requests

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
        highest_probability_index = model_output.argmax()
        predicted_category = self.model_categories[highest_probability_index]
        
        return predicted_category
    
    def get_category_probability(self, category, image_predictions):
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
    
    epsilons = torch.linspace(0, 1, steps = 251)
    confidence_label_vector = torch.zeros_like(epsilons)
    confidence_prediction_vector = torch.zeros_like(epsilons)
    
    largest_delta = 0
    largest_delta_image = None
    largest_delta_poi= 0
    smallest_delta = 1
    smallest_delta_image = None
    smallest_delta_poi = 0
    noisiest_properly_classified_image = image
    noisiest_proper_classification_poi = 0
    
    for index, epsilon in enumerate(epsilons):
        augmented_image = attack.augment_image(epsilon)
        output = attack.make_prediction(augmented_image)
        label_confidence = attack.get_category_probability(image_label, output)
        confidence_label_vector[index] = label_confidence
        
        predicted_class = attack.get_predicted_category(output)
        predicted_class_confidence = attack.get_category_probability(predicted_class, output)
        confidence_prediction_vector[index] = predicted_class_confidence
        
        delta = predicted_class_confidence - label_confidence
        if (delta == 0):
            noisiest_properly_classified_image = augmented_image
            noisiest_proper_classification_poi = epsilon
        
        if (delta > largest_delta):
            largest_delta = delta
            largest_delta_image = augmented_image
            largest_delta_poi = index
            
        if (delta < smallest_delta) & (delta > 0):
            smallest_delta = delta
            smallest_delta_image = augmented_image
            smallest_delta_poi = index    

    attack.plot_image(noisiest_properly_classified_image)
    attack.plot_image(largest_delta_image)
    attack.plot_image(smallest_delta_image)
    
    plt.ylabel('Confidence')
    plt.xlabel('Epsilon Value')
    plt.plot(epsilons, confidence_prediction_vector, label = 'Prediction Confidence')
    plt.plot(epsilons, confidence_label_vector, label = 'Labeled Class Confidence')
    
    plt.vlines(epsilons[largest_delta_poi], 0, confidence_prediction_vector[largest_delta_poi], 
               label = 'Largest Confidence Gap', color = 'limegreen')
    plt.vlines(epsilons[smallest_delta_poi], 0, confidence_prediction_vector[smallest_delta_poi],
               label = 'Smallest Confidence Gap', color = 'red')
    
    plt.legend()