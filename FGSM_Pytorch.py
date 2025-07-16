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
        """
        Initializes the necessary inputs image, ground truth label, model, and 
        list of classes and performs a fast gradient sign method attack to fool
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
        self.pytorch_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=True).eval()
        model_categories_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
        self.model_categories = requests.get(model_categories_url).text.split('\n')
        
        self.original_prediction = self.make_prediction(self.image_tensor)
        self.adverserial_pattern = self.create_adverserial_pattern()

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
        model_output = self.pytorch_model(self.image_tensor)
        prediction = torch.nn.functional.softmax(model_output[0], dim = 0)
        
        #calculate cross entropy loss on image and backpropogate it
        loss = torch.nn.functional.nll_loss(prediction, label_vector.long())
        loss.backward()
        
        #extract the sign of the resultant gradient at the input layer 
        self.image_tensor.requires_grad = False        
        adverserial_pattern = self.image_tensor.grad.data
        adverserial_pattern = adverserial_pattern/torch.abs(adverserial_pattern)
        
        return adverserial_pattern
    
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
            transforms.Resize((224, 244)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        image = preprocessing(image)
        mini_batch = image.unsqueeze(0)
        
        return mini_batch
    
    def postprocess_image(self, image_tensor):
        
        postprocessing = transforms.Compose([
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225])
            ])
        display_image = postprocessing(image)[0]
        display_image = display_image.permute(1,2,0).numpy()

        return display_image        
        
        
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
    
    attack = FGSM_pytorch(image, image_label)
    attack.plot_image(attack.image, 'Original Image')
    attack.plot_image(attack.image_tensor, 'Input Image')
    attack.plot_image(attack.adverserial_pattern, 'Fast Gradient Sign Method Pattern')
    
    epsilons = torch.linspace(0, .25, steps = 251)
    confidence_label_vector = torch.zeros_like(epsilons)
    confidence_prediction_vector = torch.zeros_like(epsilons)
    confidence_top5_vector = torch.zeros_like(epsilons)
    
    largest_delta = {'magnitude': 0, 'image': None, 'position': 0}
    smallest_delta = {'magnitude': 1, 'image': None, 'position': 0}
    noisiest_top1 = {'image': None, 'position': 0}
    noisiest_top5 = {'image': None, 'position': 0}
    
    for index, epsilon in enumerate(epsilons):
        augmented_image = attack.augment_image(epsilon)
        output = attack.make_prediction(augmented_image)
        label_confidence = attack.get_category_probability(image_label, output)
        confidence_label_vector[index] = label_confidence
        
        predicted_class = attack.get_predicted_category(output)
        ordered_output, indicies = torch.sort(output, descending = True)
        
        confidence_prediction_vector[index] = ordered_output[0]
        confidence_top5_vector[index] = ordered_output[4]
        
        top1_delta = ordered_output[0] - label_confidence
        top5_delta = ordered_output[4] - label_confidence
        if (top1_delta == 0):
            noisiest_top1 = {'image': augmented_image[0], 'position': index}
            
        if (top5_delta == 0):
            noisiest_top5 = {'image': augmented_image[0], 'position': index}
        
        if (top1_delta > largest_delta['magnitude']):
            largest_delta = {'magnitude': top1_delta, 'image': augmented_image[0], 'position': index}
            
        if (top1_delta < smallest_delta['magnitude']) & (top1_delta > 0):
            smallest_delta = {'magnitude': top1_delta, 'image': augmented_image[0], 'position': index}

    attack.plot_image(noisiest_top1['image'], 'Noisiest Properly Classified Image')
    attack.plot_image(largest_delta['image'], 'Largest Confidence Gap Between\nPredicted and Labeled Classes')
    attack.plot_image(smallest_delta['image'], 'Smallest Confidence Gap Between\nPredicted and Labeled Classes')
    
    plt.ylabel('Confidence')
    plt.xlabel('Epsilon Value')
    plt.plot(epsilons, confidence_prediction_vector, label = 'Prediction Confidence')
    plt.plot(epsilons, confidence_top5_vector, label = 'Top 5 Confidence')
    plt.plot(epsilons, confidence_label_vector, label = 'Labeled Class Confidence')
    
    plt.scatter([epsilons[noisiest_top1['position']], epsilons[noisiest_top5['position']]], 
                [confidence_prediction_vector[noisiest_top1['position']], confidence_top5_vector[noisiest_top5['position']]], 
                color = 'red')
    
    plt.vlines(epsilons[largest_delta['position']], confidence_label_vector[largest_delta['position']],
               confidence_prediction_vector[largest_delta['position']])
    plt.vlines(epsilons[smallest_delta['position']], confidence_label_vector[smallest_delta['position']],
               confidence_prediction_vector[smallest_delta['position']])
    
    plt.legend()