# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 17:02:23 2025

@author: anees
"""

import torch
from torchvision import transforms 

from PIL import Image
import requests
import io
from tqdm import trange

import matplotlib.pyplot as plt
import IPython.display
import numpy as np

class PGD_pytorch:
    
    def __init__(self, image, label, epsilon = .00005, itterations = 200):
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
        
        self.epsilon = epsilon
        self.itterations = itterations
        
        self.original_prediction = self.make_prediction(self.image_tensor)
        
        self.least_noisy_fooling_image = None
        self.least_noisy_top5_image = None
        self.epsilons = torch.linspace(0, (self.itterations * self.epsilon), steps = self.itterations)
        self.confidence_label_vector = torch.zeros(self.itterations)
        self.confidence_prediction_vector = torch.zeros_like(self.epsilons)
        self.confidence_top5_vector = torch.zeros_like(self.epsilons)
        self.grad_cam_gif = []
        
        self.prjected_gradient_descent(epsilon, itterations)
        self.grad_cam_gif[0].save("PGD_grad_cam.gif", save_all = True, append_images = self.grad_cam_gif[1:], duration=100, loop=0)
        

    def prjected_gradient_descent(self, epsilon, itterations):    
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
            #run inference while recording gradients
            augmented_image.requires_grad_()
            self.grad_cam_gif.append(self.grad_CAM(augmented_image))
            model_output = self.mobilenetv2(augmented_image)
            prediction = torch.nn.functional.softmax(model_output[0], dim = 0)
            
                
            self.confidence_label_vector[index] = prediction[category_index]
            ordered_output, indicies = torch.sort(prediction, descending = True)
            self.confidence_prediction_vector[index] = ordered_output[0]
            self.confidence_top5_vector[index] = ordered_output[4]
            
            if ((prediction.argmax() != category_index) & (self.least_noisy_fooling_image == None)):
                self.set_least_noisy_fooling_image(augmented_image)
            
            if ((ordered_output[4] > prediction[category_index]) & (self.least_noisy_top5_image == None)):
                self.set_least_noisy_top5_image(augmented_image)
                
            #calculate cross entropy loss on image and backpropogate it
            loss = torch.nn.functional.nll_loss(prediction, label_vector.long())
            loss.backward()
            
            #extract the sign of the resultant gradient at the input layer 
            augmented_image.requires_grad = False        
            adverserial_pattern = augmented_image.grad.data
            adverserial_pattern = adverserial_pattern/torch.abs(adverserial_pattern)
            
            #augment image
            augmented_image = augmented_image + (epsilon * adverserial_pattern)
            augmented_image = torch.clamp(augmented_image, -1, 1)
            
        self.confidence_label_vector = torch.nan_to_num(self.confidence_label_vector, nan = 0.0).detach().numpy()
        self.confidence_prediction_vector = torch.nan_to_num(self.confidence_prediction_vector, nan = 0.0).detach().numpy()
        self.confidence_top5_vector = torch.nan_to_num(self.confidence_top5_vector, nan = 0.0).detach().numpy()
    
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
        
        # ReLU and normalize
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max()
        
        display_image = self.image.resize(cam_image.shape[-2:])
        plt.imshow(display_image)
        
        cam = cam.detach().numpy()
        cam = np.uint8(255 * cam)
        cam = Image.fromarray(cam).resize(cam_image.shape[-2:], resample=Image.BILINEAR)
        cam = np.array(cam)
        
        plt.imshow(cam, cmap='jet', alpha=.50)  # Alpha for transparency
        plt.title(f"Grad-CAM for Class: {self.model_categories[predicted_class_index.item()]}")
        plt.axis('off')
        
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
        
if __name__ == "__main__":
    image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    image = Image.open(requests.get(image_url, stream = True).raw)
    image_label = 'Labrador retriever'
    
    attack = PGD_pytorch(image, image_label)
    
    fooled = (torch.Tensor(attack.confidence_label_vector - attack.confidence_prediction_vector) == 0).nonzero(as_tuple=True)[0][-1]
    not_top_5 = (torch.Tensor(attack.confidence_label_vector - attack.confidence_top5_vector) < 0).nonzero(as_tuple=True)[0][0]
    
    attack.plot_image(attack.image, 'Original Image')
    attack.plot_image(attack.image_tensor, 'Input Image')
    
    plt.ylabel('Confidence')
    plt.xlabel(f'Steps of size {attack.epsilon}')
    
    plt.plot(range(len(attack.epsilons)), attack.confidence_prediction_vector, label = 'Prediction Confidence')
    plt.plot(range(len(attack.epsilons)), attack.confidence_top5_vector, label = 'Top 5 Confidence')
    plt.plot(range(len(attack.epsilons)), attack.confidence_label_vector, label = 'Labeled Class Confidence')
    plt.scatter([fooled, not_top_5], [attack.confidence_label_vector[fooled], attack.confidence_label_vector[not_top_5]], color = 'red')
    
    