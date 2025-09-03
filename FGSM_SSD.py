# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 17:02:23 2025

@author: anees
"""

import torch
from torchvision import transforms 
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from PIL import Image
import requests
import copy
from tqdm import trange 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FGSM_SSDLite:
    
    def __init__(self, image, label):
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
        self.ssdlite = ssdlite320_mobilenet_v3_large(pretrained=True).eval();
        model_categories_url = 'https://raw.githubusercontent.com/levan92/coco-classes-mapping/refs/heads/master/coco91.names'
        self.model_categories = requests.get(model_categories_url).text.split('\n')
        self.model_categories.insert(0, '')
        self.label_index = self.model_categories.index(label)

        self.original_prediction = self.make_prediction(self.image_tensor)
        self.original_bbox = self.original_prediction[0]['boxes'][0]
        
        self.epsilons = torch.linspace(0, .25, steps = 26)
        self.confidence_label_vector = torch.zeros_like(self.epsilons)
        self.confidence_prediction_vector = torch.zeros_like(self.epsilons)
        self.iou_vector = torch.zeros_like(self.epsilons)
        self.adverserial_pattern = self.create_FGSM_detection_pattern()
        
        for index, epsilon in enumerate(self.epsilons):
            augmented_image =  self.augment_detection_image(epsilon).detach()
            augmented_detections = self.make_prediction(augmented_image)
            augmented_image_fpv = self.get_detections_fpv(augmented_image)
            
            #top detection
            augmented_bbox = augmented_detections[0]['boxes'][0]
            augmented_classification_index = augmented_detections[0]['labels'][0]
            augmented_classification = self.model_categories[augmented_classification_index]
            
            sort_keys = augmented_image_fpv[:, :, augmented_classification_index]
            sorted_indices = torch.argsort(sort_keys, descending = True)
            augmented_image_fpv = augmented_image_fpv[:,sorted_indices][0,0]
            self.confidence_label_vector[index] = augmented_image_fpv[0, self.label_index]
            self.confidence_prediction_vector[index] = augmented_image_fpv[0, augmented_classification_index]

            iou = self.calculate_iou(self.original_bbox, augmented_bbox)
            self.iou_vector[index] = iou
            
    def create_FGSM_detection_pattern(self):
        
        self.image_tensor.requires_grad = True
        original_detections = self.ssdlite(self.image_tensor)[0]
        
        label_index = self.model_categories.index(self.label)
        labeled_class_detection = (label_index == original_detections['labels']).nonzero()[0,0]
        loss = -original_detections['scores'][labeled_class_detection]
        
        self.ssdlite.zero_grad()
        loss.backward()
        adverserial_pattern = self.image_tensor.grad.sign()
        self.image_tensor = self.image_tensor.detach()
        
        return adverserial_pattern
        
    def augment_detection_image(self, epsilon):
        
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
            model_output = self.ssdlite(image_tensor)
        prediction = model_output
        
        return prediction 
    
    def get_detections_fpv(self, image_tensor):
        
        with torch.no_grad():
            # 1. Pass through backbone and head
            backbone_output = self.ssdlite.backbone(image_tensor)
            
            #2 Convert OrderedDict to list of feature maps
            features = list(backbone_output.values())
            head_outputs = self.ssdlite.head(features)
        
            # 3. Compute class logits (before NMS)
            class_logits = head_outputs['cls_logits']  # shape: [batch_size, num_anchors, num_classes]
            bbox_regression = head_outputs['bbox_regression']
        
            # 4. Apply softmax to get probabilities
            full_probability_vector = torch.softmax(class_logits, dim=-1)  # shape: [1, num_anchors, num_classes]
        
        return full_probability_vector
        class_index = self.model_categories.index(image_label)
        class_detections = class_probs[0, (class_probs[0].argmax(dim = 1) == class_index)]
        sorted_indices = torch.argsort(class_detections[:, class_index], descending = True)
        full_probability_vector = class_detections[sorted_indices]

        return full_probability_vector
    
    def calculate_iou(self, box_1, box_2):
        
        box_1_corner_1 = (box_1[0:2])
        box_1_corner_2 = (box_1[2:])
        box_1_width = abs(box_1_corner_2[0] - box_1_corner_1[0])
        box_1_height = abs(box_1_corner_2[1] - box_1_corner_1[ 1])
        box_1_area = box_1_width * box_1_height
        
        box_2_corner_1 = (box_2[0:2])
        box_2_corner_2 = (box_2[2:])
        box_2_width = abs(box_2_corner_2[0] - box_2_corner_1[0])
        box_2_height = abs(box_2_corner_2[1] - box_2_corner_1[ 1])
        box_2_area = box_2_width * box_2_height

        intersection_corner_1 = [max(box_1_corner_1[0], box_2_corner_1[0]), max(box_1_corner_1[1], box_2_corner_1[1])]
        intersection_corner_2 = [min(box_1_corner_2[0], box_2_corner_2[0]), min(box_1_corner_2[1], box_2_corner_2[1])]
        intersection_width = abs(intersection_corner_2[0] - intersection_corner_1[0])
        intersection_height = abs(intersection_corner_2[1] - intersection_corner_1[ 1])
        intersection_area = intersection_width * intersection_height
        
        iou = intersection_area / (box_1_area + box_2_area - intersection_area)
        return iou

    def prepare_image(self, image):
        """
        Transforms the input image into a fromat the model can process. 
        These transformations include resizing the image, casting it into a 
        torch.Tensor, and reshaping it into a batch of 1

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
            transforms.Resize((320, 320)),
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
    
    def plot_image(self, display_image, title = None, boxes = None):
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
            
        if boxes:
            box_index = 0
            corner1 = (boxes[0]['boxes'][box_index][0:2])
            corner2 = (boxes[0]['boxes'][box_index][2:])
            
            # Calculate rectangle properties
            x = min(corner1[0], corner2[0])
            y = min(corner1[1], corner2[1])
            width = abs(corner2[0] - corner1[0])
            height = abs(corner2[1] - corner1[1])
            
            # Plot
            rect = patches.Rectangle((x, y), width, height, edgecolor='blue', facecolor='none', linewidth=1)
            plt.gca().add_patch(rect)
            
            if (title == None):
                title = f"{self.model_categories[boxes['labels'][box_index]]} at {100 * boxes['scores'][box_index]: .4f}"
        plt.imshow(display_image)
        plt.title(title)
        plt.show()
        
if __name__ == "__main__":
    image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    image = Image.open(requests.get(image_url, stream = True).raw)
    image_label = 'dog'
    
    attack = FGSM_SSDLite(image, image_label)
    
    plt.plot(attack.epsilons, attack.confidence_prediction_vector)
    plt.plot(attack.epsilons, attack.confidence_label_vector)
    plt.plot(attack.epsilons, attack.iou_vector)