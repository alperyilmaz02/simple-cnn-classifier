#!/usr/bin/env python3

from sensor_msgs.msg import Image
from models.aeskNet import AeskNet
from models.smallaeskNet import SmallAeskNet
import cv2  
import rospy
import numpy as np
import torch
from torchvision import transforms

class Test():

    def __init__(self) -> None:
            
        # Get parameters from yaml file 
        weight_path = rospy.get_param('~weight_path')
        image_path = rospy.get_param('~image_path')
        model_type = rospy.get_param('~model_type')
        self.img_size = rospy.get_param('~img_size')
        
        if model_type == 1:
            self.model = AeskNet()
        if model_type == 2:
            self.model = SmallAeskNet()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))  # Load weights
        self.model.to(self.device)  # Move model to device
        self.transform = transforms.Compose([  # Define image transformations
            transforms.Resize((self.img_size, self.img_size)),  # Resize 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = ["red","green"]

        total_params = sum(p.numel() for p in self.model.parameters()) # Number of weight parameters to be trained
        print(f"Total number of parameters: {total_params}")
        
        rospy.Subscriber(image_path, Image, self.callback)

    def callback(self, data):
        cv_image = self.convert_ros_to_cv(data)  # Convert ROS image to OpenCV
        resized_image = cv2.resize(cv_image,(self.img_size, self.img_size))
        torch_tensor = torch.from_numpy(resized_image.transpose(2, 0, 1)).float()
        tensor_image = self.transform(torch_tensor).unsqueeze(0).to(self.device)  # Preprocess
        
        with torch.no_grad():
            outputs = self.model(tensor_image)
            probabilities = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            confidence_scores, predicted = torch.max(probabilities, 1)  # Get predicted class
            predicted_class_id = predicted.item()  # Access prediction value
            confidence_score = confidence_scores.item() #Access confidence score
            print("Confidence score:", confidence_score)
            print("Predicted class:", self.classes[predicted_class_id])

    def convert_ros_to_cv(self, ros_image):
        cv_image = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(ros_image.height, ros_image.width, -1)
        return cv_image

if __name__ == '__main__':

    rospy.init_node('tester', anonymous=True)
    test = Test()
    rospy.spin()
