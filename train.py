#!/usr/bin/env python3

from models.aeskNet import AeskNet
from models.smallaeskNet import SmallAeskNet
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch.nn as nn
import torch
from torchvision import transforms, datasets

class Train():

    def __init__(self, args) -> None:

        # Get parameters from argparse
        self.num_epochs = args.epoch
        self.train_dir = args.train_set
        self.val_dir = args.val_set
        self.batch_size = args.batch_size
        self.patience = args.patience
        self.resized_image = args.img_size
        self.model = args.model_name
        
        if self.model == 1:
            self.model = AeskNet()  # DNN model
        if self.model == 2:
            self.model = SmallAeskNet() # Small DNN model for small images

        self.criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification

        self.optimizer = torch.optim.Adam(self.model.parameters())  # Optimizer
        
        # Initialize lists to store training and validation loss/accuracy
        self.train_losses = []
        self.train_accuracies = [] 
        self.val_losses = []
        self.val_accuracies = []  

        total_params = sum(p.numel() for p in self.model.parameters()) # Number of weight parameters to be trained
        print(f"Total number of parameters: {total_params}")

        transform = transforms.Compose([
            transforms.Resize((self.resized_image, self.resized_image)),  # Adjust based on model input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the dataset as a DataLoader object, creating batches 
        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=transform)
        self.val_dataset = datasets.ImageFolder(self.val_dir, transform=transform)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) 
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Move model to the chosen device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.weights_save_dir = args.weights_save_dir


    def train(self):

        best_val_loss = float('inf')  # Initialize best validation loss to hold the best training loss
        
        # Training loop
        for epoch in range(self.num_epochs):
            
            print(epoch + 1)

            # Iterate through training batches
            for images, labels in self.train_dataloader:
                
                # Move data to the device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass, calculate loss
                outputs = self.model(images)

                l1_regularization = 0
                for param in self.model.parameters():
                    l1_regularization += torch.sum(torch.abs(param))

                train_loss = self.criterion(outputs, labels) 
                self.train_losses.append(train_loss)
            
                # Calculate accuracy 
                _, predicted = torch.max(outputs.data, 1)  # Get the index of the maximum value
                correct = (predicted == labels).sum().item()  # Count correct predictions
                accuracy = correct / len(labels)  # Calculate accuracy for this batch
                self.train_accuracies.append(accuracy)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
            
            
            # Validation loop (optional)
            if self.val_dataloader:  # Only if validation data is loaded

                with torch.no_grad():  # Disable gradient calculation for validation
                    self.model.eval()  # Set model to evaluation mode 
                    
                    val_loss_sum = 0.0

                    for images, labels in self.val_dataloader:

                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        outputs = self.model(images)

                        val_loss = self.criterion(outputs, labels) 
                        val_loss_sum += val_loss.item()
                        self.val_losses.append(val_loss)

                        # Calculate accuracy 
                        _, predicted = torch.max(outputs.data, 1)  # Get the index of the maximum value
                        correct = (predicted == labels).sum().item()  # Count correct predictions
                        accuracy = correct / len(labels)  # Calculate accuracy for this batch
                        self.val_accuracies.append(accuracy)
                    
                    avg_val_loss = val_loss_sum / len(self.val_dataloader)

                    # If results are best so far save the weights as best
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(self.model.state_dict(), self.weights_save_dir + 'best.pth')

            # if self.early_stopping(self.model, self.patience): 
            #    break

        model_weights = self.model.state_dict()
        
        torch.save(model_weights, self.weights_save_dir + 'last.pth' )  

        self.visualize_results()


    # def early_stopping(self, model, patience=50, eps=1e-1):

    #     if not hasattr(self, 'prev_weights'):
    #         # Initialize previous weights only once at the beginning
    #         self.prev_weights = {name: param.data.clone() for name, param in model.named_parameters()}
    #         self.epochs_without_improvement = 0

    #     # Track current weights after training for this epoch
    #     current_weights = {name: param.data for name, param in model.named_parameters()}

    #     max_weight_change = 0.0  # Initialize maximum weight change

    #     # Compute maximum weight change for each layer
    #     for name in self.prev_weights:
    #         prev_weight = self.prev_weights[name]
    #         curr_weight = current_weights[name]
    #         weight_change = torch.max(torch.abs(prev_weight - curr_weight))
    #         max_weight_change = max(max_weight_change, weight_change.item())
        
    #     if max_weight_change > eps:
    #         self.epochs_without_improvement = 0
    #         self.prev_weights = current_weights  # Update previous weights
    #     else:
    #         self.epochs_without_improvement += 1
        
    #     if self.epochs_without_improvement == patience:
    #         print("Early stopping triggered after", self.epochs_without_improvement, "epochs with minimal weight change")
    #         return True
    #     return False

    def visualize_results(self):

        # For visualization convert tensors to numpy arrays
        train_losses_np = np.array([loss.item() for loss in self.train_losses])
        val_losses_np = np.array([loss.item() for loss in self.val_losses])
        train_accuracies_np = np.array(self.train_accuracies)
        val_accuracies_np = np.array(self.val_accuracies)

        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot training loss
        axes[0, 0].plot(train_losses_np, label='Training Loss')
        axes[0, 0].set_xlabel('Batch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_title('Training Loss')

        # Plot validation loss
        axes[0, 1].plot(val_losses_np, label='Validation Loss')
        axes[0, 1].set_xlabel('Batch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_title('Validation Loss')

        # Plot training accuracy
        axes[1, 0].plot(train_accuracies_np, label='Training Accuracy')
        axes[1, 0].set_xlabel('Batch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_title('Training Accuracy')

        # Plot validation accuracy
        axes[1, 1].plot(val_accuracies_np, label='Validation Accuracy')
        axes[1, 1].set_xlabel('Batch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_title('Validation Accuracy')

        plt.tight_layout()
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Train LeNet-like model")

    parser.add_argument("--epoch", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--train_set", type=str, default="dataset/train", help="Path to training set")
    parser.add_argument("--val_set", type=str, default="dataset/val", help="Path to validation set")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--img_size", type=int, default=16, help="Image resize dimension (square)")
    parser.add_argument("--weights_save_dir", type=str, default="results/", help="Where to save weights")
    parser.add_argument("--model_name", type=str, default= 2, help="Which model to use 1 for normal images, 2 for small images")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main = Train(args)
    main.train()
