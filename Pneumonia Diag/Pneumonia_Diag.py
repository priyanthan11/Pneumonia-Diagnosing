import sys
import warnings

from matplotlib.patches import BoxStyle


# Ignore python-level UserWarning
warnings.filterwarnings("ignore",category=UserWarning)

import os
import lightning.pytorch as pl
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
import tempfile

from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
from torchvision import datasets, transforms
from torchvision import models as tv_models
from PIL import Image

torch.set_float32_matmul_precision('medium')

import matplotlib.pyplot as plt
import unittest

# Data dir
data_dir = "C:/_____ DeepLearning AI/2 Deep Learning Pytorch/PyTorch Techniques and ecosystem Tools/4. Efficient Training Pipelines/Pneumonia Diag/chest_xray"


def display_dataset_count(data_dir):
    """ 
    Displays a formatted count of images per class for each dataset split.
    
    """
    splits = ["train","test","val"]

    for split in splits:
        split_path = os.path.join(data_dir,split)
        #print(split_path)
        if not os.path.exists(split_path):
            #print(f"{split_path} is not Exist")
            continue

        counts = {}

        for class_folder in sorted(os.listdir(split_path)):
            class_path = os.path.join(split_path,class_folder)
            #print(class_path)
            if not os.path.isdir(class_path):
                #print(f"{class_path} is not Exist")
                continue

            counts[class_folder] = len([
                f for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path,f))
                ])
        total = sum(counts.values())
        separator = '-' * 50

        print(f"/n--- {split.capitalize()} Set ---")
        for class_name, count in counts.items():
            print(f"  - {class_name:<26}: {count} images")
        print(separator)
        print(f"  Total: {total:>32} images")


# how many of images we got
#display_dataset_count(data_dir)


def display_random_images(train_dir, num_images=4):
    """
    Displayes a side-by-side grid of rando, images from each class in the training set

    Args:
        train_dir (str): Path to the training directory containing class subfolders.
        num_images (int): Number of images to display per class (max 4).

    """


    num_images = min(num_images,4)
    class_folders = sorted([f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,f))])
    print(class_folders)


    fig, axes = plt.subplots(
        len(class_folders), 
        num_images,
        figsize=(num_images * 3,len(class_folders)*3)
    )

    # Ensure axes is always 2D even with one class
    if len(class_folders) == 1:
        axes = [axes]

    for row,class_name in enumerate(class_folders):
        class_path = os.path.join(train_dir, class_name)
        all_images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path,f))]
        selected = random.sample(all_images,min(num_images,len(all_images)))

        for col in range(num_images):
            ax = axes[row][col]
            if col < len(selected):
                img_path = os.path.join(class_path,selected[col])
                img = Image.open(img_path).convert("L") # Grayscale for X-rays
                ax.imshow(img,cmap="gray")
                ax.axis("off")

                # Only label the first image in each row
                if col == 0:
                    ax.set_title(class_name, fontsize=11,fontweight="bold",color="white",loc="left",bbox=dict(boxstyle="round,pad=0.3",facecolor="#2c7bb6" if class_name == "NORMAL" else "#d7191c",alpha=0.85))
            else:
                ax.axis("off")

    fig.suptitle("Chest X-Ray Samples by Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()

#display_random_images(train_dir="C:/_____ DeepLearning AI/2 Deep Learning Pytorch/PyTorch Techniques and ecosystem Tools/4. Efficient Training Pipelines/Pneumonia Diag/chest_xray/train")




### Building the Data Module
"""
TRAIN_TRAINSFORM and VAL_TRANSFORM: these are the torchvision transform piplines. The training pipeline includes data augmentation to help your
model generalize, while the validation pipline performs only the necessary preprocessing for evaluation.
"""

# Define the transformations to be applied to the training images.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2,contrast=0.2),
    # Perform a random affine transformations: shifitng and scaling the Image
    transforms.RandomAffine(degrees=0,translate=(0.1,0.1),scale=(0.9,1.1)),
    transforms.ToTensor(),
    # Normalize the tensor image with a precalculated mean and standard deviation of this dataset
    transforms.Normalize([0.482,0.482,0.482],[0.222,0.222,0.222])
    ])

# Define the transformation for the valiation images
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.482, 0.482, 0.482], [0.222, 0.222, 0.222])
    ])



def create_datasets(train_path,val_path,train_transform,val_transform):
    """
    Create and returns the necessary dataset for training and validatiaon
    Args:
        train_path (str): The file path to the training images.
        val_path (str): The file path to the validation images.
        train_transform (callable): Transformations to apply to the training data.
        val_transform (callable): Transformations to apply to the validation data.

    Returns:
        A tuple containing the train_dataset and val_dataset.
    """
    # Create the training and validatain datasets from the image folders.
    train_dataset = datasets.ImageFolder(train_path,train_transform)
    val_dataset = datasets.ImageFolder(val_path,val_transform)

    return train_dataset,val_dataset

def load_dataloader(train_dataset,val_dataset,batch_size,is_train_loader):
    """
    Creates and returns either a training or validation DataLoader
    Args:
        train_dataset (Dataset): The dataset for the training loader.
        val_dataset (Dataset): The dataset for the validation loader.
        batch_size (int): The number of samples per batch to load.
        is_train_loader (bool): A flag to determine which loader to create.
                                If True, creates the training loader.
                                If False, creates the validation loader.

    Returns:
        DataLoader: The configured PyTorch DataLoader.
    """
    # Check if the flag to derermine whether to create a traing or validation loader
    if is_train_loader:
        # Create the training Loader
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=10,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
            )
    else:
        # Create the val loader
        loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=10,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
            )

    return loader



### Assembling th eChestXRayDataModule
"""
The DataModule acts as the organizational backbone for the enritre data pipeline. It encapsulates all data related logic, from loading and
splitting to transforming and batching, into a single, resuable class. This practice keeps the main model code clean and ensures the data 
handing is reproducible, which is essential for any medical AI application.

"""
class ChestXRayDataModule(pl.LightningDataModule):
    """
    A LightningDataModule encapsulates all the steps involved in prepareing data for a PyTorch model.

    """
    def __init__(self,data_dir,batch_size=64):
        """
        Initializes the DataModule and stores key parameters.

        Args:
            data_dir (str): Directory where the data is stored.
            batch_size (int): Number of samples per batch in the DataLoader.
        """
        super().__init__()

        # Store the constructor arguments as instance attributes.
        self.data_dir = data_dir
        self.batch_size=batch_size

        # Assign the globally defined transformations to this instance.
        self.train_transform = TRAIN_TRANSFORM
        self.val_transform = VAL_TRANSFORM

        # Placeholders for datasets, to be assigned in setup()
        self.train_dataset = None
        self.val_dataset = None

    def setup(self,stage=None):
        """
        Assigns the train and validation datasets.

        Args:
        stage (str, optional): The stage of training (e.g., 'fit', 'test').
                               The Lightning Trainer requires this argument, but it is not
                               utilized in this implementation as the setup logic is the
                               same for all stages. Defaults to None.
        """
        # Construct the full paths to the train and validation image folders
        train_path = os.path.join(self.data_dir,"train")
        val_path = os.path.join(self.data_dir,"val")

        self.train_dataset, self.val_dataset = create_datasets(
            train_path,
            val_path,
            self.train_transform,
            self.val_transform
            )

    def train_dataloader(self):
        """ Returns the DataLoader for the training set."""
        return load_dataloader(self.train_dataset,self.val_dataset,self.batch_size,True)

    def val_dataloader(self):
        """ Returns the Dataloader for the validataion set."""
        return load_dataloader(self.train_dataset,self.val_dataset,self.batch_size,False)






# Testing

# ## Instantiate the DataModule for verificaiton
# dm_verify = ChestXRayDataModule(data_dir = data_dir,batch_size=8)

# # Setup the dataset
# dm_verify.setup()
# train_loader_verify = dm_verify.train_dataloader()
# val_loader_verify = dm_verify.val_dataloader()

# # --- Verify the Training Set ---
# print("--- Training Set ---")
# print(f"Total samples in the training dataset:    {len(dm_verify.train_dataset)}")
# print(f"DataLoader batch size:                    {train_loader_verify.batch_size}")
# print(f"DataLoader length (number of batches):    {len(train_loader_verify)}")

# # --- Verify the Validation Set ---
# print("/n--- Validation Set ---")
# print(f"Total samples in the validation dataset:  {len(dm_verify.val_dataset)}")
# print(f"DataLoader batch size:                    {val_loader_verify.batch_size}")
# print(f"DataLoader length (number of batches):    {len(val_loader_verify)}")


### Building the LightningModule

# Utilites for the LightningModule

def load_resnet18(num_classes,weights_path):
    """
    Initializes a ResNet-18 model, loads weights, and sets it up for transfer learning (feature Extraction)
     Args:
        num_classes (int): The number of output classes for the new classifier head.
        weights_path (str): The file path to the saved .pth model weights.

    Returns:
        A PyTorch model (ResNet-18) where all layers are frozen except for
        the final classifier head.
    """
    # Initialize a ResNet-18 model without pre-trained weights
    model = tv_models.resnet18(weights=None)

    # Replace the classifier head to match the number of classes for the new task
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,num_classes)

    # Load the state dictionary (weights) from the Local file
    state_dict = torch.load(weights_path,map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # Freeze all the parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze ONLY the parameters of the new classifier head.
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

# Define_optimize_and_scheduler(): this function returns the AdamW optimizer and a ReduceLROnPlateau learning rate scheduler. This scheduler will automattically
#reduce the larning rate if the validation loss stops improiving, which is a common technique for effective training.
def define_optimizer_and_scheduler(model, learning_rate,weight_decay):
    """
    Defines the optimizer and learning rate scheduler for the model.
    Args:
        model (nn.Module): The model for which to configure the optimizer.
                           Its parameters will be passed to the optimizer.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay (L2 penalty) for the optimizer.

    Returns:
        A tuple containing the configured optimizer and lr_scheduler.
    """

    # Create the optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
        )

    # Create the learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2
        )
    
    return optimizer,scheduler



# ChestXRayClassifier
class ChestXRAYClassifier(pl.LightningModule):
    """
    A LightningModule that is focused on tracking validation loss and accuracy.
    
    """

    def __init__(self,model_weights_path,num_classes=3,learning_rate=1e-3,weight_decay=1e-2):
        """
        Initializes the ChestXRayClassifier module.

        Args:
            model_weights_path (str): The file path to the pre-trained ResNet-18 model weights.
            num_classes (int): The number of classes for classification. Defaults to 3.
            learning_rate (float): The learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float): The weight decay (L2 penalty) for the optimizer. Defaults to 1e-2.
        """
        super().__init__()
        # Save all __init__ arguments (model_weights_path, num_classes,etc.) to self.hparams
        #for example, you'll access 'num_classes'as 'self.hparams.num_classes'
        self.save_hyperparameters()

        # Call the load_resnet18 function to get the pre-trainned model.
        self.model = load_resnet18(self.hparams.num_classes,self.hparams.model_weights_path)

        # Define the corss entropy loss function.
        self.loss_fn = nn.CrossEntropyLoss()

        # Define the accuracy metric using Accuracy
        self.accuracy = Accuracy(task="multiclass",num_classes=self.hparams.num_classes)


    def forward(self,x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of images.
        """
        return self.model(x)

    def training_step(self,batch,batch_idx=None):
        """
        Performs a single training step. Loss calculation is required for backpropagation.

        Args:
        batch (tuple): A tuple containing the input images and their labels.
        batch_idx (int): The index of the current batch. The Lightning Trainer
                         requires this argument, but it's not utilized in this
                         implementation as the logic is the same for all batches.
        """
        # Unpack the batch into images and labels
        images,labels = batch
        # Perform a forward pass to get the model's logits
        logits = self.forward(images)
        # Calculate the loss by comparing the logits to the true labels.
        loss = self.loss_fn(logits,labels)

        return loss

    def validation_step(self,batch,batch_idx=None):
        """
        Performs a single validation step and logs only the loss and accuracy.

        Args:
        batch (tuple): A tuple containing the input images and their labels.
        batch_idx (int): The index of the current batch. The Lightning Trainer
                         requires this argument, but it's not utilized in this
                         implementation as the logic is the same for all batches.
        """

        # Unpack the batch into images and labels
        images, labels = batch
        # Perform a forward pass to get the model's logits
        logits = self.forward(images)
        # calculate the loss
        loss = self.loss_fn(logits,labels)
        # Calculate the Accuracy
        acc = self.accuracy(logits,labels)

        # Log metrics for this valicatin epoch and show them in the progress bar
        self.log_dict({'val_loss':loss,'val_acc':acc},prog_bar=True)

    def configure_optimizers(self):
        """
        configure the iptimizers and learning rate scheduler
        
        """
        optimizer, scheduler = define_optimizer_and_scheduler(
            self.model,
            self.hparams.learning_rate,
            self.hparams.weight_decay
        )
        return {"optimizer":optimizer,"lr_scheduler":{"scheduler":scheduler,"monitor":"val_loss"}}


def setup_dummy_weights(path="./dummy_weights.pth",num_classes=2):
    """
    Creates a temporary ResNet-18 weights file for verification purposes.
    
    Args:
        path (str): Path where the dummy weights file will be saved.
    
    Returns:
        str: The path to the saved weights file.
    
    """
    # Create a ResNet-18 model with random (untrained) weights
    dummy_model = tv_models.resnet18(weights=None)

    # Replace the fc head to match exactly what load_resnet18 builds
    in_feature = dummy_model.fc.in_features
    dummy_model.fc = nn.Linear(in_feature,num_classes)

    # Save only the state dict (the wiehgts), not the full models
    torch.save(dummy_model.state_dict(),path)

    print(f"Dummy weights saved to: {path}")
    return path
def cleanup_dummy_weights(path="./dummy_weights.pth"):
    """
    Deletes the temporary dummy weights file after verification.
    
    Args:
        path (str): Path to the dummy weights file to delete.
    """
    if os.path.exists(path):
        os.remove(path)
        print(f"Cleaned up dummy weights at: {path}")
    else:
        print(f"No file found at: {path}")


### Testing 
# Instantiate the classifier for verification
# weights_path = setup_dummy_weights()
# verify_model = ChestXRAYClassifier(model_weights_path=weights_path)

# # Get the optimizer and scheduler
# optimizer_config = verify_model.configure_optimizers()
# optimizer = optimizer_config["optimizer"]
# scheduler = optimizer_config["lr_scheduler"]["scheduler"]

# # --- Print the results to verify ---
# print("--- LightningModule Components ---")
# print(f"Model Architecture: {verify_model.model.__class__.__name__}")
# print(f"Classifier Head:    {verify_model.model.fc}")
# print(f"Loss Function:      {verify_model.loss_fn.__class__.__name__}")
# print(f"Accuracy Metric:    {verify_model.accuracy.__class__.__name__}(num_classes={verify_model.accuracy.num_classes})")
# print(f"Optimizer:          {optimizer.__class__.__name__}")
# print(f"LR Scheduler:       {scheduler.__class__.__name__}")
# # Clean dummy weights

# cleanup_dummy_weights()






#### Training the Model

# Configure early stopping
def early_stopping(num_epochs,stop_threshold):
    """
     Configures and returns a Lightning EarlyStopping callback.

    Args:
        num_epochs (int): The maximum number of epochs, used to set patience.
        stop_threshold (float): The validation accuracy threshold to stop training.

    Returns:
        EarlyStopping: The configured Lightning callback.
    """
    stop = EarlyStopping(
        # Specify the metric to monitor as validation accurary
        monitor = "val_acc",
        # Set the value that monitored metric must reach to stop training.
        stopping_threshold=stop_threshold,
        # Define the number of epochs to wait for an improvement in val_acc. If the accuracy doesn't improve for this many consecutive epochs, training will stop. 
        # You should set this to half of the total num_epochs. Remember, patience must be an integer.
        patience=num_epochs//2,
        # Higher accuracy is better, so 
        mode="max"
        )
    return stop

# Define sample parameters for verification.
# num_epochs_verify = 15
# stop_threshold_verify = 0.90

# # Call the function to create the callback.
# verify_callback = early_stopping(num_epochs_verify, stop_threshold_verify)

# # --- Print the results to verify ---
# print("--- EarlyStopping Configuration ---")
# print(f"Metric to Monitor:     {verify_callback.monitor}")
# print(f"Stopping Threshold:    {verify_callback.stopping_threshold}")
# print(f"Patience:              {verify_callback.patience}")
# print(f"Mode:                  {verify_callback.mode}")




# Configuring the trainer

def run_training(model,data_module,num_epochs,callback,progress_bar=True,dry_run=False):
    """
    Configures and runs a Lightning mixed-precision training process.

    Args:
        model (pl.LightningModule): The model to be trained.
        data_module (pl.LightningDataModule): The data module that provides the datasets.
        num_epochs (int): The maximum number of epochs for training.
        callback (pl.Callback): A callback, such as for early stopping.
        progress_bar (bool): If True, shows the training progress bar. Defaults to True.
        dry_run (bool): If True, runs a quick single batch "dry run" to test the code.
                        Defaults to False.

    Returns:
        A tuple containing:
            - pl.Trainer: The trainer instance after fitting is complete.
            - pl.LightningModule: The trained model with updated weights.
    """
    # Instantiate the PyTorch Lighting Trainer with specific configurations.
    trainer =pl.Trainer(
        # Set the maximum number of training epochs.
        max_epochs=num_epochs,
        # Automatically set the best hardware
        accelerator="auto",
        # Use a single device
        devices=1,
        # Enable 16-bit mixed-precision training to speed up computation.
        precision="16-mixed",
        # Add Callback
        callbacks=[callback],



        # Disable the default logger
        logger = False,
        # Show the trainig progress bar
        enable_progress_bar=progress_bar,
        # Disable the model summary printout.
        enable_model_summary=False,
        # Disable automatic model checkpointing.
        enable_checkpointing=False,
        # fast_dev_run flag for test runs.
        fast_dev_run=dry_run
        )

    # Run the training process.
    trainer.fit(model,data_module)

    return trainer, model



# Test 
# verify_dm = ChestXRayDataModule(data_dir,2)
# weights_path = setup_dummy_weights()
# verify_model = ChestXRAYClassifier(weights_path)
# num_epoch=1
# verify_callback = early_stopping(num_epoch,0.99)

# # Call the run_training function with dry_run=True
# print(" --------------- Verifying Training Run (Dry Run) --------------")
# run_training(verify_model,verify_dm,num_epoch,verify_callback,True)

# print("The Trainer configured and ran a single batch without errors.")
# # Clean dummy weights
# cleanup_dummy_weights()




##################################
# Prediction Setup
#################################

def load_model(weights_path,num_classes=3):
    """
    Loads the ChestXRayClassifier with trained weights.
    
    Args:
        weights_path (str): Path to the saved .pth weights file.
        num_classes (int): Number of output classes.
    
    Returns:
        model: The loaded model ready for inference.
    """
    base_weights = "./resnet18_chest_xray_classifier_weights.pth"
    setup_dummy_weights(base_weights,num_classes)

    # Instantiate the model
    model = ChestXRAYClassifier(model_weights_path=base_weights,num_classes=num_classes)

    # Load YOUR trained weights on top
    model.model.load_state_dict(
        torch.load(weights_path,map_location="cuda" if torch.cuda.is_available() else "cpu")
    )

    # Set to evaluation mode - disables dropuout, batchnorm
    model.eval()
    return model

def predict_image(model,image_path,class_names):
    """
    Predicts the class of a single chest X-ray image.
    
    Args:
        model: The loaded ChestXRayClassifier model.
        image_path (str): Path to the image file.
        class_names (list): List of class names e.g. ["NORMAL", "PNEUMONIA"]
    
    Returns:
        dict: Predicted class and confidence scores for all classes.
    """

    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    img_tensor = VAL_TRANSFORM(img).unsqueeze(0)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    img_tensor = img_tensor.to(device)

    # Run inference - no gradient
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits,dim=1) # convert logits to probabilities
        predicted_idx = torch.argmax(probabilities,dim=1).item()
        confidence = probabilities[0][predicted_idx].item()

    # build confidence score
    all_scores = {
        class_names[i]:f"{probabilities[0][i].item() * 100:.2f}%"
        for i in range(len(class_names))
        }

    return {
        "predicted_class": class_names[predicted_idx],
        "confidence": f"{confidence * 100:.2f}%",
        "all_scores": all_scores
    }

def predict_folder(model,folder_path, class_name):
    """
    Predicts the class for all images in a folder.
    
    Args:
        model: The loaded ChestXRayClassifier model.
        folder_path (str): Path to folder containing images.
        class_names (list): List of class names.
    """
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    
    print(f"/n{'Image':<30} {'Prediction':<15} {'Confidence':<12} Scores")
    print("-" * 80)
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        result = predict_image(model, image_path, class_name)
        print(
            f"{image_file:<30} "
            f"{result['predicted_class']:<15} "
            f"{result['confidence']:<12} "
            f"{result['all_scores']}"
        )




##########################################################################
#                       Training Diagnostic Assistance                   #
##########################################################################

if __name__ == "__main__":
    # # Define the file path for pre-trained model weights
    # pretrained_weights = "./resnet18_chest_xray_classifier_weights.pth"
    # setup_dummy_weights(pretrained_weights,num_classes=2)
    
    # # Epochs 
    # training_epochs = 10
    # # Validation accuracy threshold to stop training once reached
    # target_accuracy = 0.85
    
    
    # # Create the EarlyStopping callback by calling the function with the defined hyperparameters
    # early_stopping_callback = early_stopping(training_epochs,target_accuracy)
    
    # # Set the random seed to ensure the experiment is reproducible
    # pl.seed_everything(15)
    
    # # Instantiate the DataModule to handle data loading and transformations
    # dm = ChestXRayDataModule(data_dir)
    # dm.setup()
    
    # # Instantiate the LightningModule, passing in the path to the pre-pretrained_weights.
    # model = ChestXRAYClassifier(pretrained_weights,num_classes=2)
    
    # # Call the training function with all the components to start the trainig process.
    # trained_trainer, trained_model = run_training(
    #     model, dm, training_epochs,early_stopping_callback
    #     )
    
    # print("/n--- Training Complete-----")
    
    # ## Get the final metics from the trainer object
    # final_metrics = trained_trainer.callback_metrics
    
    # # Extrack the validation accuracy and convert it to a number
    # final_val_acc = final_metrics["val_acc"].item()
    
    # print(f"Final Validation Accuracy: {final_val_acc:.4f}")


    # # Save the pretrained pretrained_weights
    # weights_save_path = "./trained_chest_xray_weights.pth"
    # torch.save(trained_model.model.state_dict(),weights_save_path)
    # print(f"Weights saved to: {weights_save_path}")


    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
    # Load the model with your trained weights
    model = load_model("./trained_chest_xray_weights.pth", num_classes=2)

    # ── Predict a single image ──
    result = predict_image(
        model,
        image_path="C:/_____ DeepLearning AI/2 Deep Learning Pytorch/PyTorch Techniques and ecosystem Tools/4. Efficient Training Pipelines/Pneumonia Diag/chest_xray/test/NORMAL/IM-0003-0001.jpeg",
        class_names=CLASS_NAMES
    )
    print("/n--- Single Image Prediction ---")
    print(f"  Predicted Class : {result['predicted_class']}")
    print(f"  Confidence      : {result['confidence']}")
    print(f"  All Scores      : {result['all_scores']}")



















