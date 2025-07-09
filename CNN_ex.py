import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torchmetrics.classification import Accuracy

# Import progress bar
from tqdm.auto import tqdm


# Creating a function to track time
from timeit import default_timer as timer
def print_train_time(start:float,
                     end:float,
                     device:torch.device=None):
    total_time = end - start
    print(f"Train time on {device} is {total_time:.2f} seconds.")
    return total_time


#Setup device agnostic code
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# Setup training data
train_data = datasets.FashionMNIST(root = "data", # Where the data will be downloaded.
                                   train=True,    # Do you want the training set or not.
                                   download = True,# Do you want to downoad or not.
                                   transform=ToTensor(), # How do you want to transform the data.
                                   target_transform=None) # How do you want to transform the labels/target.

test_data = datasets.FashionMNIST(root=("data"),
                                  train=False,
                                  download="True",
                                  transform=ToTensor(),
                                  target_transform=None)
class_names = train_data.classes




# # Visualise the data
# import matplotlib.pyplot as plt
# image, label = train_data[0]
# print(f"Image shape : {image.shape}")
# plt.imshow(image.squeeze(), cmap="gray")
# plt.show()

# Turn the data into batches of data
train_data_loader = DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True)

test_data_loader = DataLoader(dataset= test_data,
                              batch_size=64)

train_features_batch, train_labels_batch = next(iter(train_data_loader))
# print(train_features_batch.shape, train_labels_batch.shape) 
#torch.Size([32, 1, 28, 28]) torch.Size([32]) batch size of 32, 1 colour label, 28 width and 28 height







# # Building a baseline model
# class FashionMNISTModel(nn.Module):
#     def __init__(self,
#                  input_shape: int,
#                  hidden_units: int,
#                  output_shape: int):
#         super().__init__()
#         self.layer_stack = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=input_shape, out_features=hidden_units),
#             nn.Linear(in_features=hidden_units, out_features=output_shape)
#         )
#     def forward(self, x):
#         return self.layer_stack(x)

# torch.manual_seed(42)

# model0 = FashionMNISTModel(input_shape = 28*28,
#                            hidden_units=10,
#                            output_shape=len(class_names))

# # Steup loss and optimiser
# loss_fn = nn.CrossEntropyLoss()
# optimiser = torch.optim.SGD(params=model0.parameters(), lr=0.1)





# #Set seet and start the timer
# torch.manual_seed(42)
# train_time_start_on_cpu = timer()

# #Set the number of epochs
# epochs = 3

# #Create training and testing loop
# for epoch in tqdm(range(epochs)):
#     print(f"Epoch: {epoch}")

#     #Training
#     train_loss = 0

#     #Add a loop to loop through the training batches.
#     for batch, (X, y) in enumerate(train_data_loader):
#         model0.train()

#         #Forward pass
#         y_pred = model0(X)

#         #Calculate the loss
#         loss = loss_fn(y_pred, y)
#         train_loss += loss

#         #Optmiser zero grad
#         optimiser.zero_grad()

#         #Loss backward
#         loss.backward()

#         #Optimiser step
#         optimiser.step()

#         if batch%400==0:
#             print(f"Looked at {batch * len(X)}/{len(train_data_loader)} samples.")

#     test_loss = 0
#     model0.eval()

# with torch.inference_mode():
#     for X_test, y_test in test_data_loader:
#         test_pred = model0(X_test)
#         loss = loss_fn(test_pred, y_test)  # compute test loss for the batch
#         test_loss += loss.item()           # add the scalar loss

#     test_loss /= len(test_data_loader)     # average over number of batches

# print(f"\nTrain loss = {train_loss:.4f } | Test loss = {test_loss:.4f}")
# train_time_end_on_cpu = timer()
# total_train_time_model0 = print_train_time(start=train_time_start_on_cpu,
#                                            end = train_time_end_on_cpu,
#                                            device = str(next(model0.parameters()).device))






## Baseline model 2

# class FashionMNISTModelV1(nn.Module):
#     def __init__(self,
#                  input_shape:int,
#                  output_shape:int,
#                  hidden_units:int):
#         super().__init__()
#         self.layer_stack = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features = input_shape, out_features = hidden_units),
#             nn.ReLU(),
#             nn.Linear(in_features = hidden_units, out_features = hidden_units),
#             nn.ReLU(),
#             nn.Linear(in_features = hidden_units, out_features = output_shape),
#             nn.ReLU())
#     def forward(self, x: torch.Tensor):
#         return self.layer_stack(x)

# torch.manual_seed(42)

# model1 = FashionMNISTModelV1(input_shape=784,
#                              hidden_units=32,
#                              output_shape=len(class_names)).to(device)

# print(next(model1.parameters()).device)

# loss_fn = nn.CrossEntropyLoss()
# optimiser = torch.optim.SGD(params=model1.parameters(),
#                             lr=0.1)

# epochs = 3

# for epoch in range(epochs):
#     print(f"Epoch: {epoch}")
#     train_loss = 0

#     model1.train()
#     train_start_time = timer()
#     for batch, (X, y) in enumerate(train_data_loader):
#         X = X.to(device)
#         y = y.to(device)
        
#         #Forward pass
#         y_logits = model1(X)
#         #Calculate the loss
#         loss = loss_fn(y_logits, y)
#         train_loss+=loss
#         #Optimser zero grad
#         optimiser.zero_grad()
#         #Loss backward
#         loss.backward()
#         #Optimiser step
#         optimiser.step()

#         if batch%400==0:
#             print(f"Looked at {batch}/{len(train_data_loader)}")
#     train_end_time = timer()
# with torch.inference_mode():
#     model1.eval()
#     test_loss = 0
#     for X_test, y_test in test_data_loader:
#         X_test, y_test = X_test.to(device), y_test.to(device)
#         y_tlogits = model1(X_test)
#         loss_t = loss_fn(y_tlogits, y_test)
#         test_loss += loss_t.item()

# test_loss /= len(test_data_loader)
# train_loss /= len(train_data_loader)

# print(f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}")
# total_train_time_model0 = print_train_time(start=train_start_time,
#                                            end = train_end_time,
#                                            device = str(next(model1.parameters()).device))

class FashionMNISTCNN(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))        
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=hidden_units*3*3,
                      out_features=output_shape)
        )
    def forward(self, x):
        x=self.conv_block1(x)
        x=self.conv_block2(x)
        x=self.conv_block3(x)
        x=self.classifier(x)
        return x
    
torch.manual_seed(42)
model2 = FashionMNISTCNN(input_shape=1,
                         hidden_units=128,
                         output_shape=len(class_names)).to(device)


# # Find the shape to pass into the classifier
# rand_image_tensor = torch.randn(size=(1, 1, 28, 28)).to(device)
# model2(rand_image_tensor)





# Define the number of epochs
epochs = 7

#Define the loss function
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
#Define the optimiser
optimiser = torch.optim.Adam(params=model2.parameters(),
                            lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min',
                                                       patience=3, factor=0.5)


#Accuracy tracker
test_acc_fn = Accuracy(task="multiclass", num_classes=len(class_names)).to(device)

for epoch in range(epochs):

    train_loss = 0
    #training loop
    for batch, (X,y) in enumerate(train_data_loader):
        X, y = X.to(device), y.to(device)
        model2.train()

        #forward pass
        y_pred = model2(X)
        #calculate loss
        tr_loss = loss_fn(y_pred, y)
        train_loss+=tr_loss.item()
        #optimiser zero grad
        optimiser.zero_grad()
        #loss backward
        tr_loss.backward()
        #optimiser step
        optimiser.step()

        # if batch%400==0:
        #     print[f"Looking at {epoch}/{len(train_data_loader)}"]
    model2.eval()
    with torch.inference_mode():
        test_loss = 0
        test_correct = 0
        test_total = 0
        for (X_t, y_t) in test_data_loader:
            X_t, y_t = X_t.to(device), y_t.to(device)

            y_tpred = model2(X_t)
            ts_loss = loss_fn(y_tpred, y_t)
            test_loss+=ts_loss.item()
            test_acc_fn.update(y_tpred, y_t)
    test_accuracy = test_acc_fn.compute().item()
    test_acc_fn.reset()
    scheduler.step(test_loss/len(test_data_loader))


    print(f"For epoch {epoch}: Train Loss = {(train_loss/len(train_data_loader)):.4f} | Test Loss = {(test_loss/len(test_data_loader)):.4f} | Test Acc = {test_accuracy * 100:.2f}")


import matplotlib.pyplot as plt

# Put model in eval mode
model2.eval()

# Containers for incorrect predictions
incorrect_images = []
incorrect_true_labels = []
incorrect_pred_labels = []

# Disable gradient tracking
with torch.inference_mode():
    for X_batch, y_batch in test_data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model2(X_batch)

        # Get predicted class indices
        y_pred_labels = y_pred.argmax(dim=1)

        # Find indices where prediction != truth
        incorrect = y_pred_labels != y_batch

        # Store incorrect predictions
        incorrect_images.extend(X_batch[incorrect].cpu())
        incorrect_true_labels.extend(y_batch[incorrect].cpu())
        incorrect_pred_labels.extend(y_pred_labels[incorrect].cpu())

print(f"Number of incorrect predictions: {len(incorrect_images)}")

# How many to show
n_to_show = 12
plt.figure(figsize=(12, 10))

for i in range(n_to_show):
    image = incorrect_images[i].squeeze()  # Remove channel dim
    true_label = class_names[incorrect_true_labels[i]]
    pred_label = class_names[incorrect_pred_labels[i]]

    plt.subplot(3, 4, i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"True: {true_label}\nPred: {pred_label}", color="red")
    plt.axis("off")

plt.tight_layout()


# Import tqdm for progress bar
from tqdm.auto import tqdm

# 1. Make predictions with trained model
y_preds = []
model2.eval()
with torch.inference_mode():
  for X, y in tqdm(test_data_loader, desc="Making predictions"):
    # Send data and targets to target device
    X, y = X.to(device), y.to(device)
    # Do the forward pass
    y_logit = model2(X)
    # Turn predictions from logits -> prediction probabilities -> predictions labels
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
    # Put predictions on CPU for evaluation
    y_preds.append(y_pred.cpu())
# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)

import mlxtend
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# 2. Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
    class_names=class_names, # turn the row and column labels into class names
    figsize=(10, 7)
);
plt.show()