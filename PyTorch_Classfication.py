import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch import inference_mode, nn
import numpy as np
import pandas as pd

from sklearn.datasets import make_circles, make_blobs
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassRecall



# Create a dataset

s_samples = 1000
X, y = make_circles(n_samples = s_samples,
                    noise = 0.03, 
                    random_state= 42)


circles = pd.DataFrame(X, columns =['x1', 'x2'])
circles['label'] = y

# sns.scatterplot(data=circles, x='x1', y='x2', hue='label')
# plt.show()

# Turn the data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Set device to Apple Silicon GPU (MPS) if available, else CPU
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using Apple Silicon GPU (MPS)')
else:
    device = torch.device('cpu')
    print('Using CPU')

# Move data to device
X = X.to(device)
y = y.to(device)



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 42)

# print(f"Feature training device is {X_train.device}, Label training device is {y_train.device}.")
# print(f"Feature test device is {X_test.device}, Label test device is {y_test.device}.")


# Build the model

# The model can be built by manually putting layers inside of each other
# class CircleModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Create 2 nn layers capable of handling the shapes of the data.
#         self.layer1 = nn.Linear(in_features=2, out_features=5) # Takes in 2 features and upscales them to 5 features.
#         self.layer2 = nn.Linear(in_features=5, out_features=1) # Takes in the 5 features and outputs a single feature (same shape as "y").

#     def forward(self, x):
#         return self.layer2(self.layer1(x)) # x -> layer1 -> layer2 -> output
# model = CircleModel().to(device)




# A model with multiple hidden layers can be built using nn.Sequantial
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=1)).to(device)

# print(next(model.parameters()).device)
# print(model.parameters)

# y_pred = model(X_train)
# print(torch.round(y_pred[:10]))



# Setup a loss function and an optimiser. 
# For binary classification, binary cross-entropy, BCE loss function can be used.
# For binary classification, SGE or Adam can be used as optimiser.
loss_fn = nn.BCEWithLogitsLoss()
optimiser = torch.optim.SGD(params=model.parameters(), 
                            lr=0.01)
# Create an accuracy function
def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

# print(torch.round(torch.sigmoid(model(X_train[:5]))))


# Train the model

epochs = 10000
for epoch in range (epochs):
    # Training mode
    model.train()

    # Forward Pass
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train)
    accur = accuracy(y_train, y_pred)

    # Oprimiser zero grad
    optimiser.zero_grad()

    # Loss Backward, backpropagation
    loss.backward()

    # Optimiser step, gradient descent
    optimiser.step()

    # Testing
    model.eval()
    with inference_mode():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_accuracy = accuracy(test_pred, y_test)
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {test_loss:.5f} | Test Accuracy: {test_accuracy}")

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()





################# MULICLASS CLASSIFICATION ##################


# # Define the device
# if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     device = torch.device('mps')
#     print('Using Apple Silicon GPU (MPS)')
# else:
#     device = torch.device('cpu')
#     print('Using CPU')

# # Create data
# sample_size = 1000
# classes = 4
# num_features = 2
# random_st = 42

# X_blob, y_blob = make_blobs(n_samples = sample_size,
#                             n_features = num_features,
#                             centers= classes,
#                             cluster_std = 1.5,
#                             random_state = random_st)

# X_blob = torch.from_numpy(X_blob).type(dtype = torch.float).to(device)
# y_blob = torch.from_numpy(y_blob).type(dtype = torch.float).to(device)

# X_train, X_test, y_train, y_test = train_test_split(X_blob, y_blob, test_size=0.2,
#                                                     random_state=random_st)



# # Build the model
# model = nn.Sequential(nn.Linear(in_features=2, out_features=32),
#                       #nn.ReLU(),
#                       nn.Linear(in_features=32, out_features=32),
#                       #nn.ReLU(),
#                       nn.Linear(in_features=32, out_features=4),).to(device)

# # Define the loss and optimiser
# loss_fn = nn.CrossEntropyLoss()
# optimiser = torch.optim.SGD(params = model.parameters(), lr= 0.01)

# # Instantiate torchmetrics objects
# recall_metric = MulticlassRecall(num_classes=4, average=None).to(device)

# # Define the training loop
# epochs = 1000

# for epoch in range (epochs):
#     # Initiate the training mode
#     model.train()
#     # Forward pass
#     y_logits = model(X_train)
#     y_probs = torch.softmax(y_logits, dim = 1)
#     y_pred = torch.argmax(y_probs, dim=1)
#     # Calculate the loss
#     loss = loss_fn(y_logits, y_train)
#     # Optimiser zero grad
#     optimiser.zero_grad()
#     # Backpropagation
#     loss.backward()
#     # Gradient Descent
#     optimiser.step()
#     # Testing
#     model.eval()
#     with inference_mode():
#         test_logits = model(X_test)
#         test_probs = torch.softmax(test_logits, dim=1)
#         test_pred = torch.argmax(test_probs, dim=1)

#         test_loss = loss_fn(test_logits, y_test)
#     if epoch % 100 == 0:
#         print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {test_loss:.5f} ")

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model, X_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model, X_test, y_test)
# plt.show()

# recall_metric.update(test_pred, y_test.long())

# # pull out tensor of recalls
# per_class = recall_metric.compute()  
# print("Recall per class:", per_class.tolist())

# # if you want a single macro number:
# recall_macro = MulticlassRecall(num_classes=4, average="macro").to(device)
# recall_macro.update(test_pred, y_test.long())
# print("Macro Recall:", recall_macro.compute().item())