import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from torch import nn, inference_mode

# Create data
X, y = make_moons(n_samples=1000,
                  noise=0.2,
                  shuffle=True,
                  random_state=42)

# Visualise
# sns.set_style("whitegrid")
# sns.scatterplot(x=X[:,0],
#                 y=X[:,1], 
#                 hue=y,
#                 palette="Dark2",)
# plt.show()


# Set device
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using Apple Silicon GPU (MPS)')
else:
    device = torch.device('cpu')


# Move the data to device
X = torch.from_numpy(X).type(dtype = torch.float).to(device)
y = torch.from_numpy(y).type(dtype = torch.float).to(device)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)


# Build the model
moonmodel = nn.Sequential(nn.Linear(in_features=2, out_features=32),
                          nn.ReLU(),
                          nn.Linear(in_features=32, out_features=32),
                          nn.ReLU(),
                          nn.Linear(in_features=32, out_features=1)).to(device)

# Define the loss function
loss_fn = nn.BCEWithLogitsLoss()
optimiser = torch.optim.SGD(params=moonmodel.parameters(), lr=0.02)

# Fit the model
epochs=5000
epoch_count=[]
trloss_count=[]
ttloss_count=[]
for epoch in range (epochs):
    # Training mode
    moonmodel.train()
    # Forward pass
    y_logits = moonmodel(X_train).squeeze()
    y_probs = torch.sigmoid(y_logits)
    y_pred = torch.round(y_probs)
    # Calculate loss
    loss = loss_fn(y_logits, y_train)
    trloss_count.append(loss)
    epoch_count.append(epoch)
    # Optimiser zero grad
    optimiser.zero_grad()
    # Loss backward
    loss.backward()
    # Optimiser step
    optimiser.step()

    # Testing
    moonmodel.eval()
    with inference_mode():
        test_logits = moonmodel(X_test).squeeze()
        test_probs = torch.sigmoid(test_logits)
        test_pred = torch.round(test_probs)

        test_loss = loss_fn(test_logits, y_test)
        ttloss_count.append(test_loss)

    # Print the model performance over epochs
    if epoch%1000==0:
        print(f"Epoch: {epoch} | Training Loss: {loss} | Test Loss: {test_loss}")


# # Visualise the decision boundry
# def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
#     """Plots decision boundaries of model predicting on X in comparison to y.

#     Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
#     """
#     # Put everything to CPU (works better with NumPy + Matplotlib)
#     model.to("cpu")
#     X, y = X.to("cpu"), y.to("cpu")

#     # Setup prediction boundaries and grid
#     x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
#     y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

#     # Make features
#     X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

#     # Make predictions
#     model.eval()
#     with torch.inference_mode():
#         y_logits = model(X_to_pred_on)

#     # Test for multi-class or binary and adjust logits to prediction labels
#     if len(torch.unique(y)) > 2:
#         y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
#     else:
#         y_pred = torch.round(torch.sigmoid(y_logits))  # binary

#     # Reshape preds and plot
#     y_pred = y_pred.reshape(xx.shape).detach().numpy()
#     plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
#     plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(moonmodel, X_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(moonmodel, X_test, y_test)
# plt.show()