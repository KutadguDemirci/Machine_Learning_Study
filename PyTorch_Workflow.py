import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
# Define the worflow

# 1- Data (preparation, loading, etc.)
# 2- Build model
# 3- Fit the model to the training data
# 4- Make predictions and evaluate the model
# 5- Save and load the model (interference)
# 6- Putting it all together

########### DATA PREPARATION AND LOADING ############


# Create linear data with known parameters
weight = 0.7
bias = 0.3

start = 0
stop = 1
step = 0.02
X = torch.arange(start, stop, step).unsqueeze(1)
y = weight + bias*X

# Create training and test sets
train_split = int(0.8 * len(X))
X_train, y_train = X[: train_split], y[: train_split]
X_test, y_test = X[train_split :], y[train_split :]
# print(len(X_train), len(y_train))
# print(len(X_test), len(y_test))

# Build the model

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(1,
                                       requires_grad=True,
                                       dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1,
                                            requires_grad = True,
                                            dtype=torch.float))
    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    
# Checking the contents of the model
torch.manual_seed(42)

model0 = LinearRegressionModel()

# print(list(model0.parameters())) # Lists the parameters of the model
# print(model0.state_dict()) # Lists the named parameters of the model






# Make predictions with the model
with torch.inference_mode():
    y_pred = model0(X_test)
# print(y_pred, y_test)



# Define a loss function
loss_fn = torch.nn.L1Loss()

# Define an optimizer
optimiser = torch.optim.SGD(model0.parameters(), lr=0.01)

# an epoch is one complete pass through the dataset
epochs = 1500

# Step 1, Loop through the data
for epoch in range(epochs):

    # Set the model to training mode
    model0.train() # Train mode in PyTorch sets all the parameters to be trainable like requires_grad
    # Step 2, Forward pass
    y_pred = model0(X_train)
    # Step 3, Calculate the loss
    loss = loss_fn(y_pred, y_train)
    # Step 4, Optimiser zero grad
    optimiser.zero_grad()
    # Step 5, Backpropagation
    loss.backward()
    # Step 6, Optimiser step (perform gradient descent)
    optimiser.step()
    
    ## Tesing
    model0.eval() # Set the model to evaluation mode
    with torch.inference_mode():
        test_pred = model0(X_test)
        test_loss = loss_fn(test_pred, y_test)
print(f"Epoch: {epoch+1} | Test Loss: {test_loss:.5f} | Loss: {loss:.5f} | Weight: {model0.weights.item():.5f} | Bias: {model0.bias.item():.5f}")


