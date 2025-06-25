import torch
from torch import inference_mode, nn
import matplotlib.pyplot as plt
import seaborn as sns
# Define the worflow

# 1- Data (preparation, loading, etc.)
# 2- Build model
# 3- Fit the model to the training data
# 4- Make predictions and evaluate the model
# 5- Save and load the model (interference)
# 6- Putting it all together




# Data preperation

# Create linear data with known parameters
# weight = 0.7
# bias = 0.3

# start = 0
# stop = 1
# step = 0.02
# X = torch.arange(start, stop, step).unsqueeze(1)
# y = weight + bias*X

# # Create training and test sets
# train_split = int(0.8 * len(X))
# X_train, y_train = X[: train_split], y[: train_split]
# X_test, y_test = X[train_split :], y[train_split :]
# # print(len(X_train), len(y_train))
# # print(len(X_test), len(y_test))

# # Build the model

# class LinearRegressionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weights = nn.Parameter(torch.rand(1,
#                                        requires_grad=True,
#                                        dtype=torch.float))
#         self.bias = nn.Parameter(torch.rand(1,
#                                             requires_grad = True,
#                                             dtype=torch.float))
#     # Forward method to define the computation in the model
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.weights * x + self.bias
    
# # Checking the contents of the model
# torch.manual_seed(42)

# model0 = LinearRegressionModel()

# # print(list(model0.parameters())) # Lists the parameters of the model
# # print(model0.state_dict()) # Lists the named parameters of the model






# # Make predictions with the model
# with torch.inference_mode():
#     y_pred = model0(X_test)
# # print(y_pred, y_test)



# # Define a loss function
# loss_fn = torch.nn.L1Loss()

# # Define an optimizer
# optimiser = torch.optim.SGD(model0.parameters(), lr=0.01)

# # an epoch is one complete pass through the dataset
# epochs = 1500

# # Set device to GPU if available, else CPU (supports CUDA and Apple Silicon MPS)
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print('Using CUDA GPU')
# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     device = torch.device('mps')
#     print('Using Apple Silicon GPU (MPS)')
# else:
#     device = torch.device('cpu')
#     print('Using CPU')

# # Move data and model to the selected device
# X = X.to(device)
# y = y.to(device)
# X_train = X_train.to(device)
# y_train = y_train.to(device)
# X_test = X_test.to(device)
# y_test = y_test.to(device)
# model0 = model0.to(device)

# # Step 1, Loop through the data
# for epoch in range(epochs):

#     # Set the model to training mode
#     model0.train() # Train mode in PyTorch sets all the parameters to be trainable like requires_grad
#     # Step 2, Forward pass
#     y_pred = model0(X_train)
#     # Step 3, Calculate the loss
#     loss = loss_fn(y_pred, y_train)
#     # Step 4, Optimiser zero grad
#     optimiser.zero_grad()
#     # Step 5, Backpropagation
#     loss.backward()
#     # Step 6, Optimiser step (perform gradient descent)
#     optimiser.step()
    
#     ## Tesing
#     model0.eval() # Set the model to evaluation mode
#     with torch.inference_mode():
#         test_pred = model0(X_test)
#         test_loss = loss_fn(test_pred, y_test)
# print(f"Epoch: {epoch+1} | Test Loss: {test_loss:.5f} | Loss: {loss:.5f} | Weight: {model0.weights.item():.5f} | Bias: {model0.bias.item():.5f}")







# ######### blind coding the same basic example #########

# Data preparation
weight = 0.7 
bias = 0.3

start = 0
stop = 1
step = 0.02

torch.manual_seed(42)
X = torch.arange(start, stop, step).unsqueeze(1)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[: train_split], y[: train_split]
X_test, y_test = X[train_split :], y[train_split :]


# Set device to GPU if available, else CPU (supports CUDA and Apple Silicon MPS)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using CUDA GPU')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using Apple Silicon GPU (MPS)')
else:
    device = torch.device('cpu')
    print('Using CPU')

# Move data to device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# # Build the model
# class LinearRegressionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weight = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
#         self.bias = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.weight * x + self.bias

# # Initiate the model and move to device
# model1 = LinearRegressionModel().to(device)

# loss_fn = torch.nn.L1Loss()
# optimiser = torch.optim.SGD(params=model1.parameters(), lr=0.001)

# epochs = 10000

# for epoch in range(epochs):

#     model1.train()

#     y_pred = model1(X_train)

#     loss = loss_fn(y_pred, y_train)
#     optimiser.zero_grad()

#     loss.backward()

#     optimiser.step()


# # Testing
# with torch.inference_mode():
#     model1.eval()
#     test_pred = model1(X_test)
#     test_loss = loss_fn(test_pred, y_test)

# print(f"Epoch: {epoch+1} | Test Loss: {test_loss:.5f} | Loss: {loss:.5f} | Weight: {model1.weight.item():.5f} | Bias: {model1.bias.item():.5f}")


# fig, ax = plt.subplots()
# sns.scatterplot(x=X_train.cpu().squeeze(), y=y_train.cpu().squeeze(), label='Train Data', ax=ax)
# sns.scatterplot(x=X_test.cpu().squeeze(), y=y_test.cpu().squeeze(), label='Test Data', ax=ax)
# sns.scatterplot(x=X_test.cpu().squeeze(), y=model1(X_test).detach().cpu().squeeze(), label='Predictions', color='red', ax=ax)
# plt.legend()
# plt.show()



################ NEW EXAMPLE ################

# Create a new model
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear to create a linear layer
        self.linear_layer = nn.Linear(in_features = 1,
                                       out_features = 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
torch.manual_seed(42)
model2 = LinearRegressionModelV2().to(device)


loss_fn = torch.nn.L1Loss()
optimiser = torch.optim.SGD(params = model2.parameters(), lr = 0.001)

torch.manual_seed(42)
epochs = 3000

loss_count = []
epoch_count = []

for epoch in range (epochs):
    model2.train()

    y_pred = model2(X_train)
    loss = loss_fn(y_pred, y_train)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    model2.eval()
    with inference_mode():
        test_pred = model2(X_test)
        test_loss = loss_fn(test_pred, y_test)
    
    # Store the loss and epoch count for plotting
    loss_count.append([loss.item(), test_loss.item()])
    epoch_count.append(epoch)

fig, ax = plt.subplots()
sns.lineplot(x=epoch_count, y=[x[0] for x in loss_count], label='Train Loss', ax=ax)
sns.lineplot(x=epoch_count, y=[x[1] for x in loss_count], label='Test Loss', ax=ax)
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()

print(f"Final Train Loss: {loss.item():.5f} | Final Test Loss: {test_loss.item():.5f} | Weight: {model2.linear_layer.weight.item():.5f} | Bias: {model2.linear_layer.bias.item():.5f}")