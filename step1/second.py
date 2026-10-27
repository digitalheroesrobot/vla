"""
Teach a tiny model to learn a relationshp from examples using PyTorch.
- Pytorch is a Python library for building and training neural networks.

"""


import torch
from torch import nn

# 1. Training examples: y = 2x + 1
x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
y = 2 * x + 1

# 2. Create a model with a learnable weight and bias
model = nn.Linear(1, 1) # Creates a model with one input and one output, with a learnable weight and bias

# 3. Choose a loss function and optimizer
loss_function = nn.MSELoss() # Measures the mean squared error between the predicted and targets.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01

# 4. Train the model
for epoch in range(500): # Train for 500 epochs
    prediction = model(x)  
    loss = loss_function(prediction, y)
    """
    optimizer.zero_grad() is a fundamental call in PyTorch that clears the gradients of all parameters managed by an optimizer, 
    preventing unwanted accumulation between training iterations.
    """
    optimizer.zero_grad() # Zero the gradients before running the backward pass.
    """
    loss.backward() does not reduce the loss and does not control or update the gradients. 
    It only computes the gradients of the loss with respect to each model parameter.
    """
    loss.backward()    # Compute the gradients of the loss with respect to the model parameters.
    optimizer.step() # Update the model parameters using the computed gradients.

# 5. Test on an input not used as a training example
test_x = torch.tensor([[3.0]])

with torch.no_grad():
    test_prediction = model(test_x)

print("Prediction:", test_prediction.item())
print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())
print("Final loss:", loss.item())
