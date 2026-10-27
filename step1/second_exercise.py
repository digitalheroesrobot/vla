"""
input -1 -> target -1 (left)
        0 -> 0 (stay centered)
        1 -> 1 (right)
"""


import torch
from torch import nn

input = torch.tensor([[-1.0], [0.0], [1.0]])
target = torch.tensor([[-1.0], [0.0], [1.0]])

model = nn.Linear(1, 1) # Creates a model with one input and one output, with a learnable weight and bias
loss = nn.MSELoss() # Measures the mean squared error between the predicted and targets.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01

for epoch in range(500): # Train for 500 epochs
    prediction = model(input)  
    loss_value = loss(prediction, target)
    optimizer.zero_grad() # Zero the gradients before running the backward pass.
    loss_value.backward()    # Compute the gradients of the loss with respect to the model parameters.
    optimizer.step() # Update the model parameters using the computed gradients.

input_test = torch.tensor([[0.5]])
with torch.no_grad():
    test_prediction = model(input_test)

print("Prediction:", test_prediction.item())    
print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())
print("Final loss:", loss_value.item())