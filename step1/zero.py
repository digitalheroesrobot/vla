
import torch
from torch import nn

# 1. Training examples: y = 2x + 1
x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
y = 2 * x + 1

# 2. Create a model with a learnable weight and bias
model = nn.Linear(1, 1)

# 3. Choose a loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. Train the model
for epoch in range(500):
    prediction = model(x)
    loss = loss_function(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. Test on an input not used as a training example
test_x = torch.tensor([[3.0]])

with torch.no_grad():
    test_prediction = model(test_x)

print("Prediction:", test_prediction.item())
print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())
print("Final loss:", loss.item())
