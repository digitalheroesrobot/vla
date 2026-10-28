"""
magine you have a robot moving around a 5 × 5 grid.

The robot knows its current position, and the target has a position on the same grid.

For example:
- Robot position: (1, 2)
- Target position: (4, 3)
Your goal is to train a neural network to determine how the robot should move.

The expected answer is:
- Move right because the target is to the right.
- Move up because the target is above.
The final movement is right + up.
"""

import torch
from torch import nn

# Action mappings
x_labels = ["left", "stay", "right"]
y_labels = ["down", "stay", "up"]

examples = []

# Generate dataset
for robot_x in range(5):
    for robot_y in range(5):
        for target_x in range(5):
            for target_y in range(5):

                # X-axis action: 0=left, 1=stay, 2=right
                if target_x < robot_x:
                    action_x = 0
                elif target_x > robot_x:
                    action_x = 2
                else:
                    action_x = 1

                # Y-axis action: 0=down, 1=stay, 2=up
                if target_y < robot_y:
                    action_y = 0
                elif target_y > robot_y:
                    action_y = 2
                else:
                    action_y = 1

                examples.append(
                    ([robot_x, robot_y, target_x, target_y],
                     [action_x, action_y])
                )

# Convert data to tensors
train_x = torch.tensor(
    [item[0] for item in examples],
    dtype=torch.float32
)

train_y = torch.tensor(
    [item[1] for item in examples],
    dtype=torch.long
)

print("Input shape:", train_x.shape)    # (625, 4)
print("Target shape:", train_y.shape)   # (625, 2)


# Model with two independent output heads
class RobotModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU()
        )

        self.x_head = nn.Linear(16, 3)
        self.y_head = nn.Linear(16, 3)

    def forward(self, x):
        features = self.shared(x)

        x_scores = self.x_head(features)
        y_scores = self.y_head(features)

        return x_scores, y_scores


model = RobotModel()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# Training
for epoch in range(500):
    # Forward pass
    x_scores, y_scores = model(train_x)

    # Calculate loss for both axes
    loss_x = loss(x_scores, train_y[:, 0])
    loss_y = loss(y_scores, train_y[:, 1])

    total_loss = loss_x + loss_y

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {total_loss.item():.4f}"
        )


# Test the model
test = [1, 2, 4, 3]
# Inference Mode
model.eval()

with torch.no_grad():
    test_tensor = torch.tensor(
        [test],
        dtype=torch.float32
    )

    x_scores, y_scores = model(test_tensor)

    # Get predicted class indices
    x_action = x_scores.argmax(dim=1).item()
    y_action = y_scores.argmax(dim=1).item()

    print("\nRobot position:", test[:2])
    print("Target position:", test[2:])
    print("Predicted X action:", x_labels[x_action])
    print("Predicted Y action:", y_labels[y_action])

    print(
        "Combined action:",
        x_labels[x_action],
        "+",
        y_labels[y_action]
    )