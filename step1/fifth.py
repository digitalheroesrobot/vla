"""
fifth.py

Lesson 5: Training vs Testing

Learn the difference between training data and testing data.

A robot receives:
    robot_x, robot_y, target_x, target_y

It predicts:
    0 = Move left
    1 = Stay
    2 = Move right

The dataset is divided into:
    80% Training data
    20% Testing data

The model only learns from training data.
Testing data is used to evaluate whether the model
can generalize to unseen examples.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import random

# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

torch.manual_seed(42)
random.seed(42)

# --------------------------------------------------
# 2. Generate robot navigation dataset
# --------------------------------------------------

def generate_dataset(num_samples=1000):

    X = []
    y = []

    for _ in range(num_samples):

        # Random robot position
        robot_x = random.randint(0, 10)
        robot_y = random.randint(0, 10)

        # Random target position
        target_x = random.randint(0, 10)
        target_y = random.randint(0, 10)

        # Calculate horizontal distance
        distance_x = target_x - robot_x

        # Determine the correct action
        if distance_x < 0:
            action = 0       # Move left

        elif distance_x > 0:
            action = 2       # Move right

        else:
            action = 1       # Stay

        X.append([
            robot_x,
            robot_y,
            target_x,
            target_y
        ])

        y.append(action)

    return torch.tensor(X, dtype=torch.float32), \
           torch.tensor(y, dtype=torch.long)


# Generate 1000 examples
X, y = generate_dataset(1000)

print("Dataset generated!")
print("Input shape:", X.shape)
print("Label shape:", y.shape)


# --------------------------------------------------
# 3. Split training and testing data
# --------------------------------------------------

dataset_size = len(X)

# Generate random indices
indices = torch.randperm(dataset_size)

# 80% training, 20% testing
train_size = int(0.8 * dataset_size)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Create training dataset
X_train = X[train_indices]
y_train = y[train_indices]

# Create testing dataset
X_test = X[test_indices]
y_test = y[test_indices]

print("\nDataset split:")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 4. Define neural network
# --------------------------------------------------

class RobotPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(4, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 3)

        )

    def forward(self, x):

        return self.network(x)


model = RobotPolicy()


# --------------------------------------------------
# 5. Define loss and optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# 6. Train the model
# --------------------------------------------------

epochs = 200

print("\nStarting training...\n")

for epoch in range(epochs):

    # Training mode
    model.train()

    # Forward pass
    predictions = model(X_train)

    # Calculate loss
    loss = criterion(predictions, y_train)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print progress
    if (epoch + 1) % 20 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )


# --------------------------------------------------
# 7. Evaluate training accuracy
# --------------------------------------------------

model.eval()

with torch.no_grad():

    train_predictions = model(X_train)

    train_predicted_classes = torch.argmax(
        train_predictions,
        dim=1
    )

    train_accuracy = (
        train_predicted_classes == y_train
    ).float().mean()


# --------------------------------------------------
# 8. Evaluate testing accuracy
# --------------------------------------------------

with torch.no_grad():

    test_predictions = model(X_test)

    test_predicted_classes = torch.argmax(
        test_predictions,
        dim=1
    )

    test_accuracy = (
        test_predicted_classes == y_test
    ).float().mean()


# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\n========== RESULTS ==========")

print(
    f"Training Accuracy: "
    f"{train_accuracy.item() * 100:.2f}%"
)

print(
    f"Testing Accuracy:  "
    f"{test_accuracy.item() * 100:.2f}%"
)


# --------------------------------------------------
# 10. Test unseen robot situations
# --------------------------------------------------

print("\n========== UNSEEN EXAMPLES ==========")

action_names = {
    0: "LEFT",
    1: "STAY",
    2: "RIGHT"
}

# New situations that were not in the dataset
new_examples = torch.tensor([

    [2, 3, 8, 3],
    [8, 4, 2, 4],
    [5, 5, 5, 5],
    [1, 2, 4, 2],
    [9, 8, 3, 8]

], dtype=torch.float32)

model.eval()

with torch.no_grad():

    outputs = model(new_examples)

    predicted_actions = torch.argmax(
        outputs,
        dim=1
    )

for i, action in enumerate(predicted_actions):

    robot_x, robot_y, target_x, target_y = \
        new_examples[i].tolist()

    print(
        f"Robot: ({robot_x:.0f}, {robot_y:.0f}) | "
        f"Target: ({target_x:.0f}, {target_y:.0f}) | "
        f"Predicted action: {action_names[action.item()]}"
    )

print("\nLesson completed!")