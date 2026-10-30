
"""
STEP 1 - LESSON 9
Classification versus Regression

Classification:
    Predict a discrete robot action:
    LEFT, RIGHT, or FORWARD.

Regression:
    Predict continuous robot commands:
    vx, vy, and a normalized joint position.

Both models receive the same input:
    [robot_x, robot_y, target_x, target_y]

Run from the repository root:
    python step1/sixth.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader, random_split


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

torch.manual_seed(42)
np.random.seed(42)

NUM_SAMPLES = 3000
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.001

ACTION_NAMES = {
    0: "LEFT",
    1: "RIGHT",
    2: "FORWARD",
}

# Use the same device for both models.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# --------------------------------------------------
# 2. Generate a synthetic robot dataset
# --------------------------------------------------

def generate_dataset(num_samples):
    """
    Each input contains:
        robot_x, robot_y, target_x, target_y

    Classification label:
        0 = LEFT
        1 = RIGHT
        2 = FORWARD

    Regression target:
        0 = desired x velocity
        1 = desired y velocity
        2 = normalized joint position
    """

    robot_x = np.random.uniform(0, 10, num_samples)
    robot_y = np.random.uniform(0, 10, num_samples)

    target_x = np.random.uniform(0, 10, num_samples)
    target_y = np.random.uniform(0, 10, num_samples)

    dx = target_x - robot_x
    dy = target_y - robot_y

    # ----------------------------------------------
    # Classification labels
    # ----------------------------------------------

    actions = np.zeros(num_samples, dtype=np.int64)

    actions[dx > 0.5] = 1       # RIGHT
    actions[dx <= 0.5] = 2      # FORWARD
    actions[dx < -0.5] = 0      # LEFT

    # ----------------------------------------------
    # Regression targets
    # ----------------------------------------------

    # Desired movement velocities, normalized to [-1, 1].
    vx = np.clip(dx / 5.0, -1.0, 1.0)
    vy = np.clip(dy / 5.0, -1.0, 1.0)

    # An illustrative normalized joint-position target.
    # dx = -10 gives 0.0; dx = 0 gives 0.5; dx = 10 gives 1.0.
    joint_position = np.clip(0.5 + dx / 20.0, 0.0, 1.0)

    # Model input: four features.
    X = np.column_stack([
        robot_x,
        robot_y,
        target_x,
        target_y,
    ]).astype(np.float32)

    # Classification output: one integer per sample.
    y_class = actions

    # Regression output: three continuous values per sample.
    y_regression = np.column_stack([
        vx,
        vy,
        joint_position,
    ]).astype(np.float32)

    return X, y_class, y_regression


X, y_class, y_regression = generate_dataset(NUM_SAMPLES)

print("\nDataset generated!")
print("Input shape:", X.shape)
print("Classification labels shape:", y_class.shape)
print("Regression targets shape:", y_regression.shape)


# --------------------------------------------------
# 3. Convert NumPy arrays to PyTorch tensors
# --------------------------------------------------

X_tensor = torch.tensor(X, dtype=torch.float32)
y_class_tensor = torch.tensor(y_class, dtype=torch.long)
y_regression_tensor = torch.tensor(
    y_regression,
    dtype=torch.float32,
)

dataset = TensorDataset(
    X_tensor,
    y_class_tensor,
    y_regression_tensor,
)

# 80% training data, 20% testing data.
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("\nDataset split:")
print("Training samples:", train_size)
print("Testing samples:", test_size)


# --------------------------------------------------
# 4. Classification model
# --------------------------------------------------

class ClassificationModel(nn.Module):
    """
    Input:
        [robot_x, robot_y, target_x, target_y]

    Output:
        Three class scores (logits):
        LEFT, RIGHT, FORWARD
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# 5. Regression model
# --------------------------------------------------

class RegressionModel(nn.Module):
    """
    Input:
        [robot_x, robot_y, target_x, target_y]

    Output:
        [vx, vy, joint_position]
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 3),

            # Keep predicted values between 0 and 1
            # for this particular synthetic dataset?
        )

    def forward(self, x):
        return self.network(x)


classification_model = ClassificationModel().to(DEVICE)
regression_model = RegressionModel().to(DEVICE)


# --------------------------------------------------
# 6. Loss functions and optimizers
# --------------------------------------------------

# Classification:
# CrossEntropyLoss combines LogSoftmax and NLLLoss.
# Give it raw logits, not Softmax probabilities.
classification_loss_fn = nn.CrossEntropyLoss()

# Regression:
# Mean Squared Error measures numerical prediction error.
regression_loss_fn = nn.MSELoss()

classification_optimizer = torch.optim.Adam(
    classification_model.parameters(),
    lr=LEARNING_RATE,
)

regression_optimizer = torch.optim.Adam(
    regression_model.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# 7. Train both models
# --------------------------------------------------

classification_losses = []
regression_losses = []

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    classification_model.train()
    regression_model.train()

    total_classification_loss = 0.0
    total_regression_loss = 0.0

    for inputs, class_labels, regression_targets in train_loader:

        inputs = inputs.to(DEVICE)
        class_labels = class_labels.to(DEVICE)
        regression_targets = regression_targets.to(DEVICE)

        # ------------------------------------------
        # Train classification model
        # ------------------------------------------

        classification_optimizer.zero_grad()

        class_logits = classification_model(inputs)

        class_loss = classification_loss_fn(
            class_logits,
            class_labels,
        )

        class_loss.backward()
        classification_optimizer.step()

        # ------------------------------------------
        # Train regression model
        # ------------------------------------------

        regression_optimizer.zero_grad()

        regression_predictions = regression_model(inputs)

        regression_loss = regression_loss_fn(
            regression_predictions,
            regression_targets,
        )

        regression_loss.backward()
        regression_optimizer.step()

        total_classification_loss += class_loss.item()
        total_regression_loss += regression_loss.item()

    average_classification_loss = (
        total_classification_loss / len(train_loader)
    )

    average_regression_loss = (
        total_regression_loss / len(train_loader)
    )

    classification_losses.append(average_classification_loss)
    regression_losses.append(average_regression_loss)

    if (epoch + 1) % 15 == 0:
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Classification Loss: {average_classification_loss:.4f} "
            f"| Regression Loss: {average_regression_loss:.4f}"
        )


# --------------------------------------------------
# 8. Evaluate classification model
# --------------------------------------------------

classification_model.eval()

correct = 0
total = 0

with torch.no_grad():

    for inputs, class_labels, _ in test_loader:

        inputs = inputs.to(DEVICE)
        class_labels = class_labels.to(DEVICE)

        logits = classification_model(inputs)

        # Softmax converts logits into probabilities.
        probabilities = torch.softmax(logits, dim=1)

        predicted_classes = torch.argmax(probabilities, dim=1)

        correct += (predicted_classes == class_labels).sum().item()
        total += class_labels.size(0)

classification_accuracy = 100.0 * correct / total


# --------------------------------------------------
# 9. Evaluate regression model
# --------------------------------------------------

regression_model.eval()

total_squared_error = 0.0
total_values = 0

with torch.no_grad():

    for inputs, _, regression_targets in test_loader:

        inputs = inputs.to(DEVICE)
        regression_targets = regression_targets.to(DEVICE)

        predictions = regression_model(inputs)

        squared_error = (
            (predictions - regression_targets) ** 2
        )

        total_squared_error += squared_error.sum().item()
        total_values += regression_targets.numel()

regression_mse = total_squared_error / total_values


# --------------------------------------------------
# 10. Display results
# --------------------------------------------------

print("\n========== TEST RESULTS ==========")

print(
    f"Classification Accuracy: "
    f"{classification_accuracy:.2f}%"
)

print(
    f"Regression Mean Squared Error: "
    f"{regression_mse:.6f}"
)


# --------------------------------------------------
# 11. Test both models on new examples
# --------------------------------------------------

print("\n========== NEW ROBOT EXAMPLES ==========")

new_examples = np.array([
    [2, 3, 8, 3],
    [8, 4, 2, 4],
    [5, 5, 5, 9],
    [1, 2, 4, 8],
    [9, 8, 3, 2],
], dtype=np.float32)

new_inputs = torch.tensor(
    new_examples,
    dtype=torch.float32,
).to(DEVICE)

classification_model.eval()
regression_model.eval()

with torch.no_grad():

    logits = classification_model(new_inputs)

    probabilities = torch.softmax(logits, dim=1)

    predicted_actions = torch.argmax(
        probabilities,
        dim=1,
    )

    continuous_commands = regression_model(new_inputs)

for i in range(len(new_examples)):

    robot_x, robot_y, target_x, target_y = new_examples[i]

    action_id = predicted_actions[i].item()
    action_name = ACTION_NAMES[action_id]

    vx, vy, joint_position = (
        continuous_commands[i].cpu().numpy()
    )

    print(f"\nExample {i + 1}")
    print(f"Robot position: ({robot_x:.1f}, {robot_y:.1f})")
    print(f"Target position: ({target_x:.1f}, {target_y:.1f})")

    print(f"Classification action: {action_name}")

    print(
        "Action probabilities:",
        probabilities[i].cpu().numpy().round(3),
    )

    print(f"Predicted vx: {vx:.3f}")
    print(f"Predicted vy: {vy:.3f}")
    print(f"Predicted joint position: {joint_position:.3f}")


# --------------------------------------------------
# 12. Visualize training losses
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    classification_losses,
    label="Classification - Cross Entropy",
)

plt.plot(
    regression_losses,
    label="Regression - MSE",
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Classification versus Regression Training")
plt.legend()
plt.grid(True)

# Save relative to this Python file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "classification_regression")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plot_path = os.path.join(OUTPUT_DIR, "training_losses.png")

plt.savefig(plot_path, dpi=150, bbox_inches="tight")

print("\nTraining loss graph saved to:", plot_path)

plt.show()


print("\nLesson completed!")
