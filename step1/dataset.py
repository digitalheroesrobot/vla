"""
dataset.py

Lesson 8: Creating a Dataset for Robot Learning

This exercise teaches:

1. Generate robot observations.
2. Generate corresponding robot actions.
3. Understand features and labels.
4. Save datasets using NumPy.
5. Load datasets using NumPy.
6. Prepare datasets using PyTorch.
7. Visualize the dataset using Matplotlib.

Robot observation:
    [robot_x, robot_y, target_x, target_y]

Robot actions:
    0 = LEFT
    1 = STAY
    2 = RIGHT
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt


# ==================================================
# 1. Configuration
# ==================================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

NUM_SAMPLES = 2000

GRID_SIZE = 10

DATA_DIR = "dataset/robot_navigation"

DATASET_PATH = os.path.join(
    DATA_DIR,
    "navigation_dataset.npz"
)

PLOT_PATH = os.path.join(
    DATA_DIR,
    "dataset_visualization.png"
)

ACTION_NAMES = {
    0: "LEFT",
    1: "STAY",
    2: "RIGHT"
}


# ==================================================
# 2. Generate robot learning dataset
# ==================================================

def generate_dataset(num_samples):

    observations = []
    actions = []

    for _ in range(num_samples):

        # Generate robot position
        robot_x = random.randint(0, GRID_SIZE)
        robot_y = random.randint(0, GRID_SIZE)

        # Generate target position
        target_x = random.randint(0, GRID_SIZE)
        target_y = random.randint(0, GRID_SIZE)

        # Calculate horizontal distance
        distance_x = target_x - robot_x

        # Determine the correct action
        if distance_x < 0:
            action = 0       # LEFT

        elif distance_x > 0:
            action = 2       # RIGHT

        else:
            action = 1       # STAY

        # Store observation
        observation = [
            robot_x,
            robot_y,
            target_x,
            target_y
        ]

        observations.append(observation)

        # Store corresponding action
        actions.append(action)

    # Convert Python lists into NumPy arrays
    observations = np.array(
        observations,
        dtype=np.float32
    )

    actions = np.array(
        actions,
        dtype=np.int64
    )

    return observations, actions


# ==================================================
# 3. Generate dataset
# ==================================================

print("\n========== GENERATING DATASET ==========")

X, y = generate_dataset(NUM_SAMPLES)

print("Dataset generated successfully!")

print("Observations shape:", X.shape)
print("Actions shape:", y.shape)

print("\nFirst five examples:")

for i in range(5):

    print(
        f"Observation: {X[i]} | "
        f"Action: {ACTION_NAMES[y[i]]}"
    )


# ==================================================
# 4. Save dataset using NumPy
# ==================================================

os.makedirs(DATA_DIR, exist_ok=True)

np.savez_compressed(
    DATASET_PATH,
    observations=X,
    actions=y
)

print("\n========== SAVING DATASET ==========")

print("Dataset saved to:", DATASET_PATH)


# ==================================================
# 5. Load dataset from disk
# ==================================================

print("\n========== LOADING DATASET ==========")

loaded_data = np.load(DATASET_PATH)

X_loaded = loaded_data["observations"]
y_loaded = loaded_data["actions"]

print("Dataset loaded successfully!")

print("Loaded observations:", X_loaded.shape)
print("Loaded actions:", y_loaded.shape)


# ==================================================
# 6. Verify dataset
# ==================================================

assert np.array_equal(X, X_loaded)
assert np.array_equal(y, y_loaded)

print("\nDataset verification passed!")

print("Saved data matches original data.")


# ==================================================
# 7. Prepare dataset using PyTorch
# ==================================================

print("\n========== PYTORCH DATASET ==========")

# Convert NumPy arrays into PyTorch tensors
X_tensor = torch.tensor(
    X_loaded,
    dtype=torch.float32
)

y_tensor = torch.tensor(
    y_loaded,
    dtype=torch.long
)

# Combine observations and actions
robot_dataset = TensorDataset(
    X_tensor,
    y_tensor
)

# Create batches
dataloader = DataLoader(
    robot_dataset,
    batch_size=32,
    shuffle=True
)

print("Total samples:", len(robot_dataset))
print("Batch size:", 32)
print("Number of batches:", len(dataloader))


# ==================================================
# 8. Inspect one training batch
# ==================================================

print("\n========== INSPECTING A BATCH ==========")

for batch_observations, batch_actions in dataloader:

    print(
        "Batch observations shape:",
        batch_observations.shape
    )

    print(
        "Batch actions shape:",
        batch_actions.shape
    )

    print("\nFirst five samples in this batch:")

    for i in range(5):

        observation = batch_observations[i]

        action = batch_actions[i].item()

        print(
            f"Observation: {observation.tolist()} | "
            f"Action: {ACTION_NAMES[action]}"
        )

    break


# ==================================================
# 9. Analyze action distribution
# ==================================================

print("\n========== ACTION DISTRIBUTION ==========")

unique_actions, action_counts = np.unique(
    y_loaded,
    return_counts=True
)

for action, count in zip(unique_actions, action_counts):

    percentage = count / len(y_loaded) * 100

    print(
        f"{ACTION_NAMES[action]}: "
        f"{count} samples ({percentage:.2f}%)"
    )


# ==================================================
# 10. Visualize dataset
# ==================================================

print("\n========== VISUALIZING DATASET ==========")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Plot 1: Robot positions
scatter = axes[0].scatter(
    X_loaded[:, 0],
    X_loaded[:, 1],
    c=y_loaded,
    cmap="viridis",
    alpha=0.6
)

axes[0].set_title("Robot Position Distribution")
axes[0].set_xlabel("Robot X")
axes[0].set_ylabel("Robot Y")
axes[0].set_xlim(-1, GRID_SIZE + 1)
axes[0].set_ylim(-1, GRID_SIZE + 1)
axes[0].grid(True)

# Plot 2: Target positions
axes[1].scatter(
    X_loaded[:, 2],
    X_loaded[:, 3],
    c=y_loaded,
    cmap="viridis",
    alpha=0.6
)

axes[1].set_title("Target Position Distribution")
axes[1].set_xlabel("Target X")
axes[1].set_ylabel("Target Y")
axes[1].set_xlim(-1, GRID_SIZE + 1)
axes[1].set_ylim(-1, GRID_SIZE + 1)
axes[1].grid(True)

# Plot 3: Action distribution
action_labels = [
    ACTION_NAMES[int(action)]
    for action in unique_actions
]

axes[2].bar(
    action_labels,
    action_counts
)

axes[2].set_title("Robot Action Distribution")
axes[2].set_xlabel("Action")
axes[2].set_ylabel("Number of Samples")
axes[2].grid(axis="y")

plt.tight_layout()

plt.savefig(PLOT_PATH, dpi=150)

print("Visualization saved to:", PLOT_PATH)

plt.show()


# ==================================================
# 11. Final summary
# ==================================================

print("\n========== DATASET SUMMARY ==========")

print("Total observations:", len(X_loaded))

print("Features per observation:", X_loaded.shape[1])

print("Number of possible actions:", len(ACTION_NAMES))

print("Storage format: NumPy NPZ")

print("Training format: PyTorch TensorDataset")

print("\nDataset lesson completed successfully!")
