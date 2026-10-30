
"""
STEP 1 - LESSON 12
Imitation Learning V2

Improvements:
1. Structured grid observations: 3 x 10 x 10
2. Residual CNN architecture
3. Multiple demonstrations per environment
4. Separate training and validation environments
5. Class-weighted cross entropy
6. Validation metrics and early stopping
7. Per-action accuracy
8. Strict and recovery navigation evaluation

Run:
    python step1/imitation_learning.py
"""

import os
import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

GRID_SIZE = 10

# Generate multiple environments.
NUM_TRAIN_MAPS = 1200
NUM_VAL_MAPS = 300

BATCH_SIZE = 256
MAX_EPOCHS = 100
LEARNING_RATE = 0.001

OBSTACLE_PROBABILITY = 0.15

EARLY_STOPPING_PATIENCE = 12

MAX_ROLLOUT_STEPS = 100
NUM_EVALUATION_EPISODES = 200

NUM_ACTIONS = 4

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}

ACTION_DELTAS = {
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0),
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "imitation_learning_v2",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "imitation_model_v2.pth",
)

LOSS_GRAPH_PATH = os.path.join(
    OUTPUT_DIR,
    "training_history.png",
)

SEED = int(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 60)
print("IMITATION LEARNING V2")
print("=" * 60)

print("Device:", DEVICE)


# ============================================================
# 2. BFS EXPERT
# ============================================================

def get_neighbors(position, obstacles):

    x, y = position

    for action, (dx, dy) in ACTION_DELTAS.items():

        nx = x + dx
        ny = y + dy

        if not (0 <= nx < GRID_SIZE):
            continue

        if not (0 <= ny < GRID_SIZE):
            continue

        next_position = (nx, ny)

        if next_position in obstacles:
            continue

        yield next_position, action


def bfs_expert(robot, target, obstacles):
    """
    Find the shortest path using BFS.

    Returns the first action on the shortest path.
    """

    if robot == target:
        return None

    queue = deque([robot])

    visited = {
        robot: (None, None)
    }

    while queue:

        current = queue.popleft()

        if current == target:
            break

        for neighbor, action in get_neighbors(
            current,
            obstacles
        ):

            if neighbor in visited:
                continue

            visited[neighbor] = (
                current,
                action
            )

            queue.append(neighbor)

    if target not in visited:
        return None

    current = target

    while True:

        previous, action = visited[current]

        if previous == robot:
            return action

        current = previous


# ============================================================
# 3. GENERATE ENVIRONMENTS
# ============================================================

def generate_environment():

    all_cells = [
        (x, y)
        for x in range(GRID_SIZE)
        for y in range(GRID_SIZE)
    ]

    while True:

        robot = random.choice(all_cells)
        target = random.choice(all_cells)

        if robot == target:
            continue

        obstacles = set()

        for cell in all_cells:

            if cell == robot or cell == target:
                continue

            if random.random() < OBSTACLE_PROBABILITY:
                obstacles.add(cell)

        # Verify that a path exists.
        if bfs_expert(robot, target, obstacles) is not None:

            return robot, target, obstacles


# ============================================================
# 4. STRUCTURED GRID OBSERVATION
# ============================================================

def render_grid(robot, target, obstacles):
    """
    Return a structured observation.

    Shape: [3, 10, 10]

    Channel 0: Robot
    Channel 1: Target
    Channel 2: Obstacles

    This is a structured visual representation,
    not a raw RGB camera image.
    """

    observation = np.zeros(
        (3, GRID_SIZE, GRID_SIZE),
        dtype=np.uint8
    )

    rx, ry = robot
    tx, ty = target

    observation[0, ry, rx] = 1
    observation[1, ty, tx] = 1

    for x, y in obstacles:
        observation[2, y, x] = 1

    return observation


# ============================================================
# 5. COLLECT MULTIPLE DEMONSTRATIONS PER MAP
# ============================================================

def collect_demonstrations(num_maps, description):

    images = []
    actions = []

    print(
        f"\nCollecting {description} demonstrations..."
    )

    for map_index in range(num_maps):

        robot_start, target, obstacles = generate_environment()

        # Every reachable free cell can be a robot state.
        # BFS provides the expert action for that state.

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):

                robot = (x, y)

                if robot == target:
                    continue

                if robot in obstacles:
                    continue

                expert_action = bfs_expert(
                    robot,
                    target,
                    obstacles
                )

                # Ignore unreachable positions.
                if expert_action is None:
                    continue

                observation = render_grid(
                    robot,
                    target,
                    obstacles
                )

                images.append(observation)
                actions.append(expert_action)

        if (map_index + 1) % 100 == 0:

            print(
                f"{description}: "
                f"{map_index + 1}/{num_maps} maps | "
                f"{len(images)} demonstrations"
            )

    images = np.asarray(
        images,
        dtype=np.uint8
    )

    actions = np.asarray(
        actions,
        dtype=np.int64
    )

    return images, actions


# IMPORTANT:
# Training and validation are generated from different maps.
# This prevents the same environment from appearing in both sets.

train_images, train_actions = collect_demonstrations(
    NUM_TRAIN_MAPS,
    "Training"
)

val_images, val_actions = collect_demonstrations(
    NUM_VAL_MAPS,
    "Validation"
)

print("\nDataset summary")

print("Training observations:", train_images.shape)
print("Training actions:", train_actions.shape)

print("Validation observations:", val_images.shape)
print("Validation actions:", val_actions.shape)


# ============================================================
# 6. ACTION DISTRIBUTION
# ============================================================

def print_action_distribution(actions, title):

    print(f"\n{title}")

    for action_id, action_name in ACTION_NAMES.items():

        count = int(np.sum(actions == action_id))

        percentage = (
            100.0 * count / len(actions)
        )

        print(
            f"{action_name:>6}: "
            f"{count:8d} "
            f"({percentage:5.2f}%)"
        )


print_action_distribution(
    train_actions,
    "Training action distribution"
)


# ============================================================
# 7. PYTORCH DATASETS
# ============================================================

train_images_tensor = torch.from_numpy(
    train_images
)

train_actions_tensor = torch.from_numpy(
    train_actions
).long()

val_images_tensor = torch.from_numpy(
    val_images
)

val_actions_tensor = torch.from_numpy(
    val_actions
).long()

train_dataset = TensorDataset(
    train_images_tensor,
    train_actions_tensor
)

val_dataset = TensorDataset(
    val_images_tensor,
    val_actions_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# 8. RESIDUAL CNN
# ============================================================

class ResidualBlock(nn.Module):
    """
    Two convolution layers with a skip connection.
    """

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual / skip connection.
        out = out + identity

        out = self.relu(out)

        return out


class ImitationCNNV2(nn.Module):

    def __init__(self):

        super().__init__()

        # Initial feature extraction.
        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True)
        )

        # Four residual blocks.
        self.residual_blocks = nn.Sequential(

            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        # Keep the 10x10 spatial representation.
        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * GRID_SIZE * GRID_SIZE,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                NUM_ACTIONS
            )
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.residual_blocks(x)

        x = self.classifier(x)

        return x


model = ImitationCNNV2().to(DEVICE)

print("\nModel architecture:")
print(model)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(
    f"\nTrainable parameters: {parameter_count:,}"
)


# ============================================================
# 9. CLASS-WEIGHTED LOSS
# ============================================================

class_counts = np.bincount(
    train_actions,
    minlength=NUM_ACTIONS
)

# Inverse square-root weighting is less aggressive
# than ordinary inverse-frequency weighting.

class_weights = 1.0 / np.sqrt(
    np.maximum(class_counts, 1)
)

class_weights = class_weights / class_weights.mean()

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print("\nClass weights:")

for action_id, action_name in ACTION_NAMES.items():

    print(
        f"{action_name:>6}: "
        f"{class_weights[action_id]:.3f}"
    )

loss_fn = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=4
)


# ============================================================
# 10. TRAINING AND VALIDATION FUNCTIONS
# ============================================================

def calculate_metrics(loader, training=False):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    correct = 0

    class_correct = np.zeros(
        NUM_ACTIONS,
        dtype=np.int64
    )

    class_total = np.zeros(
        NUM_ACTIONS,
        dtype=np.int64
    )

    for batch_images, batch_actions in loader:

        batch_images = batch_images.to(
            DEVICE,
            non_blocking=True
        ).float()

        batch_actions = batch_actions.to(
            DEVICE,
            non_blocking=True
        )

        if training:

            optimizer.zero_grad()

            logits = model(batch_images)

            loss = loss_fn(
                logits,
                batch_actions
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        else:

            with torch.no_grad():

                logits = model(batch_images)

                loss = loss_fn(
                    logits,
                    batch_actions
                )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        batch_size = batch_actions.size(0)

        total_loss += loss.item() * batch_size

        total_samples += batch_size

        correct += (
            predictions == batch_actions
        ).sum().item()

        for action_id in range(NUM_ACTIONS):

            mask = batch_actions == action_id

            class_total[action_id] += mask.sum().item()

            class_correct[action_id] += (
                (predictions == batch_actions) & mask
            ).sum().item()

    average_loss = total_loss / total_samples

    accuracy = 100.0 * correct / total_samples

    per_class_accuracy = {}

    for action_id, action_name in ACTION_NAMES.items():

        if class_total[action_id] > 0:

            class_accuracy = (
                100.0
                * class_correct[action_id]
                / class_total[action_id]
            )

        else:

            class_accuracy = 0.0

        per_class_accuracy[action_name] = class_accuracy

    return (
        average_loss,
        accuracy,
        per_class_accuracy
    )


# ============================================================
# 11. TRAINING LOOP WITH EARLY STOPPING
# ============================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
}

best_val_loss = float("inf")

epochs_without_improvement = 0

print("\nStarting Version 2 training...\n")

for epoch in range(MAX_EPOCHS):

    train_loss, train_accuracy, _ = calculate_metrics(
        train_loader,
        training=True
    )

    val_loss, val_accuracy, val_per_class = calculate_metrics(
        val_loader,
        training=False
    )

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    history["train_accuracy"].append(train_accuracy)
    history["val_accuracy"].append(val_accuracy)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch + 1:03d}/{MAX_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {val_accuracy:.2f}% | "
        f"LR: {current_lr:.2e}"
    )

    # Save the best validation model.
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "grid_size": GRID_SIZE,
                "num_actions": NUM_ACTIONS,
                "input_channels": 3,
                "architecture": "ImitationCNNV2",
                "action_names": ACTION_NAMES,
                "best_val_loss": best_val_loss,
            },
            MODEL_PATH
        )

        print("  -> Best model saved.")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print(
            "\nEarly stopping: validation loss "
            "has not improved."
        )

        break


# ============================================================
# 12. LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nLoaded best validation checkpoint.")

print(
    "Best validation loss:",
    checkpoint["best_val_loss"]
)


# ============================================================
# 13. FINAL VALIDATION METRICS
# ============================================================

val_loss, val_accuracy, per_class_accuracy = calculate_metrics(
    val_loader,
    training=False
)

print("\n" + "=" * 60)
print("FINAL VALIDATION RESULTS")
print("=" * 60)

print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.2f}%")

print("\nAccuracy for each action:")

for action_name, accuracy in per_class_accuracy.items():

    print(
        f"{action_name:>6}: {accuracy:.2f}%"
    )


# ============================================================
# 14. TRAINING GRAPHS
# ============================================================

epochs = range(
    1,
    len(history["train_loss"]) + 1
)

plt.figure(figsize=(10, 5))

plt.plot(
    epochs,
    history["train_loss"],
    label="Training loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")

plt.title("Imitation Learning V2 - Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    LOSS_GRAPH_PATH,
    dpi=150
)

plt.show()


# Accuracy graph.
plt.figure(figsize=(10, 5))

plt.plot(
    epochs,
    history["train_accuracy"],
    label="Training accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    label="Validation accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title("Imitation Learning V2 - Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "training_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=150
)

plt.show()


# ============================================================
# 15. NAVIGATION EVALUATION
# ============================================================

def run_expert_episode(robot, target, obstacles):

    robot = tuple(robot)

    for _ in range(MAX_ROLLOUT_STEPS):

        if robot == target:
            return True

        action = bfs_expert(
            robot,
            target,
            obstacles
        )

        if action is None:
            return False

        dx, dy = ACTION_DELTAS[action]

        robot = (
            robot[0] + dx,
            robot[1] + dy
        )

        if robot in obstacles:
            return False

    return robot == target


def run_learned_episode(
    robot,
    target,
    obstacles,
    allow_recovery=False
):

    robot = tuple(robot)

    for _ in range(MAX_ROLLOUT_STEPS):

        if robot == target:
            return True

        observation = render_grid(
            robot,
            target,
            obstacles
        )

        image_tensor = torch.from_numpy(
            observation
        ).unsqueeze(0).to(DEVICE).float()

        with torch.no_grad():

            logits = model(image_tensor)

            action = torch.argmax(
                logits,
                dim=1
            ).item()

        dx, dy = ACTION_DELTAS[action]

        next_robot = (
            robot[0] + dx,
            robot[1] + dy
        )

        nx, ny = next_robot

        invalid = (
            not (0 <= nx < GRID_SIZE)
            or not (0 <= ny < GRID_SIZE)
            or next_robot in obstacles
        )

        if invalid:

            if allow_recovery:
                # Robot stays in place and gets another decision.
                continue

            # Strict evaluation: invalid action is failure.
            return False

        robot = next_robot

    return robot == target


def evaluate_navigation(num_episodes):

    expert_successes = 0

    strict_successes = 0

    recovery_successes = 0

    print(
        f"\nEvaluating {num_episodes} unseen environments..."
    )

    for episode in range(num_episodes):

        robot, target, obstacles = generate_environment()

        expert_success = run_expert_episode(
            robot,
            target,
            obstacles
        )

        strict_success = run_learned_episode(
            robot,
            target,
            obstacles,
            allow_recovery=False
        )

        recovery_success = run_learned_episode(
            robot,
            target,
            obstacles,
            allow_recovery=True
        )

        expert_successes += int(expert_success)

        strict_successes += int(strict_success)

        recovery_successes += int(recovery_success)

        if (episode + 1) % 50 == 0:

            print(
                f"Evaluated {episode + 1}/{num_episodes}"
            )

    expert_rate = (
        100.0 * expert_successes / num_episodes
    )

    strict_rate = (
        100.0 * strict_successes / num_episodes
    )

    recovery_rate = (
        100.0 * recovery_successes / num_episodes
    )

    print("\n" + "=" * 60)
    print("NAVIGATION EVALUATION")
    print("=" * 60)

    print(
        f"Expert success rate:       {expert_rate:.2f}%"
    )

    print(
        f"Learned strict success:    {strict_rate:.2f}%"
    )

    print(
        f"Learned recovery success:  {recovery_rate:.2f}%"
    )

    return (
        expert_rate,
        strict_rate,
        recovery_rate
    )


expert_rate, strict_rate, recovery_rate = evaluate_navigation(
    NUM_EVALUATION_EPISODES
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("IMITATION LEARNING V2 COMPLETED")
print("=" * 60)

print("1. Generated multiple training environments.")
print("2. Collected demonstrations from many robot positions.")
print("3. Used structured 10x10 grid observations.")
print("4. Trained a residual CNN.")
print("5. Used weighted cross-entropy loss.")
print("6. Monitored validation performance.")
print("7. Saved the best validation checkpoint.")
print("8. Evaluated complete navigation episodes.")

print("\nModel saved to:")
print(MODEL_PATH)

print("\nTraining graphs saved to:")
print(OUTPUT_DIR)

print("\nIMPORTANT:")
print("Validation accuracy measures individual action prediction.")
print("Navigation success measures complete robot tasks.")

print("\nCompare Version 1 and Version 2 using navigation success.")
