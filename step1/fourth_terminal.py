
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1. SETTINGS
# ============================================================

GRID_SIZE = 5
NUM_ACTIONS = 5

# Actions:
# 0 = Stay
# 1 = Left
# 2 = Right
# 3 = Down
# 4 = Up

ACTIONS = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}

ACTION_NAMES = {
    0: "Stay",
    1: "Left",
    2: "Right",
    3: "Down",
    4: "Up",
}

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)


# ============================================================
# 2. ENVIRONMENT
# ============================================================

def inside_grid(position):
    """Check whether a position is inside the grid."""

    x, y = position

    return (
        0 <= x < GRID_SIZE
        and 0 <= y < GRID_SIZE
    )


def valid_position(position, obstacles):
    """Check grid boundaries and obstacles."""

    return (
        inside_grid(position)
        and position not in obstacles
    )


def get_next_position(position, action):
    """Calculate the next position after an action."""

    x, y = position
    dx, dy = ACTIONS[action]

    return (x + dx, y + dy)


# ============================================================
# 3. PATHFINDING: FIND A SAFE ROUTE
# ============================================================

def find_path(start, goal, obstacles):
    """
    Find a shortest safe path using BFS.

    Returns a list of positions, or None if no path exists.
    """

    if not valid_position(start, obstacles):
        return None

    if not valid_position(goal, obstacles):
        return None

    queue = deque([start])

    # Remember how we reached each position.
    parent = {start: None}

    while queue:

        current = queue.popleft()

        if current == goal:
            break

        # Explore left, right, down, and up.
        for action in [1, 2, 3, 4]:

            next_position = get_next_position(
                current,
                action
            )

            if (
                valid_position(next_position, obstacles)
                and next_position not in parent
            ):
                parent[next_position] = current
                queue.append(next_position)

    if goal not in parent:
        return None

    # Reconstruct the route.
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()

    return path


# ============================================================
# 4. ENCODE THE OBSERVATION
# ============================================================

def encode_observation(position, goal, obstacles):
    """
    Convert the robot's situation into 29 numbers.

    4 numbers: robot and target coordinates.
    25 numbers: obstacle map.
    """

    robot_x, robot_y = position
    goal_x, goal_y = goal

    features = [
        robot_x / (GRID_SIZE - 1),
        robot_y / (GRID_SIZE - 1),
        goal_x / (GRID_SIZE - 1),
        goal_y / (GRID_SIZE - 1),
    ]

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):

            if (x, y) in obstacles:
                features.append(1.0)
            else:
                features.append(0.0)

    return np.array(features, dtype=np.float32)


# ============================================================
# 5. GENERATE TRAINING DATA
# ============================================================

def generate_training_data(number_of_examples=2500):

    training_inputs = []
    training_labels = []

    generated = 0
    attempts = 0

    while (
        generated < number_of_examples
        and attempts < number_of_examples * 10
    ):

        attempts += 1

        # Random obstacle layout.
        obstacles = set()

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):

                if random.random() < 0.18:
                    obstacles.add((x, y))

        # Random robot start and target.
        start = (
            random.randrange(GRID_SIZE),
            random.randrange(GRID_SIZE),
        )

        goal = (
            random.randrange(GRID_SIZE),
            random.randrange(GRID_SIZE),
        )

        if start == goal:
            continue

        # Start and target cannot be blocked.
        obstacles.discard(start)
        obstacles.discard(goal)

        # Find a successful route.
        path = find_path(start, goal, obstacles)

        if path is None or len(path) < 2:
            continue

        # Convert each route step into a labeled example.
        for index in range(len(path) - 1):

            current = path[index]
            next_position = path[index + 1]

            dx = next_position[0] - current[0]
            dy = next_position[1] - current[1]

            if (dx, dy) == (-1, 0):
                action = 1

            elif (dx, dy) == (1, 0):
                action = 2

            elif (dx, dy) == (0, -1):
                action = 3

            elif (dx, dy) == (0, 1):
                action = 4

            else:
                continue

            observation = encode_observation(
                current,
                goal,
                obstacles
            )

            training_inputs.append(observation)
            training_labels.append(action)

        generated += 1

    if not training_inputs:
        raise RuntimeError(
            "Could not generate training examples."
        )

    X = torch.tensor(
        np.array(training_inputs),
        dtype=torch.float32
    )

    y = torch.tensor(
        training_labels,
        dtype=torch.long
    )

    return X, y


# ============================================================
# 6. NEURAL NETWORK
# ============================================================

class RobotBrain(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, NUM_ACTIONS)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 7. TRAIN THE AI
# ============================================================

def train_robot():

    print("\nGenerating training data...")

    X, y = generate_training_data()

    # Split into training and test sets.
    permutation = torch.randperm(len(X))

    split = int(len(X) * 0.8)

    train_indices = permutation[:split]
    test_indices = permutation[split:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    model = RobotBrain()

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 60

    print("Training the neural network...\n")

    for epoch in range(epochs):

        model.train()

        predictions = model(X_train)

        loss = loss_function(
            predictions,
            y_train
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:

            model.eval()

            with torch.no_grad():

                test_predictions = model(X_test)

                predicted_actions = (
                    test_predictions.argmax(dim=1)
                )

                accuracy = (
                    predicted_actions == y_test
                ).float().mean().item()

            print(
                f"Epoch {epoch + 1:2d}/{epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"Test accuracy: {accuracy:.1%}"
            )

    print("\nTraining complete!")

    return model


# ============================================================
# 8. AI ACTION SELECTION
# ============================================================

def choose_action(model, position, goal, obstacles):
    """
    The neural network scores actions.

    Pathfinding restricts the choices to safe actions
    that reduce the shortest remaining distance.
    """

    current_path = find_path(
        position,
        goal,
        obstacles
    )

    if current_path is None:
        return None

    if len(current_path) <= 1:
        return 0

    observation = encode_observation(
        position,
        goal,
        obstacles
    )

    input_tensor = torch.tensor(
        observation,
        dtype=torch.float32
    ).unsqueeze(0)

    model.eval()

    with torch.no_grad():

        scores = model(input_tensor)[0]

        allowed_actions = []

        current_distance = len(current_path) - 1

        for action in range(1, NUM_ACTIONS):

            next_position = get_next_position(
                position,
                action
            )

            if not valid_position(
                next_position,
                obstacles
            ):
                continue

            next_path = find_path(
                next_position,
                goal,
                obstacles
            )

            if next_path is None:
                continue

            next_distance = len(next_path) - 1

            if next_distance == current_distance - 1:
                allowed_actions.append(action)

        if not allowed_actions:
            return None

        best_action = max(
            allowed_actions,
            key=lambda action: scores[action].item()
        )

    return best_action


# ============================================================
# 9. DISPLAY THE GRID
# ============================================================

def display_grid(position, goal, obstacles, trail):

    print("\nGrid:")

    for y in reversed(range(GRID_SIZE)):

        row = []

        for x in range(GRID_SIZE):

            cell = (x, y)

            if cell == position:
                symbol = "R"       # Robot

            elif cell == goal:
                symbol = "G"       # Goal

            elif cell in obstacles:
                symbol = "#"       # Obstacle

            elif cell in trail:
                symbol = "*"       # Visited cell

            else:
                symbol = "."       # Empty cell

            row.append(symbol)

        print(" ".join(row))

    print("\nLegend: R=Robot  G=Goal  #=Obstacle  *=Trail")


# ============================================================
# 10. RUN THE ROBOT
# ============================================================

def run_robot(model):

    # Fixed environment for easier learning.
    obstacles = {
        (2, 1),
        (2, 2),
        (2, 3),
    }

    position = (0, 0)
    goal = (4, 4)

    trail = [position]
    steps = 0

    print("\n" + "=" * 45)
    print("AI ROBOT NAVIGATION")
    print("=" * 45)

    print("Start:", position)
    print("Goal:", goal)
    print("Obstacles:", sorted(obstacles))

    display_grid(
        position,
        goal,
        obstacles,
        trail
    )

    while position != goal:

        action = choose_action(
            model,
            position,
            goal,
            obstacles
        )

        if action is None:
            print("\nNo safe route found. Stopping.")
            return

        next_position = get_next_position(
            position,
            action
        )

        # Final safety check.
        if not valid_position(
            next_position,
            obstacles
        ):
            print("\nSafety check failed. Stopping.")
            return

        position = next_position
        trail.append(position)
        steps += 1

        print(
            f"\nStep {steps}: "
            f"{ACTION_NAMES[action]} -> {position}"
        )

        display_grid(
            position,
            goal,
            obstacles,
            trail
        )

        # Pause so you can follow each step.
        input("\nPress ENTER for the next step...")

    print("\n" + "=" * 45)
    print("SUCCESS! ROBOT REACHED THE GOAL.")
    print("Total steps:", steps)
    print("=" * 45)


# ============================================================
# 11. MAIN
# ============================================================

if __name__ == "__main__":

    model = train_robot()

    run_robot(model)
