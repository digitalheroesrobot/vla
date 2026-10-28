"""
Train an AI Robot to Navigate Around Obstacles
Imagine a wall blocks the robot's direct route to its target.

- Recognize that a cell is blocked.
- Choose a route around the obstacle.
- Reach the target without crossing the wall.

Important: Your current model doesn't know about obstacles, 
so simply drawing a wall won't teach it to avoid one. 
We'll need to update the training data and the model's inputs.
our robot must find a route around the wall instead of walking straight into it.

"""

import random
from collections import deque

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1. SETTINGS
# ============================================================

GRID_SIZE = 5
CELL_SIZE = 100
PANEL_HEIGHT = 90

WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + PANEL_HEIGHT

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

DEVICE = torch.device("cpu")

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


def is_obstacle(position, obstacles):
    """Check whether a position contains an obstacle."""

    return position in obstacles


def valid_position(position, obstacles):
    """Check boundaries and obstacles."""

    return (
        inside_grid(position)
        and not is_obstacle(position, obstacles)
    )


def get_next_position(position, action):
    """Calculate the position after an action."""

    x, y = position
    dx, dy = ACTIONS[action]

    return (x + dx, y + dy)


# ============================================================
# 3. PATHFINDING
# ============================================================

def find_path(start, goal, obstacles):
    """
    Find a shortest safe route using BFS.

    Returns a list of positions from start to goal.
    Returns None if no safe route exists.
    """

    if not valid_position(start, obstacles):
        return None

    if not valid_position(goal, obstacles):
        return None

    queue = deque([start])

    # Remember the position from which each cell was reached.
    parent = {start: None}

    while queue:

        current = queue.popleft()

        if current == goal:
            break

        # Try left, right, down, and up.
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

    # Reconstruct the path from goal back to start.
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
    Create 29 numerical input features.

    4 coordinate features:
        Robot x, robot y, goal x, goal y

    25 obstacle features:
        1 = obstacle
        0 = free cell
    """

    robot_x, robot_y = position
    goal_x, goal_y = goal

    features = [
        robot_x / (GRID_SIZE - 1),
        robot_y / (GRID_SIZE - 1),
        goal_x / (GRID_SIZE - 1),
        goal_y / (GRID_SIZE - 1),
    ]

    # Encode the grid row by row.
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

        # Randomly generate obstacles.
        obstacles = set()

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):

                if random.random() < 0.18:
                    obstacles.add((x, y))

        # Choose random starting and target positions.
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

        # Start and goal must remain unblocked.
        obstacles.discard(start)
        obstacles.discard(goal)

        # Find a safe route.
        path = find_path(start, goal, obstacles)

        if path is None or len(path) < 2:
            continue

        # Convert each step of the route into a training example.
        for index in range(len(path) - 1):

            current_position = path[index]
            next_position = path[index + 1]

            dx = next_position[0] - current_position[0]
            dy = next_position[1] - current_position[1]

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
                current_position,
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
# 7. TRAIN THE NEURAL NETWORK
# ============================================================

def train_robot():

    print("Generating training examples...")

    X, y = generate_training_data()

    # Randomly split examples into training and testing sets.
    permutation = torch.randperm(len(X))

    split = int(len(X) * 0.8)

    train_indices = permutation[:split]
    test_indices = permutation[split:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    model = RobotBrain().to(DEVICE)

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 60

    print("Training robot brain...")

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
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"Test accuracy: {accuracy:.1%}"
            )

    print("Training complete!")

    return model


# ============================================================
# 8. IMPROVED AI ACTION SELECTION
# ============================================================

def choose_action(model, position, goal, obstacles):
    """
    Use the neural network to score actions.

    Then restrict the choice to actions that:
    1. Stay inside the grid.
    2. Avoid obstacles.
    3. Reduce the shortest remaining path by one step.

    This prevents the robot from oscillating between cells
    when a safe route to the target exists.
    """

    # Find the shortest safe route from the current position.
    current_path = find_path(
        position,
        goal,
        obstacles
    )

    # No safe route exists.
    if current_path is None:
        return 0

    # The robot is already at its target.
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

        # The model scores all five possible actions.
        action_scores = model(input_tensor)[0].clone()

        # Initially, every action is disallowed.
        allowed_actions = []

        current_distance = len(current_path) - 1

        # Consider each movement direction.
        for action in range(1, NUM_ACTIONS):

            next_position = get_next_position(
                position,
                action
            )

            # Reject obstacles and positions outside the grid.
            if not valid_position(
                next_position,
                obstacles
            ):
                continue

            # Find the shortest remaining route after this move.
            next_path = find_path(
                next_position,
                goal,
                obstacles
            )

            if next_path is None:
                continue

            next_distance = len(next_path) - 1

            # Only allow moves that reduce the shortest
            # remaining distance by exactly one.
            if next_distance == current_distance - 1:
                allowed_actions.append(action)

        # If a path exists, at least one progress-making
        # action should be available.
        if not allowed_actions:
            return 0

        # Select the model's highest-scoring action
        # among the allowed actions.
        best_action = max(
            allowed_actions,
            key=lambda action: action_scores[action].item()
        )

    return best_action


# ============================================================
# 9. VISUAL SIMULATION
# ============================================================

def run_simulation(model):

    pygame.init()

    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT)
    )

    pygame.display.set_caption(
        "AI Robot - Smart Obstacle Navigation"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 25)
    small_font = pygame.font.SysFont(None, 20)

    robot_start = (0, 0)
    robot_position = robot_start

    goal = (4, 4)

    # Vertical wall with gaps at the top and bottom.
    obstacles = {
        (2, 1),
        (2, 2),
        (2, 3),
    }

    trail = [robot_position]

    running = True
    autonomous = False
    steps = 0

    message = "SPACE: step | A: auto | R: reset"

    last_auto_move = 0

    def perform_ai_step():

        nonlocal robot_position
        nonlocal steps
        nonlocal trail
        nonlocal message
        nonlocal autonomous

        if robot_position == goal:
            message = "Target reached!"
            autonomous = False
            return

        action = choose_action(
            model,
            robot_position,
            goal,
            obstacles
        )

        proposed_position = get_next_position(
            robot_position,
            action
        )

        # Final safety check before moving.
        if not valid_position(
            proposed_position,
            obstacles
        ):
            message = "Safety check blocked a move!"
            autonomous = False
            return

        robot_position = proposed_position

        trail.append(robot_position)
        steps += 1

        message = (
            f"AI action: {ACTION_NAMES[action]} | "
            f"Steps: {steps}"
        )

        if robot_position == goal:
            message = f"Target reached in {steps} steps!"
            autonomous = False

    while running:

        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    perform_ai_step()

                elif event.key == pygame.K_a:

                    if robot_position == goal:
                        message = "Target reached! Press R to reset."
                        autonomous = False
                    else:
                        autonomous = not autonomous

                        message = (
                            "Autonomous mode ON"
                            if autonomous
                            else "Autonomous mode OFF"
                        )

                elif event.key == pygame.K_r:

                    robot_position = robot_start
                    trail = [robot_position]
                    steps = 0
                    autonomous = False

                    message = "Robot reset!"

            elif event.type == pygame.MOUSEBUTTONDOWN:

                mouse_x, mouse_y = event.pos

                if mouse_y < GRID_SIZE * CELL_SIZE:

                    cell_x = mouse_x // CELL_SIZE

                    cell_y = (
                        GRID_SIZE
                        - 1
                        - mouse_y // CELL_SIZE
                    )

                    clicked_position = (
                        cell_x,
                        cell_y
                    )

                    if (
                        valid_position(
                            clicked_position,
                            obstacles
                        )
                        and clicked_position != robot_position
                    ):

                        goal = clicked_position

                        trail = [robot_position]
                        steps = 0
                        autonomous = False

                        message = f"New target: {goal}"

        # ----------------------------------------------------
        # AUTONOMOUS MOVEMENT
        # ----------------------------------------------------

        if (
            autonomous
            and robot_position != goal
            and current_time - last_auto_move > 350
        ):

            perform_ai_step()

            last_auto_move = current_time

        # ----------------------------------------------------
        # DRAW THE ENVIRONMENT
        # ----------------------------------------------------

        screen.fill((245, 245, 245))

        # Draw the grid and obstacles.
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):

                screen_x = x * CELL_SIZE

                screen_y = (
                    GRID_SIZE - 1 - y
                ) * CELL_SIZE

                rect = pygame.Rect(
                    screen_x,
                    screen_y,
                    CELL_SIZE,
                    CELL_SIZE
                )

                pygame.draw.rect(
                    screen,
                    (210, 210, 210),
                    rect,
                    1
                )

                if (x, y) in obstacles:

                    pygame.draw.rect(
                        screen,
                        (70, 70, 70),
                        rect.inflate(-8, -8),
                        border_radius=8
                    )

        # Draw the robot's trail.
        for x, y in trail:

            center = (
                x * CELL_SIZE + CELL_SIZE // 2,
                (GRID_SIZE - 1 - y)
                * CELL_SIZE
                + CELL_SIZE // 2
            )

            pygame.draw.circle(
                screen,
                (170, 210, 255),
                center,
                10
            )

        # Draw the target.
        goal_center = (
            goal[0] * CELL_SIZE + CELL_SIZE // 2,
            (GRID_SIZE - 1 - goal[1])
            * CELL_SIZE
            + CELL_SIZE // 2
        )

        pygame.draw.circle(
            screen,
            (220, 60, 60),
            goal_center,
            25
        )

        # Draw the robot.
        robot_center = (
            robot_position[0] * CELL_SIZE
            + CELL_SIZE // 2,

            (GRID_SIZE - 1 - robot_position[1])
            * CELL_SIZE
            + CELL_SIZE // 2
        )

        pygame.draw.circle(
            screen,
            (40, 110, 230),
            robot_center,
            22
        )

        # Draw the status panel.
        panel_y = GRID_SIZE * CELL_SIZE

        pygame.draw.rect(
            screen,
            (230, 230, 230),
            (
                0,
                panel_y,
                WINDOW_WIDTH,
                PANEL_HEIGHT
            )
        )

        status = font.render(
            f"Robot: {robot_position}  "
            f"Target: {goal}  "
            f"Steps: {steps}",
            True,
            (20, 20, 20)
        )

        screen.blit(
            status,
            (10, panel_y + 10)
        )

        instruction = small_font.render(
            message,
            True,
            (30, 30, 30)
        )

        screen.blit(
            instruction,
            (10, panel_y + 40)
        )

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()


# ============================================================
# 10. START THE PROGRAM
# ============================================================

if __name__ == "__main__":

    robot_model = train_robot()

    run_simulation(robot_model)
