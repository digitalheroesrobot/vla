
import os
import random
from collections import deque

import numpy as np
import pygame
import torch
import torch.nn as nn


# ============================================================
# 1. CONFIGURATION
# ============================================================

GRID_SIZE = 10
OBSTACLE_PROBABILITY = 0.15
MAX_STEPS = 100

ACTIONS = {
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

# Resolve the checkpoint relative to this Python file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "imitation_learning_v2",
    "imitation_model_v2.pth",
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. V2 MODEL ARCHITECTURE
# Must match ImitationCNNV2 in imitation_learning.py.
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + residual

        return self.relu(x)


class ImitationCNNV2(nn.Module):

    def __init__(self):

        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 10 * 10, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 4),
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.residual_blocks(x)

        x = self.classifier(x)

        return x


# ============================================================
# 3. LOAD V2 CHECKPOINT
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"\nV2 checkpoint not found:\n{MODEL_PATH}\n\n"
            "Train the V2 model first by running:\n"
            "python imitation_learning.py"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = ImitationCNNV2().to(DEVICE)

    # V2 saves a checkpoint dictionary.
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    print("\nV2 imitation model loaded.")
    print("Checkpoint:", MODEL_PATH)
    print("Device:", DEVICE)

    return model


# ============================================================
# 4. ENVIRONMENT GENERATION
# ============================================================

def neighbors(position):

    x, y = position

    for action, (dx, dy) in ACTION_DELTAS.items():

        nx = x + dx
        ny = y + dy

        if (
            0 <= nx < GRID_SIZE
            and 0 <= ny < GRID_SIZE
        ):
            yield (nx, ny), action


def bfs_path(start, target, obstacles):

    queue = deque([(start, [])])

    visited = {start}

    while queue:

        current, path = queue.popleft()

        if current == target:
            return path

        for next_position, action in neighbors(current):

            if next_position in obstacles:
                continue

            if next_position in visited:
                continue

            visited.add(next_position)

            queue.append(
                (
                    next_position,
                    path + [action],
                )
            )

    return None


def generate_environment():

    while True:

        robot_start = (
            random.randrange(GRID_SIZE),
            random.randrange(GRID_SIZE),
        )

        target = (
            random.randrange(GRID_SIZE),
            random.randrange(GRID_SIZE),
        )

        if robot_start == target:
            continue

        obstacles = set()

        for x in range(GRID_SIZE):

            for y in range(GRID_SIZE):

                position = (x, y)

                if position in (robot_start, target):
                    continue

                if random.random() < OBSTACLE_PROBABILITY:
                    obstacles.add(position)

        path = bfs_path(
            robot_start,
            target,
            obstacles,
        )

        if path is not None and len(path) > 0:

            return robot_start, target, obstacles


# ============================================================
# 5. V2 GRID REPRESENTATION
#
# Channel 0: Robot
# Channel 1: Target
# Channel 2: Obstacles
#
# Shape: (3, 10, 10)
# ============================================================

def render_grid(robot, target, obstacles):

    grid = np.zeros(
        (3, GRID_SIZE, GRID_SIZE),
        dtype=np.uint8,
    )

    rx, ry = robot
    tx, ty = target

    grid[0, ry, rx] = 1
    grid[1, ty, tx] = 1

    for x, y in obstacles:

        grid[2, y, x] = 1

    return grid


# ============================================================
# 6. EXPERT POLICY
# ============================================================

def expert_policy(robot, target, obstacles):

    path = bfs_path(
        robot,
        target,
        obstacles,
    )

    if path is None or len(path) == 0:
        return None

    return path[0]


# ============================================================
# 7. LEARNED V2 POLICY
# ============================================================

def learned_policy(model, robot, target, obstacles):

    grid = render_grid(
        robot,
        target,
        obstacles,
    )

    # NumPy shape: (3, 10, 10)
    # Add batch dimension: (1, 3, 10, 10)
    grid_tensor = torch.from_numpy(
        grid.astype(np.float32)
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model(grid_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

    action = prediction.item()

    confidence = confidence.item()

    return action, confidence


# ============================================================
# 8. MOVEMENT AND COLLISION CHECKS
# ============================================================

def move_robot(robot, action, obstacles):

    dx, dy = ACTION_DELTAS[action]

    new_position = (
        robot[0] + dx,
        robot[1] + dy,
    )

    x, y = new_position

    # Boundary check.
    if not (
        0 <= x < GRID_SIZE
        and 0 <= y < GRID_SIZE
    ):
        return robot, False

    # Obstacle check.
    if new_position in obstacles:
        return robot, False

    return new_position, True


# ============================================================
# 9. PYGAME INITIALIZATION
# ============================================================

pygame.init()

PANEL_WIDTH = 500
PANEL_HEIGHT = 590

WINDOW_WIDTH = PANEL_WIDTH * 2
WINDOW_HEIGHT = 730

screen = pygame.display.set_mode(
    (WINDOW_WIDTH, WINDOW_HEIGHT)
)

pygame.display.set_caption(
    "Imitation Learning V2: Expert vs Learned Robot"
)

font = pygame.font.SysFont("Arial", 19)
small_font = pygame.font.SysFont("Arial", 15)
title_font = pygame.font.SysFont(
    "Arial",
    24,
    bold=True,
)

clock = pygame.time.Clock()


# ============================================================
# 10. DRAWING HELPERS
# ============================================================

def draw_text(
    text,
    x,
    y,
    selected_font=font,
    color=(20, 20, 20),
):

    surface = selected_font.render(
        str(text),
        True,
        color,
    )

    screen.blit(surface, (x, y))


def draw_agent_panel(
    offset_x,
    title,
    robot,
    target,
    obstacles,
    action,
    confidence,
    steps,
    blocked_moves,
    status,
    is_expert,
):

    pygame.draw.rect(
        screen,
        (240, 240, 240),
        (offset_x, 0, PANEL_WIDTH, PANEL_HEIGHT),
    )

    draw_text(
        title,
        offset_x + 20,
        15,
        title_font,
    )

    # Grid drawing.
    board_size = 400
    cell = board_size // GRID_SIZE

    board_x = offset_x + 50
    board_y = 65

    for x in range(GRID_SIZE):

        for y in range(GRID_SIZE):

            position = (x, y)

            rect = pygame.Rect(
                board_x + x * cell,
                board_y + y * cell,
                cell,
                cell,
            )

            color = (245, 245, 245)

            if position in obstacles:
                color = (45, 45, 45)

            if position == target:
                color = (230, 65, 65)

            if position == robot:
                color = (40, 110, 240)

            pygame.draw.rect(
                screen,
                color,
                rect,
            )

            pygame.draw.rect(
                screen,
                (210, 210, 210),
                rect,
                1,
            )

    # Information.
    info_y = 490

    draw_text(
        f"Robot position: {robot}",
        offset_x + 20,
        info_y,
    )

    draw_text(
        f"Target position: {target}",
        offset_x + 20,
        info_y + 25,
    )

    draw_text(
        f"Last action: {action}",
        offset_x + 20,
        info_y + 50,
    )

    if not is_expert:

        draw_text(
            f"Model confidence: {confidence:.1%}",
            offset_x + 20,
            info_y + 75,
        )

    draw_text(
        f"Steps: {steps}",
        offset_x + 20,
        info_y + 100,
    )

    draw_text(
        f"Blocked moves: {blocked_moves}",
        offset_x + 20,
        info_y + 125,
    )

    draw_text(
        f"Status: {status}",
        offset_x + 20,
        info_y + 150,
    )


# ============================================================
# 11. SIMULATION STATE
# ============================================================

model = load_model()

robot_start, target, obstacles = generate_environment()

expert_robot = robot_start
learned_robot = robot_start

expert_steps = 0
learned_steps = 0

expert_blocked = 0
learned_blocked = 0

expert_action_name = "NOT STARTED"
learned_action_name = "NOT STARTED"

learned_confidence = 0.0

expert_status = "RUNNING"
learned_status = "RUNNING"

automatic = False

last_message = "Press SPACE to advance one step."


# ============================================================
# 12. RESET SIMULATION
# ============================================================

def reset_simulation():

    global robot_start
    global target
    global obstacles

    global expert_robot
    global learned_robot

    global expert_steps
    global learned_steps

    global expert_blocked
    global learned_blocked

    global expert_action_name
    global learned_action_name
    global learned_confidence

    global expert_status
    global learned_status

    global last_message

    robot_start, target, obstacles = generate_environment()

    expert_robot = robot_start
    learned_robot = robot_start

    expert_steps = 0
    learned_steps = 0

    expert_blocked = 0
    learned_blocked = 0

    expert_action_name = "NOT STARTED"
    learned_action_name = "NOT STARTED"

    learned_confidence = 0.0

    expert_status = "RUNNING"
    learned_status = "RUNNING"

    last_message = "New environment generated."


# ============================================================
# 13. ADVANCE ONE SIMULATION STEP
# ============================================================

def simulation_step():

    global expert_robot
    global learned_robot

    global expert_steps
    global learned_steps

    global expert_blocked
    global learned_blocked

    global expert_action_name
    global learned_action_name
    global learned_confidence

    global expert_status
    global learned_status

    global last_message

    # --------------------------------------------------------
    # EXPERT ROBOT
    # --------------------------------------------------------

    if expert_status == "RUNNING":

        if expert_robot == target:

            expert_status = "SUCCESS"
            expert_action_name = "TARGET REACHED"

        elif expert_steps >= MAX_STEPS:

            expert_status = "TIME LIMIT"

        else:

            action = expert_policy(
                expert_robot,
                target,
                obstacles,
            )

            if action is None:

                expert_status = "NO PATH"

            else:

                expert_action_name = ACTIONS[action]

                expert_robot, moved = move_robot(
                    expert_robot,
                    action,
                    obstacles,
                )

                expert_steps += 1

                if not moved:
                    expert_blocked += 1

                if expert_robot == target:
                    expert_status = "SUCCESS"

    # --------------------------------------------------------
    # LEARNED ROBOT
    # --------------------------------------------------------

    if learned_status == "RUNNING":

        if learned_robot == target:

            learned_status = "SUCCESS"
            learned_action_name = "TARGET REACHED"

        elif learned_steps >= MAX_STEPS:

            learned_status = "TIME LIMIT"

        else:

            action, confidence = learned_policy(
                model,
                learned_robot,
                target,
                obstacles,
            )

            learned_confidence = confidence

            learned_action_name = ACTIONS[action]

            learned_robot, moved = move_robot(
                learned_robot,
                action,
                obstacles,
            )

            learned_steps += 1

            if not moved:

                learned_blocked += 1

                learned_action_name += " (BLOCKED)"

            if learned_robot == target:
                learned_status = "SUCCESS"

    last_message = (
        f"Expert: {expert_status} | "
        f"Learned: {learned_status}"
    )


# ============================================================
# 14. MAIN PYGAME LOOP
# ============================================================

running = True

last_auto_step = 0
AUTO_STEP_DELAY = 350

while running:

    current_time = pygame.time.get_ticks()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:

                running = False

            elif event.key == pygame.K_SPACE:

                automatic = False

                simulation_step()

            elif event.key == pygame.K_a:

                automatic = not automatic

            elif event.key == pygame.K_r:

                automatic = False

                reset_simulation()

            elif event.key == pygame.K_n:

                automatic = False

                reset_simulation()

    # Automatic simulation.
    if automatic:

        if current_time - last_auto_step >= AUTO_STEP_DELAY:

            simulation_step()

            last_auto_step = current_time

            if (
                expert_status != "RUNNING"
                and learned_status != "RUNNING"
            ):
                automatic = False

    # Draw background.
    screen.fill((255, 255, 255))

    # Expert panel.
    draw_agent_panel(
        0,
        "EXPERT ROBOT (BFS)",
        expert_robot,
        target,
        obstacles,
        expert_action_name,
        1.0,
        expert_steps,
        expert_blocked,
        expert_status,
        True,
    )

    # Learned panel.
    draw_agent_panel(
        PANEL_WIDTH,
        "LEARNED ROBOT (V2 CNN)",
        learned_robot,
        target,
        obstacles,
        learned_action_name,
        learned_confidence,
        learned_steps,
        learned_blocked,
        learned_status,
        False,
    )

    # Bottom controls.
    pygame.draw.rect(
        screen,
        (220, 220, 220),
        (0, PANEL_HEIGHT, WINDOW_WIDTH, 140),
    )

    draw_text(
        "SPACE: One step    A: Auto play    R/N: New environment    ESC: Quit",
        20,
        600,
    )

    draw_text(
        "V2 input: 3 x 10 x 10 grid | Actions: UP, DOWN, LEFT, RIGHT",
        20,
        630,
        small_font,
    )

    draw_text(
        last_message,
        20,
        660,
        small_font,
    )

    pygame.display.flip()

    clock.tick(30)


pygame.quit()
