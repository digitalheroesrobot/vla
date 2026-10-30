
"""
V4: INTERACTIVE LANGUAGE-CONDITIONED ROBOT GUI

Loads:
    data/language_conditioning_v4/language_conditioned_policy_v4.pth

CONTROLS:
    1       Select RED object task
    2       Select BLUE object task
    3       Select LEFT side task
    4       Select RIGHT side task

    Q       Change wording (same task)
    SPACE   Play / Pause automatic navigation
    RIGHT   Execute one action
    R       Reset robot
    N       Generate new environment
    ESC     Quit

The robot uses the trained neural network directly.
BFS is NOT used to control the robot.
"""

import os
import re
import random

import numpy as np
import pygame
import torch
import torch.nn as nn


# ============================================================
# 1. CONFIGURATION
# ============================================================

GRID_SIZE = 10
NUM_CHANNELS = 4

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 850

FPS = 60

CELL_SIZE = 40
GRID_X = 75
GRID_Y = 180

MODEL_PATH = os.path.join(
    "data",
    "language_conditioning_v4",
    "language_conditioned_policy_v4.pth",
)

ACTION_NAMES = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "STOP",
]

ACTION_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "STOP": (0, 0),
}

TASK_INSTRUCTIONS = {
    "RED": [
        "move to the red object",
        "go to the red object",
        "reach the red object",
    ],

    "BLUE": [
        "move to the blue object",
        "go to the blue object",
        "reach the blue object",
    ],

    "LEFT": [
        "go to the left side",
        "move to the left side",
        "reach the left side",
    ],

    "RIGHT": [
        "go to the right side",
        "move to the right side",
        "reach the right side",
    ],
}

TASK_COLORS = {
    "RED": (225, 65, 65),
    "BLUE": (55, 125, 245),
    "LEFT": (35, 165, 95),
    "RIGHT": (155, 85, 220),
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_STEPS = 100
AUTO_STEP_INTERVAL = 450


# ============================================================
# 2. LANGUAGE PROCESSING
# ============================================================

def tokenize(text):
    """Convert a sentence into lowercase word tokens."""

    return re.findall(
        r"[a-z]+",
        text.lower(),
    )


def encode_instruction(
    text,
    vocabulary,
    max_length,
):
    """
    Convert a sentence into token IDs.

    Returns:
        token_ids: (1, max_length)
        lengths:   (1,)
    """

    tokens = tokenize(text)

    token_ids = [
        vocabulary.get(
            word,
            vocabulary["<UNK>"],
        )
        for word in tokens
    ]

    length = min(
        len(token_ids),
        max_length,
    )

    token_ids = token_ids[:max_length]

    token_ids += [
        vocabulary["<PAD>"]
    ] * (max_length - len(token_ids))

    token_tensor = torch.tensor(
        token_ids,
        dtype=torch.long,
    ).unsqueeze(0)

    length_tensor = torch.tensor(
        [length],
        dtype=torch.long,
    )

    return token_tensor, length_tensor


# ============================================================
# 3. MODEL ARCHITECTURE
# ============================================================

class LanguageConditionedPolicy(nn.Module):
    """
    This architecture MUST match language_conditioning.py.

    Visual branch:
        Grid -> CNN -> 64 features

    Language branch:
        Tokens -> Embeddings -> GRU -> 64 features

    Fusion:
        Visual + Language -> Policy -> Action
    """

    def __init__(
        self,
        vocabulary_size,
        embedding_dim=32,
        language_hidden=64,
        visual_features=64,
    ):
        super().__init__()

        # -----------------------------
        # Visual encoder
        # -----------------------------

        self.visual_encoder = nn.Sequential(

            nn.Conv2d(
                NUM_CHANNELS,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(
                64,
                visual_features,
            ),

            nn.ReLU(),
        )

        # -----------------------------
        # Language encoder
        # -----------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.language_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=language_hidden,
            batch_first=True,
        )

        # -----------------------------
        # Multimodal fusion
        # -----------------------------

        self.policy_head = nn.Sequential(

            nn.Linear(
                visual_features + language_hidden,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                len(ACTION_NAMES),
            ),
        )

    def forward(
        self,
        observation,
        instruction,
        lengths,
    ):

        visual_features = self.visual_encoder(
            observation
        )

        embedded_words = self.embedding(
            instruction
        )

        packed_words = nn.utils.rnn.pack_padded_sequence(
            embedded_words,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.language_encoder(
            packed_words
        )

        language_features = hidden[-1]

        combined_features = torch.cat(
            [
                visual_features,
                language_features,
            ],
            dim=1,
        )

        return self.policy_head(
            combined_features
        )


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run language_conditioning.py first."
        )

    try:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

    except TypeError:

        # Compatibility with older PyTorch.
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )

    vocabulary = checkpoint["vocabulary"]

    model = LanguageConditionedPolicy(
        vocabulary_size=len(vocabulary),
        embedding_dim=checkpoint["embedding_dim"],
        language_hidden=checkpoint["language_hidden"],
        visual_features=checkpoint["visual_features"],
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    max_text_length = checkpoint["max_text_length"]

    print("\nMODEL LOADED SUCCESSFULLY")
    print("=" * 50)

    print("Checkpoint:", MODEL_PATH)
    print("Device:", DEVICE)

    print(
        "Training test accuracy:",
        f"{checkpoint['test_accuracy']:.2f}%",
    )

    print("Vocabulary size:", len(vocabulary))

    return (
        model,
        vocabulary,
        max_text_length,
    )


# ============================================================
# 5. ENVIRONMENT GENERATION
# ============================================================

def generate_environment():

    while True:

        obstacles = np.zeros(
            (GRID_SIZE, GRID_SIZE),
            dtype=np.float32,
        )

        # Generate random obstacles.
        for y in range(GRID_SIZE):

            for x in range(GRID_SIZE):

                if random.random() < 0.12:

                    obstacles[y, x] = 1.0

        free_positions = [

            (x, y)

            for y in range(GRID_SIZE)

            for x in range(GRID_SIZE)

            if obstacles[y, x] == 0
        ]

        if len(free_positions) < 20:
            continue

        red_position, blue_position = random.sample(
            free_positions,
            2,
        )

        candidates = [

            position

            for position in free_positions

            if position not in (
                red_position,
                blue_position,
            )
        ]

        # Choose a starting position near the center.
        candidates.sort(
            key=lambda p: (
                abs(p[0] - GRID_SIZE // 2)
                + abs(p[1] - GRID_SIZE // 2)
            )
        )

        start_position = random.choice(
            candidates[:8]
        )

        return {
            "obstacles": obstacles,
            "red": red_position,
            "blue": blue_position,
            "start": start_position,
        }


# ============================================================
# 6. TASK GOALS
# ============================================================

def get_side_goal(environment, side):

    obstacles = environment["obstacles"]

    if side == "LEFT":

        target_x = 1

    else:

        target_x = GRID_SIZE - 2

    occupied = {
        environment["red"],
        environment["blue"],
    }

    candidates = []

    for y in range(GRID_SIZE):

        position = (
            target_x,
            y,
        )

        if (
            obstacles[y, target_x] == 0
            and position not in occupied
        ):

            candidates.append(position)

    if not candidates:

        for y in range(GRID_SIZE):

            for x in range(GRID_SIZE):

                position = (
                    x,
                    y,
                )

                if (
                    obstacles[y, x] == 0
                    and position not in occupied
                ):

                    candidates.append(position)

    return min(
        candidates,
        key=lambda position: abs(
            position[1] - GRID_SIZE // 2
        ),
    )


def get_goal(environment, task):

    if task == "RED":

        return environment["red"]

    elif task == "BLUE":

        return environment["blue"]

    elif task == "LEFT":

        return get_side_goal(
            environment,
            "LEFT",
        )

    elif task == "RIGHT":

        return get_side_goal(
            environment,
            "RIGHT",
        )

    raise ValueError(
        f"Unknown task: {task}"
    )


# ============================================================
# 7. VISUAL OBSERVATION
# ============================================================

def build_observation(
    environment,
    robot_position,
):
    """
    Observation shape: (4, 10, 10)

    Channel 0: Robot
    Channel 1: Red object
    Channel 2: Blue object
    Channel 3: Obstacles
    """

    observation = np.zeros(
        (
            NUM_CHANNELS,
            GRID_SIZE,
            GRID_SIZE,
        ),
        dtype=np.float32,
    )

    # Robot.
    x, y = robot_position

    observation[0, y, x] = 1.0

    # Red object.
    red_x, red_y = environment["red"]

    observation[1, red_y, red_x] = 1.0

    # Blue object.
    blue_x, blue_y = environment["blue"]

    observation[2, blue_y, blue_x] = 1.0

    # Obstacles.
    observation[3] = environment["obstacles"]

    return observation


# ============================================================
# 8. MODEL PREDICTION
# ============================================================

@torch.no_grad()
def predict_action(
    model,
    vocabulary,
    max_text_length,
    environment,
    robot_position,
    instruction_text,
):

    observation = build_observation(
        environment,
        robot_position,
    )

    observation_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    instruction_tensor, lengths = encode_instruction(
        instruction_text,
        vocabulary,
        max_text_length,
    )

    instruction_tensor = instruction_tensor.to(
        DEVICE
    )

    logits = model(
        observation_tensor,
        instruction_tensor,
        lengths,
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    action_id = probabilities.argmax().item()

    confidence = probabilities[action_id].item()

    return (
        ACTION_NAMES[action_id],
        confidence,
        probabilities.cpu().numpy(),
    )


# ============================================================
# 9. PYGAME INITIALIZATION
# ============================================================

pygame.init()

screen = pygame.display.set_mode(
    (
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
    )
)

pygame.display.set_caption(
    "V4 - Language-Conditioned Robot"
)

clock = pygame.time.Clock()

FONT_TITLE = pygame.font.SysFont(
    "arial",
    30,
    bold=True,
)

FONT_LARGE = pygame.font.SysFont(
    "arial",
    23,
    bold=True,
)

FONT_MEDIUM = pygame.font.SysFont(
    "arial",
    18,
)

FONT_SMALL = pygame.font.SysFont(
    "arial",
    15,
)

FONT_TINY = pygame.font.SysFont(
    "arial",
    13,
)

WHITE = (255, 255, 255)
BACKGROUND = (245, 247, 250)
DARK = (35, 42, 52)
GRAY = (135, 143, 153)
LIGHT_GRAY = (220, 224, 230)
GREEN = (35, 170, 95)
RED = (225, 65, 65)
BLUE = (55, 125, 245)
YELLOW = (250, 195, 65)


# ============================================================
# 10. DRAWING HELPERS
# ============================================================

def draw_text(
    text,
    x,
    y,
    font,
    color=DARK,
):

    surface = font.render(
        str(text),
        True,
        color,
    )

    screen.blit(
        surface,
        (x, y),
    )


def draw_panel(
    rect,
    title,
):

    pygame.draw.rect(
        screen,
        WHITE,
        rect,
        border_radius=12,
    )

    pygame.draw.rect(
        screen,
        LIGHT_GRAY,
        rect,
        width=2,
        border_radius=12,
    )

    draw_text(
        title,
        rect.x + 16,
        rect.y + 12,
        FONT_LARGE,
    )


# ============================================================
# 11. DRAW ROBOT ENVIRONMENT
# ============================================================

def draw_environment(
    environment,
    robot_position,
    goal,
):

    obstacles = environment["obstacles"]

    # Grid background.
    grid_rect = pygame.Rect(
        GRID_X - 3,
        GRID_Y - 3,
        GRID_SIZE * CELL_SIZE + 6,
        GRID_SIZE * CELL_SIZE + 6,
    )

    pygame.draw.rect(
        screen,
        (250, 250, 250),
        grid_rect,
        border_radius=6,
    )

    # Draw cells and obstacles.
    for y in range(GRID_SIZE):

        for x in range(GRID_SIZE):

            cell = pygame.Rect(
                GRID_X + x * CELL_SIZE,
                GRID_Y + y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )

            pygame.draw.rect(
                screen,
                (238, 241, 245),
                cell,
            )

            pygame.draw.rect(
                screen,
                (210, 216, 224),
                cell,
                width=1,
            )

            if obstacles[y, x] == 1:

                pygame.draw.rect(
                    screen,
                    (95, 103, 115),
                    cell.inflate(-4, -4),
                    border_radius=4,
                )

    # Red object.
    rx, ry = environment["red"]

    red_center = (
        GRID_X + rx * CELL_SIZE + CELL_SIZE // 2,
        GRID_Y + ry * CELL_SIZE + CELL_SIZE // 2,
    )

    pygame.draw.circle(
        screen,
        RED,
        red_center,
        14,
    )

    draw_text(
        "R",
        red_center[0] - 5,
        red_center[1] - 9,
        FONT_SMALL,
        WHITE,
    )

    # Blue object.
    bx, by = environment["blue"]

    blue_center = (
        GRID_X + bx * CELL_SIZE + CELL_SIZE // 2,
        GRID_Y + by * CELL_SIZE + CELL_SIZE // 2,
    )

    pygame.draw.circle(
        screen,
        BLUE,
        blue_center,
        14,
    )

    draw_text(
        "B",
        blue_center[0] - 5,
        blue_center[1] - 9,
        FONT_SMALL,
        WHITE,
    )

    # Goal marker.
    gx, gy = goal

    pygame.draw.rect(
        screen,
        YELLOW,
        (
            GRID_X + gx * CELL_SIZE + 5,
            GRID_Y + gy * CELL_SIZE + 5,
            CELL_SIZE - 10,
            CELL_SIZE - 10,
        ),
        width=3,
        border_radius=5,
    )

    # Robot.
    px, py = robot_position

    robot_center = (
        GRID_X + px * CELL_SIZE + CELL_SIZE // 2,
        GRID_Y + py * CELL_SIZE + CELL_SIZE // 2,
    )

    pygame.draw.circle(
        screen,
        (40, 45, 55),
        robot_center,
        16,
    )

    pygame.draw.circle(
        screen,
        (100, 220, 150),
        robot_center,
        11,
    )


# ============================================================
# 12. DRAW LANGUAGE PANEL
# ============================================================

def draw_language_panel(
    task,
    instruction,
    phrase_index,
):

    rect = pygame.Rect(
        620,
        110,
        620,
        210,
    )

    draw_panel(
        rect,
        "2. HUMAN LANGUAGE INSTRUCTION",
    )

    draw_text(
        f"Selected task: {task}",
        645,
        165,
        FONT_MEDIUM,
        TASK_COLORS[task],
    )

    draw_text(
        "Human says:",
        645,
        205,
        FONT_MEDIUM,
        GRAY,
    )

    pygame.draw.rect(
        screen,
        (239, 243, 250),
        (640, 235, 575, 55),
        border_radius=8,
    )

    draw_text(
        f'"{instruction}"',
        655,
        250,
        FONT_LARGE,
        DARK,
    )

    draw_text(
        f"Wording {phrase_index + 1}/3 | Press Q to change wording",
        645,
        295,
        FONT_SMALL,
        GRAY,
    )


# ============================================================
# 13. DRAW MODEL DECISION PANEL
# ============================================================

def draw_decision_panel(
    action,
    confidence,
    probabilities,
    step_count,
    status,
):

    rect = pygame.Rect(
        620,
        345,
        620,
        270,
    )

    draw_panel(
        rect,
        "3. ROBOT'S DECISION",
    )

    draw_text(
        "Predicted action:",
        650,
        405,
        FONT_MEDIUM,
        GRAY,
    )

    if status == "SUCCESS":

        action_color = GREEN

    elif status in (
        "COLLISION",
        "STOPPED",
        "MAX STEPS",
    ):

        action_color = RED

    else:

        action_color = DARK

    draw_text(
        action,
        850,
        398,
        FONT_TITLE,
        action_color,
    )

    draw_text(
        f"Confidence: {confidence * 100:.1f}%",
        650,
        445,
        FONT_MEDIUM,
    )

    draw_text(
        f"Navigation steps: {step_count}",
        650,
        480,
        FONT_MEDIUM,
    )

    draw_text(
        f"Status: {status}",
        650,
        515,
        FONT_MEDIUM,
        action_color,
    )

    draw_text(
        "Action probabilities:",
        650,
        550,
        FONT_SMALL,
        GRAY,
    )

    bar_x = 650
    bar_y = 578
    bar_width = 510

    for index, name in enumerate(ACTION_NAMES):

        value = float(probabilities[index])

        width = int(
            bar_width * value
        )

        color = (
            GREEN
            if name == action
            else (190, 198, 208)
        )

        pygame.draw.rect(
            screen,
            (232, 235, 240),
            (
                bar_x,
                bar_y + index * 7,
                bar_width,
                5,
            ),
            border_radius=2,
        )

        pygame.draw.rect(
            screen,
            color,
            (
                bar_x,
                bar_y + index * 7,
                width,
                5,
            ),
            border_radius=2,
        )


# ============================================================
# 14. DRAW CONTROLS AND EXPLANATION
# ============================================================

def draw_controls():

    draw_text(
        "1-4: Select task",
        55,
        635,
        FONT_SMALL,
    )

    draw_text(
        "Q: Change wording",
        250,
        635,
        FONT_SMALL,
    )

    draw_text(
        "SPACE: Play/Pause",
        445,
        635,
        FONT_SMALL,
    )

    draw_text(
        "RIGHT: One step",
        650,
        635,
        FONT_SMALL,
    )

    draw_text(
        "R: Reset",
        830,
        635,
        FONT_SMALL,
    )

    draw_text(
        "N: New map",
        960,
        635,
        FONT_SMALL,
    )

    draw_text(
        "ESC: Quit",
        1100,
        635,
        FONT_SMALL,
    )


def draw_explanation():

    rect = pygame.Rect(
        30,
        680,
        1190,
        125,
    )

    draw_panel(
        rect,
        "4. WHAT IS THE ROBOT LEARNING?",
    )

    draw_text(
        "CNN: understands the visual grid and object positions.",
        55,
        730,
        FONT_SMALL,
    )

    draw_text(
        "Embedding + GRU: converts the instruction into language features.",
        55,
        755,
        FONT_SMALL,
    )

    draw_text(
        "Fusion: combines visual and language features before predicting an action.",
        55,
        780,
        FONT_SMALL,
    )


# ============================================================
# 15. ROBOT GUI APPLICATION
# ============================================================

class RobotGUI:

    def __init__(
        self,
        model,
        vocabulary,
        max_text_length,
    ):

        self.model = model
        self.vocabulary = vocabulary
        self.max_text_length = max_text_length

        self.environment = generate_environment()

        self.task = "RED"
        self.phrase_index = 0

        self.instruction = TASK_INSTRUCTIONS[
            self.task
        ][self.phrase_index]

        self.goal = get_goal(
            self.environment,
            self.task,
        )

        self.robot_position = self.environment["start"]

        self.action = "NOT EVALUATED"

        self.confidence = 0.0

        self.probabilities = np.zeros(
            len(ACTION_NAMES),
            dtype=np.float32,
        )

        self.step_count = 0

        # Remember positions visited during this navigation episode.
        self.visited_positions = {
            self.robot_position
        }   

        self.status = "READY"

        self.running = False

        self.last_step_time = 0

    # --------------------------------------------------------
    # Reset robot
    # --------------------------------------------------------

    def reset_robot(self):

        self.robot_position = self.environment["start"]

        self.step_count = 0

        self.visited_positions = {
            self.robot_position
        }

        self.action = "NOT EVALUATED"

        self.confidence = 0.0

        self.probabilities = np.zeros(
            len(ACTION_NAMES),
            dtype=np.float32,
        )

        self.status = "READY"

        self.running = False

    # --------------------------------------------------------
    # Change task
    # --------------------------------------------------------

    def change_task(self, task):

        self.task = task

        self.phrase_index = 0

        self.instruction = TASK_INSTRUCTIONS[
            task
        ][0]

        self.goal = get_goal(
            self.environment,
            task,
        )

        # Same environment, same starting position.
        # Only the requested task changes.

        self.reset_robot()

    # --------------------------------------------------------
    # Change wording
    # --------------------------------------------------------

    def change_wording(self):

        self.phrase_index = (
            self.phrase_index + 1
        ) % len(TASK_INSTRUCTIONS[self.task])

        self.instruction = TASK_INSTRUCTIONS[
            self.task
        ][self.phrase_index]

        # Keep the robot and environment unchanged.
        # Clear the previous prediction.

        self.action = "NOT EVALUATED"

        self.confidence = 0.0

        self.probabilities = np.zeros(
            len(ACTION_NAMES),
            dtype=np.float32,
        )

        self.status = "WORDING CHANGED"

        self.running = False

    # --------------------------------------------------------
    # Generate new map
    # --------------------------------------------------------

    def new_environment(self):

        self.environment = generate_environment()

        self.goal = get_goal(
            self.environment,
            self.task,
        )

        self.reset_robot()

    # --------------------------------------------------------
    # Execute one learned action
    # --------------------------------------------------------

    def take_step(self):

        terminal_states = (
            "SUCCESS",
            "COLLISION",
            "STOPPED",
            "MAX STEPS",
            "STUCK",
        )

        if self.status in terminal_states:
            self.running = False
            return

        # --------------------------------------------------------
        # SAFETY CHECK 1: Has the robot already reached its goal?
        # --------------------------------------------------------

        if self.robot_position == self.goal:

            self.action = "STOP"
            self.confidence = 1.0
            self.status = "SUCCESS"
            self.running = False

            return

        # --------------------------------------------------------
        # 1. Get the neural network prediction
        # --------------------------------------------------------

        (
            predicted_action,
            predicted_confidence,
            probabilities,
        ) = predict_action(
            self.model,
            self.vocabulary,
            self.max_text_length,
            self.environment,
            self.robot_position,
            self.instruction,
        )

        self.probabilities = probabilities

        # --------------------------------------------------------
        # 2. Rank movement actions by model probability
        # --------------------------------------------------------

        movement_actions = [
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
        ]

        ranked_actions = sorted(
            movement_actions,
            key=lambda action: probabilities[
                ACTION_NAMES.index(action)
            ],
            reverse=True,
        )

        selected_action = None
        selected_position = None

        x, y = self.robot_position

        # --------------------------------------------------------
        # 3. Find the best legal action that avoids loops
        # --------------------------------------------------------

        for action in ranked_actions:

            dx, dy = ACTION_DELTAS[action]

            next_position = (
                x + dx,
                y + dy,
            )

            nx, ny = next_position

            # Check grid boundaries.
            if not (
                0 <= nx < GRID_SIZE
                and 0 <= ny < GRID_SIZE
            ):
                continue

            # Check obstacles.
            if self.environment["obstacles"][ny, nx] == 1:
                continue

            # Allow reaching the goal even if it was visited.
            if next_position == self.goal:

                selected_action = action
                selected_position = next_position

                break

            # Avoid revisiting previously occupied positions.
            if next_position in self.visited_positions:
                continue

            # Select the highest-probability valid action.
            selected_action = action
            selected_position = next_position

            break

        # --------------------------------------------------------
        # 4. No useful legal movement remains
        # --------------------------------------------------------

        if selected_action is None:

            self.action = "NO VALID MOVE"

            self.status = "STUCK"

            self.running = False

            return

        # --------------------------------------------------------
        # 5. Execute selected action
        # --------------------------------------------------------

        self.action = selected_action

        self.confidence = float(
            probabilities[
                ACTION_NAMES.index(selected_action)
            ]
        )

        self.robot_position = selected_position

        self.visited_positions.add(
            self.robot_position
        )

        self.step_count += 1

        # --------------------------------------------------------
        # 6. Check goal completion
        # --------------------------------------------------------

        if self.robot_position == self.goal:

            self.action = "STOP"

            self.status = "SUCCESS"

            self.running = False

        elif self.step_count >= MAX_STEPS:

            self.status = "MAX STEPS"

            self.running = False

        else:

            self.status = "RUNNING"

    # --------------------------------------------------------
    # Draw complete GUI
    # --------------------------------------------------------

    def draw(self):

        screen.fill(BACKGROUND)

        # Main title.
        draw_text(
            "V4: LANGUAGE-CONDITIONED ROBOT",
            35,
            25,
            FONT_TITLE,
        )

        draw_text(
            "Same environment. Different human instructions.",
            650,
            38,
            FONT_MEDIUM,
            (110, 75, 180),
        )

        # Environment panel.
        environment_rect = pygame.Rect(
            30,
            110,
            560,
            500,
        )

        draw_panel(
            environment_rect,
            "1. ROBOT ENVIRONMENT",
        )

        draw_environment(
            self.environment,
            self.robot_position,
            self.goal,
        )

        draw_text(
            f"Start: {self.environment['start']}",
            55,
            585,
            FONT_SMALL,
            GRAY,
        )

        draw_text(
            f"Goal: {self.goal}",
            270,
            585,
            FONT_SMALL,
            GRAY,
        )

        # Language panel.
        draw_language_panel(
            self.task,
            self.instruction,
            self.phrase_index,
        )

        # Decision panel.
        draw_decision_panel(
            self.action,
            self.confidence,
            self.probabilities,
            self.step_count,
            self.status,
        )

        # Controls and explanation.
        draw_controls()

        draw_explanation()

        pygame.display.flip()


# ============================================================
# 16. MAIN EVENT LOOP
# ============================================================

def main():

    print("\n" + "=" * 60)

    print("V4 LANGUAGE-CONDITIONED ROBOT GUI")

    print("=" * 60)

    model, vocabulary, max_text_length = load_model()

    app = RobotGUI(
        model,
        vocabulary,
        max_text_length,
    )

    running = True

    while running:

        clock.tick(FPS)

        # ----------------------------------------------------
        # Handle keyboard events
        # ----------------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:

                    running = False

                # Select tasks.
                elif event.key == pygame.K_1:

                    app.change_task("RED")

                elif event.key == pygame.K_2:

                    app.change_task("BLUE")

                elif event.key == pygame.K_3:

                    app.change_task("LEFT")

                elif event.key == pygame.K_4:

                    app.change_task("RIGHT")

                # Change wording.
                elif event.key == pygame.K_q:

                    app.change_wording()

                # Play / pause.
                elif event.key == pygame.K_SPACE:

                    terminal_states = (
    "SUCCESS",
    "COLLISION",
    "STOPPED",
    "MAX STEPS",
    "STUCK",
)

                    if app.status in terminal_states:

                        app.reset_robot()

                    app.running = not app.running

                    if app.running:

                        app.status = "RUNNING"

                # Execute one action.
                elif event.key == pygame.K_RIGHT:

                    app.running = False

                    app.take_step()

                # Reset.
                elif event.key == pygame.K_r:

                    app.reset_robot()

                # New environment.
                elif event.key == pygame.K_n:

                    app.new_environment()

        # ----------------------------------------------------
        # Automatic navigation
        # ----------------------------------------------------

        current_time = pygame.time.get_ticks()

        if (
            app.running
            and current_time - app.last_step_time
            >= AUTO_STEP_INTERVAL
        ):

            app.take_step()

            app.last_step_time = current_time

        # ----------------------------------------------------
        # Draw GUI
        # ----------------------------------------------------

        app.draw()

    pygame.quit()


if __name__ == "__main__":

    main()
