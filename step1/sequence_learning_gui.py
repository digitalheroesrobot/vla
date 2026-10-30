
"""
V3: Sequential Imitation Learning GUI
=====================================

Visualizes why a robot needs observation history.

Models:
    1. Single-frame MLP
    2. Vanilla RNN
    3. LSTM
    4. Transformer

Controls:
    SPACE       Play / Pause
    RIGHT       Next timestep
    LEFT        Previous timestep
    R           Restart sequence
    TAB         Switch LEFT / RIGHT scenario
    ESC         Quit

Run:
    python sequence_learning_gui.py


A robot sometimes needs to remember what it saw earlier to decide what it should do now.    
For example, imagine a robot performing a manipulation task:
1. It sees a cup on the table.
2. It receives the instruction to pick up the cup.
3. Its camera view changes as it moves.
4. The cup becomes temporarily invisible.
5. It must still remember what it was trying to accomplish.
"""

import sys
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn as nn


# ============================================================
# 1. CONFIGURATION
# ============================================================

WIDTH = 1250
HEIGHT = 850

FPS = 60

SEQUENCE_LENGTH = 5

OBSERVATION_SIZE = 4
ACTION_SIZE = 4
INPUT_SIZE = 8

HIDDEN_SIZE = 64
NUM_ACTIONS = 4

FORWARD = 0
LEFT = 1
RIGHT = 2
STOP = 3

ACTION_NAMES = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "STOP"
]

OUTPUT_DIR = Path("data/sequence_learning_v3")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "V3 - Sequential Imitation Learning"
)

clock = pygame.time.Clock()


# ============================================================
# 2. COLORS
# ============================================================

WHITE = (245, 245, 245)
BLACK = (30, 30, 30)
GRAY = (150, 150, 150)
LIGHT_GRAY = (220, 220, 220)

BLUE = (40, 110, 240)
RED = (230, 65, 65)
GREEN = (50, 170, 90)
YELLOW = (245, 195, 50)

DARK_BLUE = (25, 55, 100)
PURPLE = (145, 80, 200)
ORANGE = (230, 135, 40)


# ============================================================
# 3. FONTS
# ============================================================

FONT = pygame.font.SysFont(
    "arial",
    20
)

SMALL_FONT = pygame.font.SysFont(
    "arial",
    15
)

MEDIUM_FONT = pygame.font.SysFont(
    "arial",
    25,
    bold=True
)

BIG_FONT = pygame.font.SysFont(
    "arial",
    32,
    bold=True
)


# ============================================================
# 4. MODELS
# ============================================================

class SingleFramePolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(INPUT_SIZE, 64),

            nn.ReLU(),

            nn.Linear(64, 64),

            nn.ReLU(),

            nn.Linear(64, NUM_ACTIONS)
        )

    def forward(self, x):

        return self.network(x)


class RNNPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.rnn = nn.RNN(

            input_size=INPUT_SIZE,

            hidden_size=HIDDEN_SIZE,

            num_layers=1,

            batch_first=True,

            nonlinearity="tanh"
        )

        self.classifier = nn.Linear(
            HIDDEN_SIZE,
            NUM_ACTIONS
        )

    def forward(self, x):

        output, hidden = self.rnn(x)

        return self.classifier(
            output[:, -1, :]
        )


class LSTMPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(

            input_size=INPUT_SIZE,

            hidden_size=HIDDEN_SIZE,

            num_layers=1,

            batch_first=True
        )

        self.classifier = nn.Linear(
            HIDDEN_SIZE,
            NUM_ACTIONS
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        return self.classifier(
            output[:, -1, :]
        )


class TransformerPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_projection = nn.Linear(
            INPUT_SIZE,
            HIDDEN_SIZE
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                SEQUENCE_LENGTH,
                HIDDEN_SIZE
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=HIDDEN_SIZE,

            nhead=4,

            dim_feedforward=128,

            dropout=0.1,

            batch_first=True,

            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=2
        )

        self.classifier = nn.Sequential(

            nn.Linear(HIDDEN_SIZE, 64),

            nn.ReLU(),

            nn.Linear(64, NUM_ACTIONS)
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = x + self.position_embedding

        output = self.transformer(x)

        return self.classifier(
            output[:, -1, :]
        )


# ============================================================
# 5. LOAD TRAINED CHECKPOINTS
# ============================================================

def load_checkpoint(model, filename):

    path = OUTPUT_DIR / filename

    if not path.exists():

        print(f"Checkpoint not found: {path}")

        return None

    try:

        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.to(DEVICE)

        model.eval()

        print(f"Loaded: {filename}")

        return model

    except Exception as error:

        print(f"Could not load {filename}: {error}")

        return None


def load_models():

    models = {

        "Single-frame MLP": load_checkpoint(
            SingleFramePolicy(),
            "single_frame_mlp_v3.pth"
        ),

        "Vanilla RNN": load_checkpoint(
            RNNPolicy(),
            "vanilla_rnn_v3.pth"
        ),

        "LSTM": load_checkpoint(
            LSTMPolicy(),
            "lstm_v3.pth"
        ),

        "Transformer": load_checkpoint(
            TransformerPolicy(),
            "transformer_v3.pth"
        )
    }

    return models


# ============================================================
# 6. GENERATE ROBOT TRAJECTORY
# ============================================================

def generate_trajectory(direction):

    positions = [
        0.00,
        0.25,
        0.50,
        0.75,
        1.00
    ]

    observations = []

    for t in range(SEQUENCE_LENGTH):

        cue_left = 0.0
        cue_right = 0.0

        at_junction = 0.0

        if t == 0:

            if direction == 0:

                cue_left = 1.0

            else:

                cue_right = 1.0

        if t == SEQUENCE_LENGTH - 1:

            at_junction = 1.0

        observations.append([

            positions[t],

            at_junction,

            cue_left,

            cue_right

        ])

    observations = np.array(
        observations,
        dtype=np.float32
    )

    # Expert actions.
    expert_actions = [
        FORWARD,
        FORWARD,
        FORWARD,
        FORWARD
    ]

    if direction == 0:

        expert_actions.append(LEFT)

    else:

        expert_actions.append(RIGHT)

    # Build previous-action history.
    sequence = []

    for t in range(SEQUENCE_LENGTH):

        previous_action = np.zeros(
            ACTION_SIZE,
            dtype=np.float32
        )

        if t > 0:

            previous_action[
                expert_actions[t - 1]
            ] = 1.0

        combined = np.concatenate([

            observations[t],

            previous_action

        ])

        sequence.append(combined)

    return (

        observations,

        expert_actions,

        np.array(
            sequence,
            dtype=np.float32
        )

    )


# ============================================================
# 7. PREDICTIONS
# ============================================================

def predict_models(models, sequence):

    tensor = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    predictions = {}

    with torch.no_grad():

        for name, model in models.items():

            if model is None:

                predictions[name] = None

                continue

            if name == "Single-frame MLP":

                model_input = tensor[:, -1, :]

            else:

                model_input = tensor

            logits = model(model_input)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            action = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0, action
            ].item()

            predictions[name] = {

                "action": ACTION_NAMES[action],

                "confidence": confidence,

                "probabilities": probabilities.cpu().numpy()[0]

            }

    return predictions


# ============================================================
# 8. DRAWING HELPERS
# ============================================================

def draw_text(
    surface,
    text,
    x,
    y,
    font=FONT,
    color=BLACK
):

    rendered = font.render(
        str(text),
        True,
        color
    )

    surface.blit(
        rendered,
        (x, y)
    )


def draw_panel(
    surface,
    rect,
    title,
    background=WHITE
):

    pygame.draw.rect(
        surface,
        background,
        rect,
        border_radius=12
    )

    pygame.draw.rect(
        surface,
        LIGHT_GRAY,
        rect,
        width=2,
        border_radius=12
    )

    draw_text(
        surface,
        title,
        rect.x + 15,
        rect.y + 10,
        MEDIUM_FONT
    )


# ============================================================
# 9. ROBOT ENVIRONMENT
# ============================================================

def draw_environment(
    surface,
    timestep,
    direction
):

    rect = pygame.Rect(
        30,
        80,
        570,
        350
    )

    draw_panel(
        surface,
        rect,
        "1. ROBOT ENVIRONMENT"
    )

    center_y = 275

    start_x = 100
    end_x = 530

    # Corridor.
    pygame.draw.line(
        surface,
        GRAY,
        (start_x, center_y - 55),
        (end_x, center_y - 55),
        5
    )

    pygame.draw.line(
        surface,
        GRAY,
        (start_x, center_y + 55),
        (end_x, center_y + 55),
        5
    )

    # Junction.
    junction_x = 530

    pygame.draw.line(
        surface,
        GRAY,
        (junction_x, center_y - 55),
        (junction_x, center_y - 145),
        5
    )

    pygame.draw.line(
        surface,
        GRAY,
        (junction_x, center_y + 55),
        (junction_x, center_y + 145),
        5
    )

    # Position of the robot.
    robot_x = int(
        start_x + (
            (end_x - start_x)
            * timestep
            / (SEQUENCE_LENGTH - 1)
        )
    )

    robot_y = center_y

    # Robot.
    pygame.draw.circle(
        surface,
        BLUE,
        (robot_x, robot_y),
        22
    )

    pygame.draw.circle(
        surface,
        WHITE,
        (robot_x, robot_y),
        22,
        3
    )

    draw_text(
        surface,
        "ROBOT",
        robot_x - 30,
        robot_y + 30,
        SMALL_FONT
    )

    # Starting clue.
    if timestep == 0:

        clue_text = (
            "TURN LEFT"
            if direction == 0
            else "TURN RIGHT"
        )

        clue_color = (
            GREEN
            if direction == 0
            else ORANGE
        )

        pygame.draw.rect(
            surface,
            clue_color,
            (75, 135, 180, 55),
            border_radius=8
        )

        draw_text(
            surface,
            clue_text,
            92,
            150,
            MEDIUM_FONT,
            WHITE
        )

    else:

        pygame.draw.rect(
            surface,
            LIGHT_GRAY,
            (75, 135, 180, 55),
            border_radius=8
        )

        draw_text(
            surface,
            "CLUE LOST",
            105,
            150,
            MEDIUM_FONT,
            BLACK
        )

    # Junction status.
    if timestep == SEQUENCE_LENGTH - 1:

        draw_text(
            surface,
            "JUNCTION REACHED!",
            365,
            130,
            MEDIUM_FONT,
            RED
        )

    else:

        draw_text(
            surface,
            "Moving forward...",
            370,
            130,
            FONT,
            DARK_BLUE
        )

    draw_text(
        surface,
        f"Timestep: {timestep + 1}/{SEQUENCE_LENGTH}",
        60,
        390,
        FONT
    )


# ============================================================
# 10. OBSERVATION TIMELINE
# ============================================================

def draw_timeline(
    surface,
    observations,
    expert_actions,
    timestep
):

    rect = pygame.Rect(
        620,
        80,
        600,
        350
    )

    draw_panel(
        surface,
        rect,
        "2. OBSERVATION HISTORY"
    )

    labels = [
        "Position",
        "Junction",
        "Left cue",
        "Right cue"
    ]

    for t in range(SEQUENCE_LENGTH):

        card_x = 640 + t * 112

        card_y = 135

        card_rect = pygame.Rect(
            card_x,
            card_y,
            102,
            240
        )

        if t == timestep:

            color = (215, 235, 255)

        elif t < timestep:

            color = (225, 245, 225)

        else:

            color = (235, 235, 235)

        pygame.draw.rect(
            surface,
            color,
            card_rect,
            border_radius=8
        )

        pygame.draw.rect(
            surface,
            GRAY,
            card_rect,
            width=1,
            border_radius=8
        )

        draw_text(
            surface,
            f"TIME {t}",
            card_x + 12,
            card_y + 10,
            SMALL_FONT
        )

        for i, label in enumerate(labels):

            value = observations[t, i]

            if i == 0:

                display_value = f"{value:.2f}"

            else:

                display_value = (
                    "YES" if value == 1 else "NO"
                )

            draw_text(
                surface,
                label,
                card_x + 7,
                card_y + 45 + i * 30,
                SMALL_FONT
            )

            draw_text(
                surface,
                display_value,
                card_x + 7,
                card_y + 63 + i * 30,
                SMALL_FONT,
                BLUE if value else GRAY
            )

        action_name = ACTION_NAMES[
            expert_actions[t]
        ]

        draw_text(
            surface,
            "Expert:",
            card_x + 7,
            card_y + 185,
            SMALL_FONT
        )

        draw_text(
            surface,
            action_name,
            card_x + 7,
            card_y + 207,
            SMALL_FONT,
            GREEN
        )

    draw_text(
        surface,
        "Green = past | Blue = current | Gray = future",
        650,
        395,
        SMALL_FONT
    )


# ============================================================
# 11. CURRENT OBSERVATION VS HISTORY
# ============================================================

def draw_memory_explanation(
    surface,
    timestep
):

    rect = pygame.Rect(
        30,
        455,
        570,
        155
    )

    draw_panel(
        surface,
        rect,
        "3. WHY DOES MEMORY MATTER?"
    )

    if timestep == 0:

        message = [
            "The robot sees a directional clue.",
            "It can observe whether the target is LEFT or RIGHT.",
            "The robot must retain this information."
        ]

    elif timestep < SEQUENCE_LENGTH - 1:

        message = [
            "The directional clue has disappeared.",
            "The current observation no longer reveals",
            "which direction the robot should choose."
        ]

    else:

        message = [
            "The robot has reached the junction.",
            "Both directions have identical current observations.",
            "It must remember the original directional clue!"
        ]

    for i, line in enumerate(message):

        draw_text(
            surface,
            line,
            50,
            500 + i * 28,
            SMALL_FONT
        )


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

def draw_predictions(
    surface,
    predictions,
    timestep,
    direction
):

    rect = pygame.Rect(
        620,
        455,
        600,
        310
    )

    draw_panel(
        surface,
        rect,
        "4. MODEL DECISIONS"
    )

    names = [
        "Single-frame MLP",
        "Vanilla RNN",
        "LSTM",
        "Transformer"
    ]

    y = 510

    for name in names:

        prediction = predictions.get(name)

        draw_text(
            surface,
            name,
            650,
            y,
            FONT
        )

        if prediction is None:

            draw_text(
                surface,
                "Checkpoint unavailable",
                900,
                y,
                SMALL_FONT,
                RED
            )

        elif timestep < SEQUENCE_LENGTH - 1:

            draw_text(
                surface,
                "Waiting for complete sequence...",
                900,
                y,
                SMALL_FONT,
                GRAY
            )

        else:

            action = prediction["action"]

            confidence = prediction["confidence"]

            color = (
                GREEN
                if action == (
                    "LEFT" if direction == 0 else "RIGHT"
                )
                else RED
            )

            draw_text(
                surface,
                action,
                900,
                y,
                FONT,
                color
            )

            draw_text(
                surface,
                f"{confidence * 100:.1f}%",
                1080,
                y,
                FONT
            )

        y += 48

    if timestep == SEQUENCE_LENGTH - 1:

        draw_text(
            surface,
            "Green = correct | Red = incorrect",
            650,
            710,
            SMALL_FONT
        )

    else:

        draw_text(
            surface,
            "Predictions are revealed when the robot reaches the junction.",
            650,
            710,
            SMALL_FONT
        )


# ============================================================
# 13. CONTROLS
# ============================================================

def draw_controls(surface, playing):

    rect = pygame.Rect(
        30,
        635,
        570,
        130
    )

    draw_panel(
        surface,
        rect,
        "5. CONTROLS"
    )

    status = "PLAYING" if playing else "PAUSED"

    draw_text(
        surface,
        f"Status: {status}",
        50,
        680,
        FONT,
        GREEN if playing else RED
    )

    draw_text(
        surface,
        "SPACE: Play/Pause",
        50,
        710,
        SMALL_FONT
    )

    draw_text(
        surface,
        "LEFT/RIGHT: Step backward/forward",
        260,
        680,
        SMALL_FONT
    )

    draw_text(
        surface,
        "R: Restart     TAB: Switch direction",
        260,
        710,
        SMALL_FONT
    )

    draw_text(
        surface,
        "ESC: Quit",
        50,
        740,
        SMALL_FONT
    )


# ============================================================
# 14. MAIN GUI
# ============================================================

def main():

    models = load_models()

    direction = 0

    timestep = 0

    playing = False

    last_update = pygame.time.get_ticks()

    observations, expert_actions, sequence = (
        generate_trajectory(direction)
    )

    predictions = predict_models(
        models,
        sequence
    )

    running = True

    while running:

        clock.tick(FPS)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:

                    running = False

                elif event.key == pygame.K_SPACE:

                    playing = not playing

                    last_update = pygame.time.get_ticks()

                elif event.key == pygame.K_RIGHT:

                    playing = False

                    timestep = min(
                        timestep + 1,
                        SEQUENCE_LENGTH - 1
                    )

                elif event.key == pygame.K_LEFT:

                    playing = False

                    timestep = max(
                        timestep - 1,
                        0
                    )

                elif event.key == pygame.K_r:

                    playing = False

                    timestep = 0

                elif event.key == pygame.K_TAB:

                    playing = False

                    direction = 1 - direction

                    timestep = 0

                    (
                        observations,
                        expert_actions,
                        sequence
                    ) = generate_trajectory(direction)

                    predictions = predict_models(
                        models,
                        sequence
                    )

        # ----------------------------------------------------
        # Automatic playback
        # ----------------------------------------------------

        if playing:

            now = pygame.time.get_ticks()

            if now - last_update > 1100:

                if timestep < SEQUENCE_LENGTH - 1:

                    timestep += 1

                else:

                    playing = False

                last_update = now

        # ----------------------------------------------------
        # Render
        # ----------------------------------------------------

        screen.fill(WHITE)

        draw_text(
            screen,
            "V3: SEQUENTIAL IMITATION LEARNING",
            30,
            20,
            BIG_FONT,
            DARK_BLUE
        )

        direction_text = (
            "LEFT scenario"
            if direction == 0
            else "RIGHT scenario"
        )

        draw_text(
            screen,
            f"Current scenario: {direction_text}",
            870,
            30,
            FONT,
            PURPLE
        )

        draw_environment(
            screen,
            timestep,
            direction
        )

        draw_timeline(
            screen,
            observations,
            expert_actions,
            timestep
        )

        draw_memory_explanation(
            screen,
            timestep
        )

        draw_controls(
            screen,
            playing
        )

        draw_predictions(
            screen,
            predictions,
            timestep,
            direction
        )

        pygame.display.flip()

    pygame.quit()

    sys.exit()


if __name__ == "__main__":

    main()
