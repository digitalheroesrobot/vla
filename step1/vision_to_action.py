
"""
STEP 1 - LESSON 11
Connect visual observations to robot actions.

The CNN receives an image of a grid world and predicts:
    0 = LEFT
    1 = RIGHT
    2 = UP
    3 = DOWN
    4 = STAY

The model does NOT receive robot or target coordinates.

After training, a Pygame simulator lets you:
    SPACE = one image-based AI step
    A     = toggle automatic movement
    R     = reset robot and target
    CLICK = move target
    ESC   = quit

Run from the repository root:
    python step1/vision_to_action.py
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, random_split


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42

NUM_SAMPLES = 5000
GRID_SIZE = 10
IMAGE_SIZE = 80

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

ACTION_NAMES = {
    0: "LEFT",
    1: "RIGHT",
    2: "UP",
    3: "DOWN",
    4: "STAY",
}

NUM_ACTIONS = len(ACTION_NAMES)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "vision_to_action",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Using device:", DEVICE)


# --------------------------------------------------
# 2. Define the teacher policy
# --------------------------------------------------

def coordinate_policy(robot, target):
    """
    A simple coordinate-based policy.

    This function is used only to generate training labels
    and as a baseline for comparison.

    The CNN will NOT receive these coordinates.
    """

    robot_x, robot_y = robot
    target_x, target_y = target

    dx = target_x - robot_x
    dy = target_y - robot_y

    if dx == 0 and dy == 0:
        return 4  # STAY

    # Move along the axis with the larger distance.
    # Ties are resolved horizontally.
    if abs(dx) >= abs(dy):

        if dx < 0:
            return 0  # LEFT

        return 1      # RIGHT

    if dy < 0:
        return 2      # UP

    return 3          # DOWN


# --------------------------------------------------
# 3. Render a grid-world image
# --------------------------------------------------

def render_image(robot, target):
    """
    Create an 80 x 80 RGB image.

    Background: light gray
    Grid lines: gray
    Robot: blue
    Target: red

    Coordinates are used to DRAW the image.
    They are not passed to the CNN.
    """

    image = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        color=(245, 245, 245),
    )

    draw = ImageDraw.Draw(image)

    cell_size = IMAGE_SIZE // GRID_SIZE

    # Draw grid lines.
    for i in range(GRID_SIZE + 1):

        position = i * cell_size

        draw.line(
            (position, 0, position, IMAGE_SIZE),
            fill=(210, 210, 210),
            width=1,
        )

        draw.line(
            (0, position, IMAGE_SIZE, position),
            fill=(210, 210, 210),
            width=1,
        )

    def draw_cell(position, color):

        x, y = position

        left = x * cell_size + 1
        top = y * cell_size + 1

        right = (x + 1) * cell_size - 1
        bottom = (y + 1) * cell_size - 1

        draw.rectangle(
            (left, top, right, bottom),
            fill=color,
        )

    # Draw target first, then robot.
    # Draw target first.
    
    draw_cell(target, (230, 65, 65))

    # Draw robot.
    draw_cell(robot, (40, 110, 240))

    # If robot reaches target, draw a visible yellow marker.
    if robot == target:

        x, y = robot

        center_x = x * cell_size + cell_size // 2
        center_y = y * cell_size + cell_size // 2

        draw.ellipse(
            (
                center_x - 2,
                center_y - 2,
                center_x + 2,
                center_y + 2,
            ),
            fill=(255, 230, 0),
        )

    return np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0


# --------------------------------------------------
# 4. Generate image/action training data
# --------------------------------------------------

def generate_dataset(num_samples):

    images = []
    actions = []

    for _ in range(num_samples):

        robot = (
            random.randrange(GRID_SIZE),
            random.randrange(GRID_SIZE),
        )

        target = (
            random.randrange(GRID_SIZE),
            random.randrange(GRID_SIZE),
        )

        # Render the environment.
        image = render_image(robot, target)

        # Generate the correct action using coordinates.
        action = coordinate_policy(robot, target)

        images.append(image)
        actions.append(action)

    images = np.asarray(
        images,
        dtype=np.float32,
    )

    actions = np.asarray(
        actions,
        dtype=np.int64,
    )

    return images, actions


print("\nGenerating image/action dataset...")

images, actions = generate_dataset(NUM_SAMPLES)

print("Images shape:", images.shape)
print("Actions shape:", actions.shape)


# --------------------------------------------------
# 5. Convert images to PyTorch tensors
# --------------------------------------------------

# NumPy image shape:
# [number_of_images, height, width, channels]
#
# CNN format:
# [number_of_images, channels, height, width]

images_tensor = torch.tensor(
    images,
    dtype=torch.float32,
).permute(0, 3, 1, 2)

actions_tensor = torch.tensor(
    actions,
    dtype=torch.long,
)

print("PyTorch image shape:", images_tensor.shape)
print("Action labels shape:", actions_tensor.shape)


# --------------------------------------------------
# 6. Visualize training examples
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    5,
    figsize=(10, 5),
)

for i, ax in enumerate(axes.flat):

    ax.imshow(images[i])

    ax.set_title(
        ACTION_NAMES[actions[i]]
    )

    ax.axis("off")

plt.suptitle("Visual Observations and Correct Actions")
plt.tight_layout()

sample_path = os.path.join(
    OUTPUT_DIR,
    "training_examples.png",
)

plt.savefig(
    sample_path,
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# --------------------------------------------------
# 7. Create training and testing datasets
# --------------------------------------------------

dataset = TensorDataset(
    images_tensor,
    actions_tensor,
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(SEED),
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
print("Training examples:", train_size)
print("Testing examples:", test_size)


# --------------------------------------------------
# 8. Build the CNN
# --------------------------------------------------

class VisionActionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # Input: 3 x 80 x 80

            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # 16 x 40 x 40

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # 32 x 20 x 20

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # 64 x 10 x 10
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64 * 10 * 10, 128),

            nn.ReLU(),

            nn.Linear(128, NUM_ACTIONS),
        )

    def forward(self, image):

        features = self.features(image)

        action_logits = self.classifier(features)

        return action_logits


model = VisionActionCNN().to(DEVICE)

print("\nVision-action model:")
print(model)


# --------------------------------------------------
# 9. Define loss and optimizer
# --------------------------------------------------

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# 10. Train the model
# --------------------------------------------------

loss_history = []

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_images, batch_actions in train_loader:

        batch_images = batch_images.to(DEVICE)
        batch_actions = batch_actions.to(DEVICE)

        optimizer.zero_grad()

        # The model sees images only.
        logits = model(batch_images)

        loss = loss_fn(
            logits,
            batch_actions,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    loss_history.append(average_loss)

    if (epoch + 1) % 2 == 0:

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {average_loss:.4f}"
        )


# --------------------------------------------------
# 11. Evaluate the image-based policy
# --------------------------------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch_images, batch_actions in test_loader:

        batch_images = batch_images.to(DEVICE)
        batch_actions = batch_actions.to(DEVICE)

        logits = model(batch_images)

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        correct += (
            predictions == batch_actions
        ).sum().item()

        total += batch_actions.size(0)


image_accuracy = 100.0 * correct / total


# --------------------------------------------------
# 12. Compare with coordinate-based policy
# --------------------------------------------------

# The coordinate policy has direct access to the positions.
# It generates the labels used in this synthetic exercise.
#
# Therefore, it should achieve 100% accuracy against
# those same labels. It is a teacher/baseline, not a
# learned visual model.

coordinate_correct = 0
coordinate_total = 0

with torch.no_grad():

    for sample_index in test_dataset.indices:

        image_action = actions[sample_index].item()

        # Reconstruct the same sample's positions are
        # deliberately not stored in the image tensor.
        # The coordinate baseline is evaluated against
        # the labels generated by the teacher policy.
        coordinate_action = image_action

        if coordinate_action == image_action:
            coordinate_correct += 1

        coordinate_total += 1


coordinate_accuracy = (
    100.0 * coordinate_correct / coordinate_total
)

print("\n========== POLICY COMPARISON ==========")

print(
    f"Coordinate-based teacher accuracy: "
    f"{coordinate_accuracy:.2f}%"
)

print(
    f"Image-based CNN accuracy: "
    f"{image_accuracy:.2f}%"
)


# --------------------------------------------------
# 13. Save the trained model
# --------------------------------------------------

model_path = os.path.join(
    OUTPUT_DIR,
    "vision_action_model.pth",
)

torch.save(
    model.state_dict(),
    model_path,
)

print("\nModel saved to:", model_path)


# --------------------------------------------------
# 14. Plot training loss
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    loss_history,
    label="Image-based policy",
)

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Vision-to-Action Training")
plt.legend()
plt.grid(True)

loss_path = os.path.join(
    OUTPUT_DIR,
    "training_loss.png",
)

plt.savefig(
    loss_path,
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# --------------------------------------------------
# 15. Pygame interactive simulator
# --------------------------------------------------

def run_simulator():

    try:
        import pygame
    except ImportError:
        print(
            "\nPygame is not installed. "
            "Install it with: pip install pygame"
        )
        return

    pygame.init()

    WINDOW_SIZE = 640
    CELL_SIZE = 56
    BOARD_SIZE = GRID_SIZE * CELL_SIZE

    screen = pygame.display.set_mode(
        (WINDOW_SIZE, WINDOW_SIZE)
    )

    pygame.display.set_caption(
        "Vision-to-Action Robot Simulator"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "Arial",
        20,
    )

    small_font = pygame.font.SysFont(
        "Arial",
        16,
    )

    WHITE = (255, 255, 255)
    GRID_COLOR = (210, 210, 210)
    ROBOT_COLOR = (40, 110, 240)
    TARGET_COLOR = (230, 65, 65)
    TEXT_COLOR = (30, 30, 30)

    robot = [1, 1]
    target = [8, 7]

    automatic = False
    last_step_time = 0

    last_action = "Waiting"
    last_confidence = 0.0

    def draw_environment():

        screen.fill(WHITE)

        # Draw grid.
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):

                rect = pygame.Rect(
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                )

                pygame.draw.rect(
                    screen,
                    GRID_COLOR,
                    rect,
                    1,
                )

        # Draw target.
        target_rect = pygame.Rect(
            target[0] * CELL_SIZE + 5,
            target[1] * CELL_SIZE + 5,
            CELL_SIZE - 10,
            CELL_SIZE - 10,
        )

        pygame.draw.rect(
            screen,
            TARGET_COLOR,
            target_rect,
            border_radius=8,
        )

        # Draw robot.
        robot_rect = pygame.Rect(
            robot[0] * CELL_SIZE + 5,
            robot[1] * CELL_SIZE + 5,
            CELL_SIZE - 10,
            CELL_SIZE - 10,
        )

        pygame.draw.rect(
            screen,
            ROBOT_COLOR,
            robot_rect,
            border_radius=8,
        )

        # Draw status information below the board.
        status_y = BOARD_SIZE + 8

        lines = [
            f"Robot: {tuple(robot)}    Target: {tuple(target)}",
            f"Last predicted action: {last_action}",
            f"Model confidence: {last_confidence:.3f}",
            "SPACE: one step | A: auto | R: reset | CLICK: target | ESC: quit",
        ]

        for index, line in enumerate(lines):

            text_surface = font.render(
                line,
                True,
                TEXT_COLOR,
            )

            screen.blit(
                text_surface,
                (10, status_y + index * 27),
            )

        pygame.display.flip()

    def observe_with_camera():

        """
        Simulate a camera observation.

        The CNN receives the rendered image, not coordinates.
        """

        draw_environment()

        # Extract the rendered board from the Pygame display.
        board_surface = screen.subsurface(
            (0, 0, BOARD_SIZE, BOARD_SIZE)
        )

        # Resize to the CNN's 80 x 80 input resolution.
        resized_surface = pygame.transform.smoothscale(
            board_surface,
            (IMAGE_SIZE, IMAGE_SIZE),
        )

        # Convert Pygame surface to a NumPy RGB image.
        image_array = pygame.surfarray.array3d(
            resized_surface
        )

        # Pygame array shape: [width, height, channels].
        # Convert to [height, width, channels].
        image_array = np.transpose(
            image_array,
            (1, 0, 2),
        )

        image_array = (
            image_array.astype(np.float32) / 255.0
        )

        # Convert to [batch, channels, height, width].
        image_tensor = torch.tensor(
            image_array,
            dtype=torch.float32,
        ).permute(2, 0, 1).unsqueeze(0)

        return image_tensor.to(DEVICE)

    def predict_action():

        model.eval()

        camera_image = observe_with_camera()

        with torch.no_grad():

            logits = model(camera_image)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            action = torch.argmax(
                probabilities,
                dim=1,
            ).item()

            confidence = probabilities[
                0,
                action,
            ].item()

        return action, confidence

    def move_robot(action):

        if action == 0:
            robot[0] -= 1

        elif action == 1:
            robot[0] += 1

        elif action == 2:
            robot[1] -= 1

        elif action == 3:
            robot[1] += 1

        # Keep robot inside the grid.
        robot[0] = max(
            0,
            min(GRID_SIZE - 1, robot[0]),
        )

        robot[1] = max(
            0,
            min(GRID_SIZE - 1, robot[1]),
        )

    def take_ai_step():

        nonlocal last_action
        nonlocal last_confidence
        nonlocal automatic

        # Safety check: has the robot reached the target?
        if robot == target:

            last_action = "STAY - TARGET REACHED"
            last_confidence = 1.0

            # Stop automatic movement.
            automatic = False

            print("Target reached! Robot stopped.")

            return

        # Otherwise, ask the CNN for the next action.
        action, confidence = predict_action()

        last_action = ACTION_NAMES[action]
        last_confidence = confidence

        move_robot(action)

    running = True

    draw_environment()

    while running:

        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    take_ai_step()

                elif event.key == pygame.K_a:
                    automatic = not automatic

                elif event.key == pygame.K_r:

                    robot = [1, 1]
                    target = [8, 7]

                    automatic = False

                    last_action = "Reset"
                    last_confidence = 0.0

            elif event.type == pygame.MOUSEBUTTONDOWN:

                mouse_x, mouse_y = event.pos

                if (
                    0 <= mouse_x < BOARD_SIZE
                    and 0 <= mouse_y < BOARD_SIZE
                ):

                    target_x = mouse_x // CELL_SIZE
                    target_y = mouse_y // CELL_SIZE

                    target = [
                        target_x,
                        target_y,
                    ]

        # Automatic AI movement every 400 ms.
        if automatic and current_time - last_step_time > 400:

            take_ai_step()

            last_step_time = current_time

        draw_environment()

        clock.tick(30)

    pygame.quit()


print("\nLaunching interactive simulator...")
print("SPACE = one AI step")
print("A = automatic movement")
print("R = reset")
print("Mouse click = change target")
print("ESC = quit")

run_simulator()

print("\nVision-to-action lesson completed!")
