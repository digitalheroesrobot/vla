"""Let's build a small visual robot simulator!
Yes—we can absolutely simulate this logic in a lightweight environment without needing a complex robotics platform.

For your current level, I recommend Pygame. It lets us build a small 2D grid-world where you can see the robot move under your trained model's control.

We'll build this:
- A 5 × 5 grid.
- A blue robot.
- A red target.
- Your trained PyTorch model controlling the robot.
- A visual trail showing where the robot has travelled.
- Controls to step through the AI's decisions or let it navigate automatically.


| Control | Behavior |
|---|---|
| `SPACE` | Ask the AI to make one movement |
| `A` | Toggle automatic AI navigation |
| `R` | Reset the robot to its starting position |
| Click a grid cell | Move the target to that cell |
| `ESC` | Exit |

"""


import pygame
import torch
from torch import nn

# ==========================================
# 1. SETTINGS
# ==========================================

torch.manual_seed(42)

GRID_SIZE = 5
CELL_SIZE = 100
PANEL_HEIGHT = 100

WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE + PANEL_HEIGHT

FPS = 60
AUTO_STEP_DELAY = 400  # milliseconds between AI movements

X_LABELS = ["left", "stay", "right"]
Y_LABELS = ["down", "stay", "up"]

# Robot and target positions use (x, y).
robot_start = (0, 0)
target_start = (4, 3)


# ==========================================
# 2. GENERATE TRAINING DATA
# ==========================================

examples = []

for robot_x in range(GRID_SIZE):
    for robot_y in range(GRID_SIZE):
        for target_x in range(GRID_SIZE):
            for target_y in range(GRID_SIZE):

                if target_x < robot_x:
                    action_x = 0
                elif target_x > robot_x:
                    action_x = 2
                else:
                    action_x = 1

                if target_y < robot_y:
                    action_y = 0
                elif target_y > robot_y:
                    action_y = 2
                else:
                    action_y = 1

                examples.append((
                    [robot_x, robot_y, target_x, target_y],
                    [action_x, action_y]
                ))

train_x = torch.tensor(
    [item[0] for item in examples],
    dtype=torch.float32
)

train_y = torch.tensor(
    [item[1] for item in examples],
    dtype=torch.long
)


# ==========================================
# 3. DEFINE YOUR ROBOT MODEL
# ==========================================

class RobotModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU()
        )

        self.x_head = nn.Linear(16, 3)
        self.y_head = nn.Linear(16, 3)

    def forward(self, inputs):
        features = self.shared(inputs)

        x_scores = self.x_head(features)
        y_scores = self.y_head(features)

        return x_scores, y_scores


model = RobotModel()

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# ==========================================
# 4. TRAIN THE MODEL
# ==========================================

print("Training robot brain...")

model.train()

for epoch in range(500):
    x_scores, y_scores = model(train_x)

    loss_x = loss_function(
        x_scores, train_y[:, 0]
    )

    loss_y = loss_function(
        y_scores, train_y[:, 1]
    )

    total_loss = loss_x + loss_y

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"loss={total_loss.item():.4f}"
        )

model.eval()

print("Robot brain ready!")


# ==========================================
# 5. AI ACTION PREDICTION
# ==========================================

def predict_actions(robot_pos, target_pos):
    rx, ry = robot_pos
    tx, ty = target_pos

    inputs = torch.tensor(
        [[rx, ry, tx, ty]],
        dtype=torch.float32
    )

    with torch.no_grad():
        x_scores, y_scores = model(inputs)

        action_x = x_scores.argmax(dim=1).item()
        action_y = y_scores.argmax(dim=1).item()

    return action_x, action_y


# ==========================================
# 6. EXECUTE THE PREDICTED ACTIONS
# ==========================================

def move_robot(robot_pos, target_pos):
    rx, ry = robot_pos

    action_x, action_y = predict_actions(
        robot_pos, target_pos
    )

    # Execute horizontal movement.
    if action_x == 0:
        rx -= 1
    elif action_x == 2:
        rx += 1

    # Execute vertical movement.
    if action_y == 0:
        ry -= 1
    elif action_y == 2:
        ry += 1

    # Enforce the grid boundaries.
    rx = max(0, min(GRID_SIZE - 1, rx))
    ry = max(0, min(GRID_SIZE - 1, ry))

    new_position = (rx, ry)

    return new_position, action_x, action_y


# ==========================================
# 7. INITIALIZE THE VISUAL ENVIRONMENT
# ==========================================

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("My AI Robot Simulator")

clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 25)
small_font = pygame.font.SysFont(None, 20)

robot_pos = robot_start
target_pos = target_start

trail = [robot_pos]

auto_mode = False
last_auto_step = 0

last_action = "Ready"
steps = 0


# ==========================================
# 8. DRAWING FUNCTIONS
# ==========================================

def cell_center(x, y):
    # Screen Y increases downward, so invert grid Y.
    screen_x = x * CELL_SIZE + CELL_SIZE // 2
    screen_y = (
        (GRID_SIZE - 1 - y) * CELL_SIZE
        + CELL_SIZE // 2
    )

    return screen_x, screen_y


def draw_grid():
    screen.fill((245, 247, 250))

    # Grid lines.
    for i in range(GRID_SIZE + 1):
        pygame.draw.line(
            screen,
            (190, 195, 205),
            (i * CELL_SIZE, 0),
            (i * CELL_SIZE, GRID_SIZE * CELL_SIZE),
            2
        )

        pygame.draw.line(
            screen,
            (190, 195, 205),
            (0, i * CELL_SIZE),
            (WIDTH, i * CELL_SIZE),
            2
        )

    # Coordinate labels.
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            cx, cy = cell_center(x, y)

            label = small_font.render(
                f"{x},{y}", True, (180, 180, 185)
            )

            screen.blit(label, (cx - 16, cy + 28))


def draw_trail():
    for pos in trail:
        cx, cy = cell_center(*pos)

        pygame.draw.circle(
            screen,
            (180, 215, 245),
            (cx, cy),
            12
        )


def draw_target():
    cx, cy = cell_center(*target_pos)

    # Red target.
    pygame.draw.circle(
        screen, (220, 65, 65), (cx, cy), 24
    )

    pygame.draw.circle(
        screen, (255, 255, 255), (cx, cy), 10
    )

    label = font.render(
        "T", True, (255, 255, 255)
    )

    screen.blit(
        label,
        label.get_rect(center=(cx, cy))
    )


def draw_robot():
    cx, cy = cell_center(*robot_pos)

    # Blue robot body.
    pygame.draw.circle(
        screen, (45, 115, 220), (cx, cy), 22
    )

    pygame.draw.circle(
        screen, (255, 255, 255), (cx, cy), 8
    )


def draw_panel():
    panel_y = GRID_SIZE * CELL_SIZE

    pygame.draw.rect(
        screen,
        (230, 234, 242),
        (0, panel_y, WIDTH, PANEL_HEIGHT)
    )

    status = (
        f"Robot: {robot_pos}   "
        f"Target: {target_pos}   "
        f"Steps: {steps}"
    )

    screen.blit(
        font.render(status, True, (35, 40, 50)),
        (10, panel_y + 10)
    )

    if robot_pos == target_pos:
        message = "TARGET REACHED!"
        color = (20, 140, 70)
    else:
        message = (
            f"AI action: {last_action}   |   "
            f"Auto: {'ON' if auto_mode else 'OFF'}"
        )
        color = (35, 40, 50)

    screen.blit(
        font.render(message, True, color),
        (10, panel_y + 37)
    )

    controls = "SPACE: step | A: auto | R: reset | Click: target"

    screen.blit(
        small_font.render(
            controls, True, (70, 75, 85)
        ),
        (10, panel_y + 68)
    )


def draw_scene():
    draw_grid()
    draw_trail()
    draw_target()
    draw_robot()
    draw_panel()

    pygame.display.flip()


# ==========================================
# 9. RESET AND AI CONTROL
# ==========================================

def reset_robot():
    global robot_pos, trail, steps, last_action

    robot_pos = robot_start
    trail = [robot_pos]
    steps = 0
    last_action = "Ready"


def ai_step():
    global robot_pos, trail, steps, last_action

    if robot_pos == target_pos:
        last_action = "Target reached"
        return

    new_pos, action_x, action_y = move_robot(
        robot_pos, target_pos
    )

    # Record the AI's predicted actions.
    last_action = (
        f"X={X_LABELS[action_x]}, "
        f"Y={Y_LABELS[action_y]}"
    )

    # Update the simulated robot.
    robot_pos = new_pos
    trail.append(robot_pos)
    steps += 1

    print(
        f"Step {steps}: position={robot_pos}, "
        f"action=({X_LABELS[action_x]}, "
        f"{Y_LABELS[action_y]})"
    )

    if robot_pos == target_pos:
        print("Target reached!")


# ==========================================
# 10. MAIN SIMULATION LOOP
# ==========================================

running = True

while running:
    now = pygame.time.get_ticks()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_SPACE:
                ai_step()

            elif event.key == pygame.K_a:
                auto_mode = not auto_mode

            elif event.key == pygame.K_r:
                auto_mode = False
                reset_robot()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos

            if my < GRID_SIZE * CELL_SIZE:
                grid_x = mx // CELL_SIZE
                grid_y = (
                    GRID_SIZE - 1
                    - my // CELL_SIZE
                )

                target_pos = (grid_x, grid_y)
                auto_mode = False

                print(f"New target: {target_pos}")

    # Automatic navigation.
    if auto_mode and now - last_auto_step >= AUTO_STEP_DELAY:
        if robot_pos == target_pos:
            auto_mode = False
        else:
            ai_step()
            last_auto_step = now

    draw_scene()
    clock.tick(FPS)

pygame.quit()