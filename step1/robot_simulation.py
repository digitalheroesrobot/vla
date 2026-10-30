
"""
Step 10: V4 Language-Conditioned Policy + PyBullet

Improvements:
    1. Robot height and orientation are stabilized.
    2. Robot stops precisely at the target center.
    3. Robot remains stationary after success.
    4. Red and blue targets are randomized independently.
    5. Targets are separated and avoid corner positions.
    6. BFS provides obstacle-aware navigation.
    7. V4 chooses between equally good navigation actions.

Run:
    python robot_simulation.py --task RED --seed 42
    python robot_simulation.py --task RED
    python robot_simulation.py --task BLUE
    python robot_simulation.py --task LEFT
    python robot_simulation.py --task RIGHT
    python robot_simulation.py --seed 123
    python robot_simulation.py --no-gui
"""

import argparse
import math
import random
import re
import time
from collections import deque

import numpy as np
import pybullet as p
import pybullet_data

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


# ============================================================
# 1. CONFIGURATION
# ============================================================

GRID_SIZE = 10

CELL_SIZE = 0.25
WORLD_SIZE = GRID_SIZE * CELL_SIZE

ROBOT_RADIUS = 0.045
ROBOT_HEIGHT = 0.12

# Keep the robot slightly above the floor.
ROBOT_Z = ROBOT_HEIGHT / 2 + 0.005

PHYSICS_HZ = 240
PHYSICS_DT = 1.0 / PHYSICS_HZ

ACTION_TIMEOUT = 2.0
POSITION_TOLERANCE = 0.015

KP = 2.5
MAX_SPEED = 0.35

MAX_POLICY_STEPS = 100

# Targets should not be adjacent.
MIN_TARGET_SEPARATION = 4

CHECKPOINT_PATH = (
    "data/language_conditioning_v4/"
    "language_conditioned_policy_v4.pth"
)

ACTION_NAMES = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "STOP",
]

# Grid coordinates: (column, row)
ACTION_DELTAS = {
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0),
}

INSTRUCTIONS = {
    "RED": "move to the red object",
    "BLUE": "move to the blue object",
    "LEFT": "go to the left side",
    "RIGHT": "go to the right side",
}


# ============================================================
# 2. LANGUAGE PROCESSING
# ============================================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def encode_instruction(text, vocabulary, max_length):

    tokens = tokenize(text)

    token_ids = [
        vocabulary.get(token, vocabulary["<UNK>"])
        for token in tokens
    ]

    token_ids = token_ids[:max_length]

    length = max(1, len(token_ids))

    while len(token_ids) < max_length:
        token_ids.append(vocabulary["<PAD>"])

    return (
        torch.tensor([token_ids], dtype=torch.long),
        torch.tensor([length], dtype=torch.long),
    )


# ============================================================
# 3. V4 MODEL
# ============================================================

class LanguageConditionedPolicy(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=32,
        language_hidden=64,
        visual_features=64,
        num_actions=5,
    ):
        super().__init__()

        # Must remain identical to the V4 training architecture.

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(
                4, 32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(64, visual_features),
            nn.ReLU(),
        )

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.language_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=language_hidden,
            batch_first=True,
        )

        self.policy_head = nn.Sequential(
            nn.Linear(
                visual_features + language_hidden,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_actions),
        )

    def forward(self, observation, tokens, lengths):

        visual_features = self.visual_encoder(
            observation
        )

        embedded = self.embedding(tokens)

        packed = rnn_utils.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.language_encoder(packed)

        language_features = hidden[-1]

        combined = torch.cat(
            [visual_features, language_features],
            dim=1,
        )

        return self.policy_head(combined)


# ============================================================
# 4. LOAD V4 CHECKPOINT
# ============================================================

def load_policy(device):

    print("\nLoading V4 checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    vocabulary = checkpoint["vocabulary"]

    model = LanguageConditionedPolicy(
        vocab_size=len(vocabulary),
        embedding_dim=checkpoint.get("embedding_dim", 32),
        language_hidden=checkpoint.get("language_hidden", 64),
        visual_features=checkpoint.get("visual_features", 64),
        num_actions=checkpoint.get("num_actions", 5),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    max_text_length = checkpoint["max_text_length"]

    saved_actions = checkpoint.get("action_names")

    if saved_actions is not None:
        if list(saved_actions) != ACTION_NAMES:
            raise ValueError(
                "Checkpoint action ordering does not match "
                "the simulator.\n"
                f"Checkpoint: {saved_actions}\n"
                f"Simulator:  {ACTION_NAMES}"
            )

    print("V4 model loaded successfully.")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Maximum text length: {max_text_length}")

    if "test_accuracy" in checkpoint:
        print(
            f"V4 test accuracy: "
            f"{checkpoint['test_accuracy']:.2f}%"
        )

    return model, vocabulary, max_text_length


# ============================================================
# 5. GRID NAVIGATION
# ============================================================

def get_neighbors(cell, obstacles):

    neighbors = []

    for dx, dy in ACTION_DELTAS.values():

        neighbor = (
            cell[0] + dx,
            cell[1] + dy,
        )

        x, y = neighbor

        if not (0 <= x < GRID_SIZE):
            continue

        if not (0 <= y < GRID_SIZE):
            continue

        if neighbor in obstacles:
            continue

        neighbors.append(neighbor)

    return neighbors


def shortest_path_distances(goal, obstacles):

    distances = {goal: 0}
    queue = deque([goal])

    while queue:

        current = queue.popleft()

        for neighbor in get_neighbors(
            current,
            obstacles,
        ):

            if neighbor in distances:
                continue

            distances[neighbor] = (
                distances[current] + 1
            )

            queue.append(neighbor)

    return distances


# ============================================================
# 6. RANDOM WORLD GENERATION
# ============================================================

def manhattan_distance(a, b):

    return (
        abs(a[0] - b[0])
        + abs(a[1] - b[1])
    )


def generate_world(task, seed=42):

    rng = random.Random(seed)

    all_cells = [
        (x, y)
        for y in range(GRID_SIZE)
        for x in range(GRID_SIZE)
    ]

    # Target candidates are kept away from the outermost
    # grid cells, so objects do not always appear in corners.

    target_cells = [
        (x, y)
        for y in range(1, GRID_SIZE - 1)
        for x in range(1, GRID_SIZE - 1)
    ]

    for _ in range(1000):

        start = (
            rng.randint(1, GRID_SIZE - 2),
            rng.randint(1, GRID_SIZE - 2),
        )

        obstacles = {
            cell
            for cell in all_cells
            if cell != start and rng.random() < 0.10
        }

        available = [
            cell
            for cell in target_cells
            if cell != start
            and cell not in obstacles
        ]

        if len(available) < 2:
            continue

        # Choose two independently randomized target positions.
        # Enforce a minimum distance between them.

        pairs = []

        for red in available:
            for blue in available:

                if red == blue:
                    continue

                if manhattan_distance(
                    red,
                    blue,
                ) < MIN_TARGET_SEPARATION:
                    continue

                pairs.append((red, blue))

        if not pairs:
            continue

        red, blue = rng.choice(pairs)

        if task == "RED":

            goal = red

        elif task == "BLUE":

            goal = blue

        elif task == "LEFT":

            goal = (0, start[1])

        elif task == "RIGHT":

            goal = (GRID_SIZE - 1, start[1])

        else:

            raise ValueError(f"Unknown task: {task}")

        if goal in obstacles:
            continue

        distances = shortest_path_distances(
            goal,
            obstacles,
        )

        if start not in distances:
            continue

        return {
            "start": start,
            "red": red,
            "blue": blue,
            "obstacles": obstacles,
            "goal": goal,
        }

    raise RuntimeError(
        "Could not generate a reachable world."
    )


# ============================================================
# 7. PYBULLET SIMULATOR
# ============================================================

class RobotSimulator:

    def __init__(self, world, gui=True):

        self.world = world
        self.gui = gui

        self.client = p.connect(
            p.GUI if gui else p.DIRECT
        )

        if self.client < 0:
            raise RuntimeError(
                "Could not connect to PyBullet."
            )

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.client,
        )

        p.setGravity(
            0, 0, -9.81,
            physicsClientId=self.client,
        )

        p.setTimeStep(
            PHYSICS_DT,
            physicsClientId=self.client,
        )

        p.setPhysicsEngineParameter(
            fixedTimeStep=PHYSICS_DT,
            numSolverIterations=100,
            physicsClientId=self.client,
        )

        p.loadURDF(
            "plane.urdf",
            physicsClientId=self.client,
        )

        self.locked_position = None

        self.create_world()
        self.create_robot()

        # Allow the robot to settle on the floor.
        for _ in range(30):
            p.stepSimulation(
                physicsClientId=self.client
            )

        self.stabilize_robot()

        if gui:

            p.resetDebugVisualizerCamera(
                cameraDistance=3.8,
                cameraYaw=0,
                cameraPitch=-65,
                cameraTargetPosition=[
                    WORLD_SIZE / 2,
                    WORLD_SIZE / 2,
                    0,
                ],
                physicsClientId=self.client,
            )

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    @staticmethod
    def grid_to_world(cell):

        col, row = cell

        x = (col + 0.5) * CELL_SIZE

        y = (
            GRID_SIZE - row - 0.5
        ) * CELL_SIZE

        return x, y

    @staticmethod
    def world_to_grid(position):

        x, y = position[0], position[1]

        col = int(x / CELL_SIZE)

        row = GRID_SIZE - 1 - int(y / CELL_SIZE)

        col = max(0, min(GRID_SIZE - 1, col))
        row = max(0, min(GRID_SIZE - 1, row))

        return col, row

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    def create_world(self):

        # Grid lines.

        for i in range(GRID_SIZE + 1):

            value = i * CELL_SIZE

            p.addUserDebugLine(
                [value, 0, 0.01],
                [value, WORLD_SIZE, 0.01],
                [0.5, 0.5, 0.5],
                1,
                physicsClientId=self.client,
            )

            p.addUserDebugLine(
                [0, value, 0.01],
                [WORLD_SIZE, value, 0.01],
                [0.5, 0.5, 0.5],
                1,
                physicsClientId=self.client,
            )

        # Obstacles.

        obstacle_half = CELL_SIZE * 0.30

        obstacle_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                obstacle_half,
                obstacle_half,
                0.12,
            ],
            physicsClientId=self.client,
        )

        obstacle_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                obstacle_half,
                obstacle_half,
                0.12,
            ],
            rgbaColor=[0.25, 0.25, 0.30, 1],
            physicsClientId=self.client,
        )

        for cell in self.world["obstacles"]:

            x, y = self.grid_to_world(cell)

            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=obstacle_collision,
                baseVisualShapeIndex=obstacle_visual,
                basePosition=[x, y, 0.12],
                physicsClientId=self.client,
            )

        # Red and blue targets.

        self.create_object(
            self.world["red"],
            [1, 0.05, 0.05, 1],
        )

        self.create_object(
            self.world["blue"],
            [0.05, 0.2, 1, 1],
        )

    def create_object(self, cell, color):

        x, y = self.grid_to_world(cell)

        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.065,
            rgbaColor=color,
            physicsClientId=self.client,
        )

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, 0.08],
            physicsClientId=self.client,
        )

    # --------------------------------------------------------
    # Robot
    # --------------------------------------------------------

    def create_robot(self):

        start_x, start_y = self.grid_to_world(
            self.world["start"]
        )

        collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=ROBOT_RADIUS,
            height=ROBOT_HEIGHT,
            physicsClientId=self.client,
        )

        visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=ROBOT_RADIUS,
            length=ROBOT_HEIGHT,
            rgbaColor=[0.1, 0.8, 0.2, 1],
            physicsClientId=self.client,
        )

        self.robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[
                start_x,
                start_y,
                ROBOT_Z,
            ],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.client,
        )

        p.changeDynamics(
            self.robot_id,
            -1,
            lateralFriction=1.0,
            spinningFriction=0.1,
            rollingFriction=0.1,
            linearDamping=0.95,
            angularDamping=0.99,
            physicsClientId=self.client,
        )

        self.locked_position = (
            start_x,
            start_y,
        )

    # --------------------------------------------------------
    # Robot stabilization
    # --------------------------------------------------------

    def stabilize_robot(self):
        """
        Keep the robot upright and at a fixed height.

        Horizontal movement and collision detection still use
        PyBullet physics. Vertical movement and tipping are
        corrected after every physics step.
        """

        position = self.get_robot_position()

        x, y = position[0], position[1]

        p.resetBasePositionAndOrientation(
            self.robot_id,
            [x, y, ROBOT_Z],
            [0, 0, 0, 1],
            physicsClientId=self.client,
        )

        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=[
                p.getBaseVelocity(
                    self.robot_id,
                    physicsClientId=self.client,
                )[0][0],
                p.getBaseVelocity(
                    self.robot_id,
                    physicsClientId=self.client,
                )[0][1],
                0,
            ],
            angularVelocity=[0, 0, 0],
            physicsClientId=self.client,
        )

    def hold_position(self):
        """
        Lock the robot at its final position and stop all motion.
        """

        if self.locked_position is None:
            return

        x, y = self.locked_position

        p.resetBasePositionAndOrientation(
            self.robot_id,
            [x, y, ROBOT_Z],
            [0, 0, 0, 1],
            physicsClientId=self.client,
        )

        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=self.client,
        )

    def lock_at_cell(self, cell):
        """
        Snap the robot to the exact center of a grid cell.
        Used when it reaches the target.
        """

        x, y = self.grid_to_world(cell)

        self.locked_position = (x, y)

        self.hold_position()

    # --------------------------------------------------------
    # State and observation
    # --------------------------------------------------------

    def get_robot_position(self):

        position, _ = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.client,
        )

        return position

    def get_robot_cell(self):

        return self.world_to_grid(
            self.get_robot_position()
        )

    def get_observation(self):

        observation = np.zeros(
            (4, GRID_SIZE, GRID_SIZE),
            dtype=np.float32,
        )

        col, row = self.get_robot_cell()

        observation[0, row, col] = 1.0

        red_col, red_row = self.world["red"]

        observation[1, red_row, red_col] = 1.0

        blue_col, blue_row = self.world["blue"]

        observation[2, blue_row, blue_col] = 1.0

        for col, row in self.world["obstacles"]:
            observation[3, row, col] = 1.0

        return observation

    # --------------------------------------------------------
    # Legal actions
    # --------------------------------------------------------

    def get_legal_actions(self, current_cell):

        legal = []

        for action_id, (dx, dy) in ACTION_DELTAS.items():

            target = (
                current_cell[0] + dx,
                current_cell[1] + dy,
            )

            x, y = target

            if not (0 <= x < GRID_SIZE):
                continue

            if not (0 <= y < GRID_SIZE):
                continue

            if target in self.world["obstacles"]:
                continue

            legal.append(action_id)

        return legal

    # --------------------------------------------------------
    # Movement
    # --------------------------------------------------------

    def execute_action(self, action_id):

        if action_id == 4:
            self.stop_robot()
            return "STOP"

        current_cell = self.get_robot_cell()

        dx, dy = ACTION_DELTAS[action_id]

        target_cell = (
            current_cell[0] + dx,
            current_cell[1] + dy,
        )

        if target_cell not in get_neighbors(
            current_cell,
            self.world["obstacles"],
        ):
            return "BLOCKED"

        target_x, target_y = self.grid_to_world(
            target_cell
        )

        max_steps = int(
            ACTION_TIMEOUT * PHYSICS_HZ
        )

        for _ in range(max_steps):

            position = self.get_robot_position()

            error_x = target_x - position[0]
            error_y = target_y - position[1]

            distance = math.hypot(
                error_x,
                error_y,
            )

            if distance <= POSITION_TOLERANCE:

                self.stop_robot()

                return "REACHED"

            vx = KP * error_x
            vy = KP * error_y

            speed = math.hypot(vx, vy)

            if speed > MAX_SPEED:

                scale = MAX_SPEED / speed

                vx *= scale
                vy *= scale

            p.resetBaseVelocity(
                self.robot_id,
                linearVelocity=[vx, vy, 0],
                angularVelocity=[0, 0, 0],
                physicsClientId=self.client,
            )

            p.stepSimulation(
                physicsClientId=self.client
            )

            # Prevent falling and tipping.

            self.stabilize_robot()

            if self.gui:
                time.sleep(PHYSICS_DT)

        self.stop_robot()

        final_position = self.get_robot_position()

        final_distance = math.hypot(
            target_x - final_position[0],
            target_y - final_position[1],
        )

        if final_distance <= POSITION_TOLERANCE:
            return "REACHED"

        return "INCOMPLETE"

    def stop_robot(self):

        position = self.get_robot_position()

        self.locked_position = (
            position[0],
            position[1],
        )

        self.hold_position()

    def close(self):

        if p.isConnected(self.client):
            p.disconnect(self.client)


# ============================================================
# 8. GOAL CHECK
# ============================================================

def goal_reached(task, robot_cell, world):

    if task == "RED":
        return robot_cell == world["red"]

    if task == "BLUE":
        return robot_cell == world["blue"]

    if task == "LEFT":
        return robot_cell[0] == 0

    if task == "RIGHT":
        return robot_cell[0] == GRID_SIZE - 1

    return False


# ============================================================
# 9. MODEL INFERENCE
# ============================================================

@torch.no_grad()
def predict_action(
    model,
    vocabulary,
    max_text_length,
    observation,
    instruction,
    device,
):

    observation_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    tokens, lengths = encode_instruction(
        instruction,
        vocabulary,
        max_text_length,
    )

    tokens = tokens.to(device)
    lengths = lengths.to(device)

    logits = model(
        observation_tensor,
        tokens,
        lengths,
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    return probabilities.cpu().numpy()


# ============================================================
# 10. HYBRID ACTION SELECTION
# ============================================================

def select_legal_action(
    probabilities,
    legal_actions,
    current_cell,
    task,
    world,
    distance_map,
):

    if goal_reached(task, current_cell, world):
        return 4

    current_distance = distance_map.get(
        current_cell
    )

    if current_distance is None:
        return None

    candidates = []

    for action_id in legal_actions:

        dx, dy = ACTION_DELTAS[action_id]

        target = (
            current_cell[0] + dx,
            current_cell[1] + dy,
        )

        target_distance = distance_map.get(
            target
        )

        # Only allow moves that reduce shortest-path distance.

        if target_distance != current_distance - 1:
            continue

        score = float(probabilities[action_id])

        candidates.append(
            (score, action_id)
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return candidates[0][1]


# ============================================================
# 11. CLOSED-LOOP SIMULATION
# ============================================================

def run_simulation(task, seed=42, gui=True):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n======================================")
    print(" VLA ROBOT SIMULATION")
    print("======================================")

    print(f"Device: {device}")
    print(f"Task: {task}")
    print(f"Instruction: {INSTRUCTIONS[task]}")

    model, vocabulary, max_text_length = load_policy(
        device
    )

    world = generate_world(
        task,
        seed=seed,
    )

    print("\nWorld:")
    print(f"Start:     {world['start']}")
    print(f"Red:       {world['red']}")
    print(f"Blue:      {world['blue']}")
    print(f"Obstacles: {len(world['obstacles'])}")

    print(
        "Red-blue separation: "
        f"{manhattan_distance(world['red'], world['blue'])} cells"
    )

    print(f"Goal:      {world['goal']}")

    distance_map = shortest_path_distances(
        world["goal"],
        world["obstacles"],
    )

    start_distance = distance_map.get(
        world["start"]
    )

    if start_distance is None:
        raise RuntimeError(
            "Start position cannot reach the goal."
        )

    print(
        f"Shortest path length: {start_distance} moves"
    )

    simulator = RobotSimulator(
        world,
        gui=gui,
    )

    instruction = INSTRUCTIONS[task]

    status = "MAX_STEPS"

    try:

        print("\nStarting closed-loop control...\n")

        for step in range(MAX_POLICY_STEPS):

            # A. OBSERVE

            observation = simulator.get_observation()

            robot_cell = simulator.get_robot_cell()

            if goal_reached(
                task,
                robot_cell,
                world,
            ):

                status = "SUCCESS"
                simulator.lock_at_cell(
                    world["goal"]
                )
                break

            # B. PREDICT

            probabilities = predict_action(
                model,
                vocabulary,
                max_text_length,
                observation,
                instruction,
                device,
            )

            predicted_id = int(
                np.argmax(probabilities)
            )

            predicted_name = ACTION_NAMES[
                predicted_id
            ]

            # C. PLAN

            legal_actions = simulator.get_legal_actions(
                robot_cell
            )

            action_id = select_legal_action(
                probabilities,
                legal_actions,
                robot_cell,
                task,
                world,
                distance_map,
            )

            if action_id is None:

                status = "STUCK"

                print(
                    "No valid shortest-path action found."
                )

                break

            action_name = ACTION_NAMES[action_id]

            confidence = float(
                probabilities[predicted_id]
            )

            print(
                f"[{step + 1:02d}] "
                f"position={robot_cell} | "
                f"model={predicted_name:5s} | "
                f"executed={action_name:5s} | "
                f"confidence={confidence:.2%} | "
                f"distance={distance_map[robot_cell]}"
            )

            if predicted_id != action_id:

                print(
                    f"     Planner override: "
                    f"{predicted_name} -> {action_name}"
                )

            # D. EXECUTE

            result = simulator.execute_action(
                action_id
            )

            new_cell = simulator.get_robot_cell()

            print(
                f"     execution={result} | "
                f"new_position={new_cell}"
            )

            if result == "INCOMPLETE":

                status = "EXECUTION_FAILED"

                print(
                    "     Robot failed to reach the next cell."
                )

                break

            if result == "BLOCKED":

                status = "BLOCKED"
                break

            # E. FEEDBACK

            if goal_reached(
                task,
                new_cell,
                world,
            ):

                status = "SUCCESS"

                # Precisely align the robot with the target.

                simulator.lock_at_cell(
                    world["goal"]
                )

                print(
                    "\nTARGET REACHED! Robot stopped."
                )

                break

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        final_cell = simulator.get_robot_cell()

        print("\n======================================")
        print(" SIMULATION SUMMARY")
        print("======================================")

        print(f"Task:             {task}")
        print(f"Instruction:      {instruction}")
        print(f"Start position:   {world['start']}")
        print(f"Final position:   {final_cell}")
        print(f"Goal position:    {world['goal']}")
        print(f"Status:           {status}")

        print(
            f"Goal reached:     "
            f"{goal_reached(task, final_cell, world)}"
        )

        if status == "SUCCESS":

            simulator.lock_at_cell(
                world["goal"]
            )

            print(
                "Robot is holding its final position."
            )

        # ----------------------------------------------------
        # Keep GUI open without allowing drift.
        # ----------------------------------------------------

        if gui:

            print(
                "\nClose the PyBullet window to finish."
            )

            while p.isConnected(simulator.client):

                if status == "SUCCESS":
                    simulator.hold_position()

                p.stepSimulation(
                    physicsClientId=simulator.client
                )

                simulator.stabilize_robot()

                if status == "SUCCESS":
                    simulator.hold_position()

                time.sleep(PHYSICS_DT)

    finally:

        simulator.close()


# ============================================================
# 12. ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Run V4 hybrid navigation in PyBullet."
    )

    parser.add_argument(
        "--task",
        default="RED",
        choices=["RED", "BLUE", "LEFT", "RIGHT"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
    )

    args = parser.parse_args()

    run_simulation(
        task=args.task,
        seed=args.seed,
        gui=not args.no_gui,
    )


if __name__ == "__main__":
    main()
