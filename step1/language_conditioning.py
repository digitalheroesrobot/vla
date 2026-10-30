
"""
V4: LANGUAGE-CONDITIONED IMITATION LEARNING

A robot learns to combine:
    1. Visual observations of its environment.
    2. Natural-language instructions from a human.

Instructions:
    - Move to the red object.
    - Move to the blue object.
    - Go to the left side.
    - Go to the right side.

Architecture:
    Grid observation -> CNN -> Visual features
    Text instruction -> Tokenization -> Embedding -> GRU -> Language features

    Visual features + Language features -> Policy -> Robot action

Actions:
    UP, DOWN, LEFT, RIGHT, STOP

Visual observation ------------------+
                                      |
                                      v
                                    POLICY ------> Action
                                      ^
                                      |
Language instruction -> Text encoder-+
"""

import os
import random
import re
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

GRID_SIZE = 10
NUM_CHANNELS = 4

NUM_TRAIN_MAPS = 300
NUM_TEST_MAPS = 80

STARTS_PER_MAP = 12

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "data/language_conditioning_v4"

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STOP"]

ACTION_TO_ID = {
    name: index for index, name in enumerate(ACTION_NAMES)
}

# (action name, dx, dy)
DIRECTIONS = [
    ("UP", 0, -1),
    ("DOWN", 0, 1),
    ("LEFT", -1, 0),
    ("RIGHT", 1, 0),
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 2. LANGUAGE: INSTRUCTIONS AND TOKENIZATION
# ============================================================

# Different sentences can express the same task.
#
# All sentences in a group have the same intended meaning.

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


def tokenize(text):
    """
    Convert a sentence into lowercase word tokens.

    Example:
        "Move to the red object"

    Becomes:
        ["move", "to", "the", "red", "object"]
    """
    return re.findall(r"[a-z]+", text.lower())


def build_vocabulary():
    """
    Build a vocabulary from all training instructions.

    PAD = 0: padding token
    UNK = 1: unknown token
    """

    words = set()

    for instructions in TASK_INSTRUCTIONS.values():
        for sentence in instructions:
            words.update(tokenize(sentence))

    vocabulary = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for word in sorted(words):
        vocabulary[word] = len(vocabulary)

    return vocabulary


VOCAB = build_vocabulary()

PAD_ID = VOCAB["<PAD>"]
UNK_ID = VOCAB["<UNK>"]

MAX_TEXT_LENGTH = 8


def encode_instruction(text):
    """
    Convert words into integer token IDs.

    Returns:
        token_ids: padded tensor of token IDs
        length: number of real tokens
    """

    tokens = tokenize(text)

    token_ids = [
        VOCAB.get(word, UNK_ID)
        for word in tokens
    ]

    length = len(token_ids)

    if length > MAX_TEXT_LENGTH:
        token_ids = token_ids[:MAX_TEXT_LENGTH]
        length = MAX_TEXT_LENGTH

    token_ids += [PAD_ID] * (MAX_TEXT_LENGTH - length)

    return torch.tensor(token_ids, dtype=torch.long), length


# ============================================================
# 3. ENVIRONMENT GENERATION
# ============================================================

def generate_map():
    """
    Generate one random grid environment.

    Observation channels:
        0 = robot
        1 = red object
        2 = blue object
        3 = obstacles

    The instruction is NOT encoded into the grid.
    It is provided separately to the policy.
    """

    size = GRID_SIZE

    obstacles = np.zeros((size, size), dtype=np.float32)

    # Random obstacles.
    for y in range(size):
        for x in range(size):
            if random.random() < 0.12:
                obstacles[y, x] = 1.0

    free_positions = [
        (x, y)
        for y in range(size)
        for x in range(size)
        if obstacles[y, x] == 0
    ]

    if len(free_positions) < 20:
        return generate_map()

    red_position, blue_position = random.sample(
        free_positions, 2
    )

    return {
        "obstacles": obstacles,
        "red": red_position,
        "blue": blue_position,
    }


def get_side_goal(environment, side):
    """
    Select a reachable candidate goal near the left or right side.

    The goal is not directly marked in the visual observation.
    The policy must interpret the language instruction.
    """

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
        position = (target_x, y)

        if obstacles[y, target_x] == 0 and position not in occupied:
            candidates.append(position)

    if not candidates:
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                position = (x, y)

                if (
                    obstacles[y, x] == 0
                    and position not in occupied
                ):
                    candidates.append(position)

    # Choose a goal near the middle of the selected side.
    return min(
        candidates,
        key=lambda position: abs(position[1] - GRID_SIZE // 2),
    )


def build_observation(environment, robot_position):
    """
    Construct a four-channel grid observation.

    Shape:
        (4, 10, 10)
    """

    observation = np.zeros(
        (NUM_CHANNELS, GRID_SIZE, GRID_SIZE),
        dtype=np.float32,
    )

    x, y = robot_position
    observation[0, y, x] = 1.0

    red_x, red_y = environment["red"]
    observation[1, red_y, red_x] = 1.0

    blue_x, blue_y = environment["blue"]
    observation[2, blue_y, blue_x] = 1.0

    observation[3] = environment["obstacles"]

    return observation


# ============================================================
# 4. EXPERT POLICY: BREADTH-FIRST SEARCH
# ============================================================

def expert_action(environment, start, goal):
    """
    Find the first action on the shortest path to the goal.

    Returns:
        Action ID: UP, DOWN, LEFT, RIGHT, or STOP.

    This expert generates demonstrations for imitation learning.
    """

    if start == goal:
        return ACTION_TO_ID["STOP"]

    obstacles = environment["obstacles"]

    queue = deque([(start, None)])
    visited = {start}

    while queue:
        position, first_action = queue.popleft()

        x, y = position

        for action_name, dx, dy in DIRECTIONS:

            next_position = (x + dx, y + dy)
            nx, ny = next_position

            if not (
                0 <= nx < GRID_SIZE
                and 0 <= ny < GRID_SIZE
            ):
                continue

            if obstacles[ny, nx] == 1:
                continue

            if next_position in visited:
                continue

            visited.add(next_position)

            if first_action is None:
                next_first_action = ACTION_TO_ID[action_name]
            else:
                next_first_action = first_action

            if next_position == goal:
                return next_first_action

            queue.append(
                (next_position, next_first_action)
            )

    # No valid path exists.
    return None


# ============================================================
# 5. DATASET GENERATION
# ============================================================

def get_task_goal(environment, task):
    if task == "RED":
        return environment["red"]

    if task == "BLUE":
        return environment["blue"]

    if task == "LEFT":
        return get_side_goal(environment, "LEFT")

    if task == "RIGHT":
        return get_side_goal(environment, "RIGHT")

    raise ValueError(f"Unknown task: {task}")


def generate_dataset(number_of_maps):
    """
    Generate demonstrations for all four language tasks.

    Each sample contains:
        - visual observation
        - language instruction
        - expert action

    Maps are independently generated for training and testing.
    """

    observations = []
    instructions = []
    lengths = []
    labels = []
    task_names = []

    for map_index in range(number_of_maps):

        environment = generate_map()

        obstacles = environment["obstacles"]

        free_positions = [
            (x, y)
            for y in range(GRID_SIZE)
            for x in range(GRID_SIZE)
            if (
                obstacles[y, x] == 0
                and (x, y) != environment["red"]
                and (x, y) != environment["blue"]
            )
        ]

        if not free_positions:
            continue

        starts = random.sample(
            free_positions,
            min(STARTS_PER_MAP, len(free_positions)),
        )

        for task in TASK_INSTRUCTIONS:

            goal = get_task_goal(environment, task)

            for start in starts:

                action = expert_action(
                    environment,
                    start,
                    goal,
                )

                if action is None:
                    continue

                # Randomly choose a sentence expressing this task.
                sentence = random.choice(
                    TASK_INSTRUCTIONS[task]
                )

                token_ids, length = encode_instruction(sentence)

                observation = build_observation(
                    environment,
                    start,
                )

                observations.append(observation)
                instructions.append(token_ids.numpy())
                lengths.append(length)
                labels.append(action)
                task_names.append(task)

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "instructions": np.asarray(instructions, dtype=np.int64),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.int64),
        "tasks": np.asarray(task_names),
    }


class LanguageRobotDataset(Dataset):

    def __init__(self, data):
        self.observations = torch.tensor(
            data["observations"],
            dtype=torch.float32,
        )

        self.instructions = torch.tensor(
            data["instructions"],
            dtype=torch.long,
        )

        self.lengths = torch.tensor(
            data["lengths"],
            dtype=torch.long,
        )

        self.labels = torch.tensor(
            data["labels"],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.observations[index],
            self.instructions[index],
            self.lengths[index],
            self.labels[index],
        )


# ============================================================
# 6. LANGUAGE-CONDITIONED POLICY
# ============================================================

class LanguageConditionedPolicy(nn.Module):
    """
    A policy that combines visual and language information.

    Visual branch:
        Grid -> CNN -> 64 features

    Language branch:
        Token IDs -> Embedding -> GRU -> 64 features

    Fusion:
        Visual features + Language features
        -> MLP -> Action
    """

    def __init__(
        self,
        vocabulary_size,
        embedding_dim=32,
        language_hidden=64,
        visual_features=64,
    ):
        super().__init__()

        # ----------------------------
        # Visual encoder
        # ----------------------------

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(64, visual_features),
            nn.ReLU(),
        )

        # ----------------------------
        # Language encoder
        # ----------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_ID,
        )

        self.language_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=language_hidden,
            batch_first=True,
        )

        # ----------------------------
        # Multimodal fusion
        # ----------------------------

        self.policy_head = nn.Sequential(
            nn.Linear(
                visual_features + language_hidden,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, len(ACTION_NAMES)),
        )

    def forward(self, observation, instruction, lengths):

        # Encode visual observation.
        visual_features = self.visual_encoder(observation)

        # Convert token IDs into word embeddings.
        embedded_words = self.embedding(instruction)

        # Ignore padding tokens when processing the sentence.
        packed_words = nn.utils.rnn.pack_padded_sequence(
            embedded_words,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Final GRU hidden state represents the instruction.
        _, hidden = self.language_encoder(packed_words)

        language_features = hidden[-1]

        # Combine both modalities.
        combined_features = torch.cat(
            [visual_features, language_features],
            dim=1,
        )

        # Predict the robot action.
        return self.policy_head(combined_features)


# ============================================================
# 7. TRAINING
# ============================================================

def train_model(model, train_loader, test_loader):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for observations, instructions, lengths, labels in train_loader:

            observations = observations.to(DEVICE)
            instructions = instructions.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(
                observations,
                instructions,
                lengths,
            )

            loss = criterion(predictions, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predicted_actions = predictions.argmax(dim=1)

            correct += (
                predicted_actions == labels
            ).sum().item()

            total += labels.size(0)

        train_accuracy = 100.0 * correct / total

        test_accuracy = evaluate(model, test_loader)

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Loss={total_loss / len(train_loader):.4f} "
            f"Train Acc={train_accuracy:.2f}% "
            f"Test Acc={test_accuracy:.2f}%"
        )


# ============================================================
# 8. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    for observations, instructions, lengths, labels in loader:

        observations = observations.to(DEVICE)
        instructions = instructions.to(DEVICE)
        labels = labels.to(DEVICE)

        predictions = model(
            observations,
            instructions,
            lengths,
        )

        predicted_actions = predictions.argmax(dim=1)

        correct += (
            predicted_actions == labels
        ).sum().item()

        total += labels.size(0)

    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate_by_task(model, data):

    model.eval()

    print("\nACCURACY BY LANGUAGE TASK")
    print("=" * 55)

    for task in TASK_INSTRUCTIONS:

        indices = np.where(data["tasks"] == task)[0]

        observations = torch.tensor(
            data["observations"][indices],
            dtype=torch.float32,
            device=DEVICE,
        )

        instructions = torch.tensor(
            data["instructions"][indices],
            dtype=torch.long,
            device=DEVICE,
        )

        lengths = torch.tensor(
            data["lengths"][indices],
            dtype=torch.long,
        )

        labels = torch.tensor(
            data["labels"][indices],
            dtype=torch.long,
            device=DEVICE,
        )

        predictions = model(
            observations,
            instructions,
            lengths,
        ).argmax(dim=1)

        accuracy = (
            predictions == labels
        ).float().mean().item() * 100

        print(f"{task:10s}: {accuracy:6.2f}%")


# ============================================================
# 9. EXPERIMENT: SAME IMAGE, DIFFERENT INSTRUCTIONS
# ============================================================

@torch.no_grad()
def compare_instructions(model):

    model.eval()

    # Create one fixed environment.
    environment = generate_map()

    robot_position = (GRID_SIZE // 2, GRID_SIZE // 2)

    # Keep the robot position valid.
    obstacles = environment["obstacles"]

    if obstacles[robot_position[1], robot_position[0]] == 1:
        obstacles[robot_position[1], robot_position[0]] = 0

    observation = build_observation(
        environment,
        robot_position,
    )

    observation_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    print("\nSAME OBSERVATION, DIFFERENT INSTRUCTIONS")
    print("=" * 65)

    print("Robot position:", robot_position)
    print("Red object:", environment["red"])
    print("Blue object:", environment["blue"])

    for task, sentences in TASK_INSTRUCTIONS.items():

        sentence = sentences[0]

        token_ids, length = encode_instruction(sentence)

        instruction_tensor = token_ids.unsqueeze(0).to(DEVICE)

        length_tensor = torch.tensor([length])

        logits = model(
            observation_tensor,
            instruction_tensor,
            length_tensor,
        )

        probabilities = torch.softmax(logits, dim=1)[0]

        predicted_action = probabilities.argmax().item()

        confidence = probabilities[predicted_action].item() * 100

        print(f"\nInstruction: {sentence}")
        print(f"Predicted action: {ACTION_NAMES[predicted_action]}")
        print(f"Confidence: {confidence:.2f}%")


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("V4: LANGUAGE-CONDITIONED IMITATION LEARNING")
    print("=" * 65)

    print("Device:", DEVICE)
    print("Vocabulary size:", len(VOCAB))

    print("\nVocabulary:")
    print(VOCAB)

    # Generate training and test environments independently.
    print("\nGenerating training dataset...")

    train_data = generate_dataset(NUM_TRAIN_MAPS)

    print("Training samples:", len(train_data["labels"]))

    print("\nGenerating test dataset...")

    test_data = generate_dataset(NUM_TEST_MAPS)

    print("Test samples:", len(test_data["labels"]))

    train_dataset = LanguageRobotDataset(train_data)
    test_dataset = LanguageRobotDataset(test_data)

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

    # Create the multimodal policy.
    model = LanguageConditionedPolicy(
        vocabulary_size=len(VOCAB),
    ).to(DEVICE)

    print("\nMODEL ARCHITECTURE")
    print("=" * 65)
    print(model)

    print("\nStarting training...\n")

    train_model(
        model,
        train_loader,
        test_loader,
    )

    final_accuracy = evaluate(model, test_loader)

    print("\nFINAL TEST ACCURACY")
    print("=" * 65)
    print(f"Overall accuracy: {final_accuracy:.2f}%")

    evaluate_by_task(model, test_data)

    # Demonstrate the effect of changing the instruction.
    compare_instructions(model)

    # Save model and vocabulary.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocabulary": VOCAB,
        "grid_size": GRID_SIZE,
        "num_channels": NUM_CHANNELS,
        "num_actions": len(ACTION_NAMES),
        "action_names": ACTION_NAMES,
        "max_text_length": MAX_TEXT_LENGTH,
        "embedding_dim": 32,
        "language_hidden": 64,
        "visual_features": 64,
        "test_accuracy": final_accuracy,
    }

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        "language_conditioned_policy_v4.pth",
    )

    torch.save(checkpoint, checkpoint_path)

    print("\nModel saved to:")
    print(checkpoint_path)

    print("\nV4 TRAINING COMPLETE!")


if __name__ == "__main__":
    main()
