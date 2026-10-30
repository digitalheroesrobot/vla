
"""
============================================================
V3: SEQUENTIAL IMITATION LEARNING
============================================================

Learning objectives:
1. Understand why a single observation can be insufficient.
2. Introduce observation sequences and action history.
3. Understand vanilla RNNs, LSTMs, and Transformers.
4. Predict actions using multiple previous observations.
5. Compare single-frame and sequence-based policies.
6. Understand why temporal information matters in robotics.

Environment:
    A robot sees a directional clue.
    It moves through a corridor.
    The clue disappears.
    At the junction, it must remember the clue and turn.

Models:
    1. SingleFramePolicy (MLP)
    2. RNNPolicy
    3. LSTMPolicy
    4. TransformerPolicy

Run:
    python sequence_learning.py
============================================================
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

SEQUENCE_LENGTH = 5

TRAIN_SAMPLES = 8000
TEST_SAMPLES = 2000

BATCH_SIZE = 128
EPOCHS = 30

LEARNING_RATE = 0.001

HIDDEN_SIZE = 64

# Observation:
# [robot_position, at_junction, cue_left, cue_right]
OBSERVATION_SIZE = 4

# Previous action:
# [FORWARD, LEFT, RIGHT, STOP]
ACTION_SIZE = 4

# Complete input:
# observation + previous action
INPUT_SIZE = OBSERVATION_SIZE + ACTION_SIZE

NUM_ACTIONS = 4

ACTION_NAMES = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "STOP"
]

FORWARD = 0
LEFT = 1
RIGHT = 2
STOP = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = Path("data/sequence_learning_v3")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 2. ENVIRONMENT
# ============================================================

"""
The robot moves through a simple corridor.

At the beginning:
    It sees a directional clue.

During movement:
    The clue disappears.

At the junction:
    It must choose LEFT or RIGHT.

The robot cannot determine the correct final action
from its current observation alone.

It needs its observation history.
"""


def generate_trajectory(direction):
    """
    Generate one expert demonstration.

    direction:
        0 = LEFT
        1 = RIGHT

    Returns:
        observations:
            A sequence of robot observations.

        expert_actions:
            The expert action at every timestep.

        sequence_input:
            Observations combined with previous actions.

        label:
            The final action to predict.
    """

    # --------------------------------------------------------
    # Robot observations
    # --------------------------------------------------------

    positions = [
        0.00,
        0.25,
        0.50,
        0.75,
        1.00
    ]

    observations = []

    for t in range(SEQUENCE_LENGTH):

        position = positions[t]

        at_junction = 0.0

        cue_left = 0.0
        cue_right = 0.0

        # Directional clue appears only at the beginning.
        if t == 0:

            if direction == 0:

                cue_left = 1.0

            else:

                cue_right = 1.0

        # Robot reaches the junction at the final timestep.
        if t == SEQUENCE_LENGTH - 1:

            at_junction = 1.0

        observation = [
            position,
            at_junction,
            cue_left,
            cue_right
        ]

        observations.append(observation)

    observations = np.array(
        observations,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Expert action sequence
    # --------------------------------------------------------

    # The expert moves forward through the corridor.
    expert_actions = [
        FORWARD,
        FORWARD,
        FORWARD,
        FORWARD
    ]

    # At the junction, choose the remembered direction.
    if direction == 0:

        expert_actions.append(LEFT)

    else:

        expert_actions.append(RIGHT)

    # --------------------------------------------------------
    # Previous-action history
    # --------------------------------------------------------

    sequence_input = []

    for t in range(SEQUENCE_LENGTH):

        previous_action = np.zeros(
            ACTION_SIZE,
            dtype=np.float32
        )

        # At timestep zero, no previous action exists.
        if t > 0:

            previous_action_id = expert_actions[t - 1]

            previous_action[previous_action_id] = 1.0

        combined = np.concatenate(
            [
                observations[t],
                previous_action
            ]
        )

        sequence_input.append(combined)

    sequence_input = np.array(
        sequence_input,
        dtype=np.float32
    )

    # The target is the action at the junction.
    label = expert_actions[-1]

    return (
        observations,
        expert_actions,
        sequence_input,
        label
    )


# ============================================================
# 3. DATASET
# ============================================================

class SequenceDataset(Dataset):
    """
    Dataset of expert robot trajectories.

    Every sample contains:

        [observation_0, previous_action_0]
        [observation_1, previous_action_1]
        ...
        [observation_4, previous_action_4]

    Target:
        Expert action at the final timestep.
    """

    def __init__(self, number_of_samples):

        self.sequences = []
        self.labels = []

        # Balanced LEFT and RIGHT demonstrations.
        directions = np.arange(number_of_samples) % 2

        np.random.shuffle(directions)

        for direction in directions:

            (
                observations,
                expert_actions,
                sequence_input,
                label
            ) = generate_trajectory(int(direction))

            self.sequences.append(sequence_input)

            self.labels.append(label)

        self.sequences = torch.tensor(
            np.array(self.sequences),
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            np.array(self.labels),
            dtype=torch.long
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.sequences[index],
            self.labels[index]
        )


# ============================================================
# 4. MODEL 1: SINGLE-FRAME POLICY
# ============================================================

class SingleFramePolicy(nn.Module):
    """
    A normal feed-forward neural network.

    It receives only the latest observation and previous action.

    It cannot directly access earlier observations.

    This is our baseline model.
    """

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


# ============================================================
# 5. MODEL 2: VANILLA RNN
# ============================================================

class RNNPolicy(nn.Module):
    """
    Recurrent Neural Network.

    An RNN processes observations sequentially.

    Its hidden state is updated at every timestep:

        h_t = RNN(x_t, h_(t-1))

    Information from earlier observations can influence
    the current hidden state.

    Basic limitation:
        Vanilla RNNs can struggle to preserve information
        over long sequences because of vanishing gradients.
    """

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

        # x:
        # [batch, sequence_length, input_size]

        output, hidden = self.rnn(x)

        # Final timestep representation.
        final_output = output[:, -1, :]

        return self.classifier(final_output)


# ============================================================
# 6. MODEL 3: LSTM
# ============================================================

class LSTMPolicy(nn.Module):
    """
    Long Short-Term Memory network.

    LSTM is designed to preserve useful information
    over time.

    It maintains:
        - Hidden state
        - Cell state

    Its gates control:
        - What information to forget
        - What information to remember
        - What information to output

    This makes it useful for robot observation histories.
    """

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

        final_output = output[:, -1, :]

        return self.classifier(final_output)


# ============================================================
# 7. MODEL 4: TRANSFORMER
# ============================================================

class TransformerPolicy(nn.Module):
    """
    Transformer encoder for robot observation sequences.

    Unlike RNNs and LSTMs, Transformers do not process
    information through a recurrent hidden state.

    They use self-attention to relate different timesteps.

    Positional embeddings provide information about
    the order of observations.

    The final timestep representation is used to predict
    the next robot action.
    """

    def __init__(self):

        super().__init__()

        self.input_projection = nn.Linear(
            INPUT_SIZE,
            HIDDEN_SIZE
        )

        # Learnable positional embeddings.
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

        # Project observations into the Transformer dimension.
        x = self.input_projection(x)

        # Add timestep information.
        x = x + self.position_embedding

        # Process the complete sequence.
        output = self.transformer(x)

        # Use the final timestep.
        final_output = output[:, -1, :]

        return self.classifier(final_output)


# ============================================================
# 8. MODEL INPUT HANDLING
# ============================================================

def get_model_input(model, sequences):
    """
    The single-frame model receives only the final observation.

    Sequence models receive the complete history.
    """

    if isinstance(model, SingleFramePolicy):

        return sequences[:, -1, :]

    return sequences


# ============================================================
# 9. EVALUATION
# ============================================================

def evaluate_model(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for sequences, labels in loader:

            sequences = sequences.to(DEVICE)

            labels = labels.to(DEVICE)

            inputs = get_model_input(
                model,
                sequences
            )

            logits = model(inputs)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100.0 * correct / total


# ============================================================
# 10. TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    test_loader,
    model_name
):

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=1e-4
    )

    history = {

        "train_loss": [],

        "test_accuracy": []
    }

    print("\n" + "=" * 65)

    print(f"TRAINING: {model_name}")

    print("=" * 65)

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for sequences, labels in train_loader:

            sequences = sequences.to(DEVICE)

            labels = labels.to(DEVICE)

            inputs = get_model_input(
                model,
                sequences
            )

            optimizer.zero_grad()

            logits = model(inputs)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(train_loader)
        )

        accuracy = evaluate_model(
            model,
            test_loader
        )

        history["train_loss"].append(
            average_loss
        )

        history["test_accuracy"].append(
            accuracy
        )

        print(

            f"Epoch {epoch + 1:02d}/{EPOCHS} | "

            f"Loss: {average_loss:.4f} | "

            f"Test Accuracy: {accuracy:.2f}%"

        )

    return model, history


# ============================================================
# 11. VISUALIZE AN EXPERT TRAJECTORY
# ============================================================

def visualize_trajectory():

    left = generate_trajectory(direction=0)

    right = generate_trajectory(direction=1)

    left_observations = left[0]

    right_observations = right[0]

    left_actions = left[1]

    right_actions = right[1]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7)
    )

    timesteps = np.arange(SEQUENCE_LENGTH)

    # LEFT trajectory.
    axes[0].plot(
        timesteps,
        left_observations[:, 0],
        marker="o",
        label="Robot position"
    )

    axes[0].plot(
        timesteps,
        left_observations[:, 2],
        marker="s",
        label="LEFT clue"
    )

    axes[0].plot(
        timesteps,
        left_observations[:, 3],
        marker="^",
        label="RIGHT clue"
    )

    axes[0].set_title(
        "LEFT trajectory"
    )

    axes[0].set_xlabel("Timestep")

    axes[0].set_ylabel("Observation value")

    axes[0].legend()

    axes[0].grid(True)

    # RIGHT trajectory.
    axes[1].plot(
        timesteps,
        right_observations[:, 0],
        marker="o",
        label="Robot position"
    )

    axes[1].plot(
        timesteps,
        right_observations[:, 2],
        marker="s",
        label="LEFT clue"
    )

    axes[1].plot(
        timesteps,
        right_observations[:, 3],
        marker="^",
        label="RIGHT clue"
    )

    axes[1].set_title(
        "RIGHT trajectory"
    )

    axes[1].set_xlabel("Timestep")

    axes[1].set_ylabel("Observation value")

    axes[1].legend()

    axes[1].grid(True)

    plt.tight_layout()

    plt.show()

    print("\nEXPERT ACTION SEQUENCES")

    print(
        "LEFT: ",
        [ACTION_NAMES[a] for a in left_actions]
    )

    print(
        "RIGHT:",
        [ACTION_NAMES[a] for a in right_actions]
    )


# ============================================================
# 12. PREDICT AN INDIVIDUAL TRAJECTORY
# ============================================================

def predict_trajectory(model, direction):

    (
        observations,
        expert_actions,
        sequence_input,
        label
    ) = generate_trajectory(direction)

    sequence = torch.tensor(
        sequence_input,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    model.eval()

    with torch.no_grad():

        inputs = get_model_input(
            model,
            sequence
        )

        logits = model(inputs)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    return (
        ACTION_NAMES[prediction],

        probabilities.cpu().numpy()[0],

        ACTION_NAMES[label]
    )


# ============================================================
# 13. PLOT MODEL COMPARISON
# ============================================================

def plot_results(histories, final_accuracies):

    epochs = range(1, EPOCHS + 1)

    # --------------------------------------------------------
    # Accuracy comparison
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for name, history in histories.items():

        plt.plot(
            epochs,
            history["test_accuracy"],
            label=name
        )

    plt.xlabel("Epoch")

    plt.ylabel("Test accuracy (%)")

    plt.title(
        "Single-frame vs Sequential Policies"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    # --------------------------------------------------------
    # Loss comparison
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for name, history in histories.items():

        plt.plot(
            epochs,
            history["train_loss"],
            label=name
        )

    plt.xlabel("Epoch")

    plt.ylabel("Training loss")

    plt.title(
        "Training Loss Comparison"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    # --------------------------------------------------------
    # Final accuracy comparison
    # --------------------------------------------------------

    names = list(final_accuracies.keys())

    values = list(final_accuracies.values())

    plt.figure(figsize=(10, 5))

    plt.bar(names, values)

    plt.ylabel("Test accuracy (%)")

    plt.ylim(0, 105)

    plt.title(
        "Final Model Accuracy Comparison"
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print("=" * 65)

    print("V3: SEQUENTIAL IMITATION LEARNING")

    print("=" * 65)

    print(f"Device: {DEVICE}")

    print(f"Sequence length: {SEQUENCE_LENGTH}")

    print(f"Training samples: {TRAIN_SAMPLES}")

    print(f"Test samples: {TEST_SAMPLES}")

    print(f"Input size: {INPUT_SIZE}")

    print(f"Number of actions: {NUM_ACTIONS}")

    # --------------------------------------------------------
    # Generate datasets
    # --------------------------------------------------------

    train_dataset = SequenceDataset(
        TRAIN_SAMPLES
    )

    test_dataset = SequenceDataset(
        TEST_SAMPLES
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False
    )

    print("\nDATASET INFORMATION")

    print("-" * 65)

    print(
        "Training input shape:",
        tuple(train_dataset.sequences.shape)
    )

    print(
        "Test input shape:",
        tuple(test_dataset.sequences.shape)
    )

    print(
        "Training labels:",
        tuple(train_dataset.labels.shape)
    )

    print(
        "Training LEFT samples:",
        (train_dataset.labels == LEFT).sum().item()
    )

    print(
        "Training RIGHT samples:",
        (train_dataset.labels == RIGHT).sum().item()
    )

    # --------------------------------------------------------
    # Visualize trajectories
    # --------------------------------------------------------

    visualize_trajectory()

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    models = {

        "Single-frame MLP": SingleFramePolicy(),

        "Vanilla RNN": RNNPolicy(),

        "LSTM": LSTMPolicy(),

        "Transformer": TransformerPolicy()
    }

    histories = {}

    final_accuracies = {}

    # --------------------------------------------------------
    # Train and evaluate each model
    # --------------------------------------------------------

    for name, model in models.items():

        trained_model, history = train_model(

            model,

            train_loader,

            test_loader,

            name
        )

        models[name] = trained_model

        histories[name] = history

        accuracy = evaluate_model(

            trained_model,

            test_loader
        )

        final_accuracies[name] = accuracy

        # Save each model independently.
        checkpoint_path = (
            OUTPUT_DIR /
            f"{name.lower().replace('-', '_').replace(' ', '_')}_v3.pth"
        )

        torch.save(

            {

                "model_state_dict": trained_model.state_dict(),

                "model_name": name,

                "sequence_length": SEQUENCE_LENGTH,

                "input_size": INPUT_SIZE,

                "hidden_size": HIDDEN_SIZE,

                "action_names": ACTION_NAMES,

                "test_accuracy": accuracy

            },

            checkpoint_path
        )

        print(
            f"\nSaved {name} to: {checkpoint_path}"
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print("FINAL MODEL COMPARISON")

    print("=" * 65)

    for name, accuracy in final_accuracies.items():

        print(
            f"{name:25s}: {accuracy:.2f}%"
        )

    baseline = final_accuracies["Single-frame MLP"]

    print("\nIMPROVEMENT OVER SINGLE-FRAME BASELINE")

    print("-" * 65)

    for name, accuracy in final_accuracies.items():

        improvement = accuracy - baseline

        print(
            f"{name:25s}: {improvement:+.2f} percentage points"
        )

    # --------------------------------------------------------
    # Test individual sequences
    # --------------------------------------------------------

    print("\nINDIVIDUAL TRAJECTORY PREDICTIONS")

    print("=" * 65)

    for direction in [0, 1]:

        print(
            f"\nExpected direction: "
            f"{ACTION_NAMES[LEFT if direction == 0 else RIGHT]}"
        )

        for name, model in models.items():

            prediction, probabilities, expected = (
                predict_trajectory(
                    model,
                    direction
                )
            )

            print(
                f"{name:25s}: "
                f"{prediction:10s} "
                f"Confidence={probabilities.max():.3f}"
            )

    # --------------------------------------------------------
    # Plot results
    # --------------------------------------------------------

    plot_results(
        histories,
        final_accuracies
    )

    print("\n" + "=" * 65)

    print("V3 EXPERIMENT COMPLETED")

    print("=" * 65)

    print(
        f"\nModels saved in: {OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()
