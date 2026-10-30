
"""
STEP 1 - COMPUTER VISION
Image-based perception with a Convolutional Neural Network (CNN).

The model learns to recognize:
    0 = CIRCLE
    1 = SQUARE
    2 = TRIANGLE

Images are synthetic grayscale images of size 32 x 32.

Run from the repository root:
    python step1/vision.py
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
NUM_SAMPLES = 3000
IMAGE_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 0.001

CLASS_NAMES = {
    0: "CIRCLE",
    1: "SQUARE",
    2: "TRIANGLE",
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# --------------------------------------------------
# 2. Generate synthetic images
# --------------------------------------------------

def generate_image(class_id):
    """
    Generate one grayscale image.

    Background:
        Dark gray

    Object:
        Bright gray

    The object position and size vary randomly.
    Small pixel noise makes the task less trivial.
    """

    image = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        color=20,
    )

    draw = ImageDraw.Draw(image)

    size = random.randint(12, 22)

    x = random.randint(2, IMAGE_SIZE - size - 2)
    y = random.randint(2, IMAGE_SIZE - size - 2)

    right = x + size
    bottom = y + size

    object_color = random.randint(210, 255)

    if class_id == 0:
        # Circle
        draw.ellipse(
            (x, y, right, bottom),
            fill=object_color,
        )

    elif class_id == 1:
        # Square
        draw.rectangle(
            (x, y, right, bottom),
            fill=object_color,
        )

    elif class_id == 2:
        # Triangle
        points = [
            (x + size // 2, y),
            (x, bottom),
            (right, bottom),
        ]

        draw.polygon(
            points,
            fill=object_color,
        )

    # Convert image to NumPy array.
    image_array = np.array(
        image,
        dtype=np.float32,
    )

    # Add a little random sensor-like noise.
    noise = np.random.normal(
        loc=0.0,
        scale=10.0,
        size=(IMAGE_SIZE, IMAGE_SIZE),
    )

    image_array = image_array + noise

    # Keep pixel values between 0 and 255.
    image_array = np.clip(
        image_array,
        0,
        255,
    )

    # Normalize pixel values to [0, 1].
    image_array = image_array / 255.0

    return image_array.astype(np.float32)


def generate_dataset(num_samples):
    images = []
    labels = []

    for _ in range(num_samples):

        class_id = random.randint(0, 2)

        image = generate_image(class_id)

        images.append(image)
        labels.append(class_id)

    images = np.array(
        images,
        dtype=np.float32,
    )

    labels = np.array(
        labels,
        dtype=np.int64,
    )

    return images, labels


images, labels = generate_dataset(NUM_SAMPLES)

print("\nDataset generated!")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)


# --------------------------------------------------
# 3. Understand image tensors
# --------------------------------------------------

# Current NumPy shape:
# [number_of_images, height, width]
#
# PyTorch CNN format:
# [number_of_images, channels, height, width]

images_tensor = torch.tensor(
    images,
    dtype=torch.float32,
).unsqueeze(1)

labels_tensor = torch.tensor(
    labels,
    dtype=torch.long,
)

print("\nPyTorch image tensor shape:", images_tensor.shape)
print("Label tensor shape:", labels_tensor.shape)

print("\nFirst image information:")
print("Shape:", images_tensor[0].shape)
print("Minimum pixel value:", images_tensor[0].min().item())
print("Maximum pixel value:", images_tensor[0].max().item())


# --------------------------------------------------
# 4. Visualize sample images
# --------------------------------------------------

fig, axes = plt.subplots(
    3,
    4,
    figsize=(9, 7),
)

for i, ax in enumerate(axes.flat):

    ax.imshow(
        images[i],
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    ax.set_title(CLASS_NAMES[labels[i]])
    ax.axis("off")

plt.suptitle("Synthetic Robot Vision Dataset")
plt.tight_layout()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "vision",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

sample_plot_path = os.path.join(
    OUTPUT_DIR,
    "sample_images.png",
)

plt.savefig(
    sample_plot_path,
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# --------------------------------------------------
# 5. Create training and testing datasets
# --------------------------------------------------

dataset = TensorDataset(
    images_tensor,
    labels_tensor,
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
print("Training images:", train_size)
print("Testing images:", test_size)


# --------------------------------------------------
# 6. Build a Convolutional Neural Network
# --------------------------------------------------

class VisionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # First convolutional layer.
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            # Second convolutional layer.
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            # Third convolutional layer.
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64 * 4 * 4, 128),

            nn.ReLU(),

            nn.Linear(128, 3),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = VisionCNN().to(DEVICE)

print("\nCNN model:")
print(model)


# --------------------------------------------------
# 7. Loss function and optimizer
# --------------------------------------------------

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# 8. Train the CNN
# --------------------------------------------------

loss_history = []

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_images, batch_labels in train_loader:

        batch_images = batch_images.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        # Clear old gradients.
        optimizer.zero_grad()

        # Forward pass.
        predictions = model(batch_images)

        # Calculate classification loss.
        loss = loss_fn(
            predictions,
            batch_labels,
        )

        # Backpropagation.
        loss.backward()

        # Update model parameters.
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
# 9. Evaluate the model
# --------------------------------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch_images, batch_labels in test_loader:

        batch_images = batch_images.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        logits = model(batch_images)

        predicted_classes = torch.argmax(
            logits,
            dim=1,
        )

        correct += (
            predicted_classes == batch_labels
        ).sum().item()

        total += batch_labels.size(0)


accuracy = 100.0 * correct / total

print("\n========== TEST RESULTS ==========")

print(f"Correct predictions: {correct}/{total}")
print(f"Test accuracy: {accuracy:.2f}%")


# --------------------------------------------------
# 10. Test individual images
# --------------------------------------------------

print("\n========== INDIVIDUAL PREDICTIONS ==========")

model.eval()

with torch.no_grad():

    for i in range(10):

        image = images_tensor[i].unsqueeze(0).to(DEVICE)

        logits = model(image)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1,
        ).item()

        actual_class = labels[i]

        confidence = probabilities[0, predicted_class].item()

        print(
            f"Image {i + 1}: "
            f"Actual={CLASS_NAMES[actual_class]} | "
            f"Predicted={CLASS_NAMES[predicted_class]} | "
            f"Confidence={confidence:.3f}"
        )


# --------------------------------------------------
# 11. Plot training loss
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    loss_history,
    label="Training loss",
)

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("CNN Training Loss")
plt.legend()
plt.grid(True)

loss_plot_path = os.path.join(
    OUTPUT_DIR,
    "training_loss.png",
)

plt.savefig(
    loss_plot_path,
    dpi=150,
    bbox_inches="tight",
)

plt.show()


print("\nSample images saved to:", sample_plot_path)
print("Training graph saved to:", loss_plot_path)
print("\nComputer vision lesson completed!")
