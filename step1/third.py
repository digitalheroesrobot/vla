
import torch
from torch import nn

# Generate all 25 examples.
# Each input is [robot_position, target_position].
# Labels: 0 = left, 1 = stay, 2 = right.

examples = []

for robot_pos in range(5):
    for target_pos in range(5):
        if target_pos < robot_pos:
            action = 0
        elif target_pos == robot_pos:
            action = 1
        else:
            action = 2

        examples.append(([robot_pos, target_pos], action))

x_train = torch.tensor(
    [item[0] for item in examples], dtype=torch.float32
)
y_train = torch.tensor(
    [item[1] for item in examples], dtype=torch.long
)

# print(x_train, y_train)

# Neural network: 2 inputs -> 16 hidden units -> 3 action scores
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

# Training loop
for epoch in range(500):
    scores = model(x_train)
    loss = loss_function(scores, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training complete!")
print("Final loss:", loss.item())


action_names = ["left", "stay", "right"]

def predict_action(robot_pos, target_pos):
    inputs = torch.tensor(
        [[robot_pos, target_pos]], dtype=torch.float32
    )

    with torch.no_grad():
        scores = model(inputs)
        action_id = scores.argmax(dim=1).item()

    return action_names[action_id]

# Start the robot at position 1; the target is at position 4.
test_cases = [
    (0, 4),  # should move right
    (3, 1),  # should move left
    (2, 2),  # should stay
    (1, 3),  # should move right
]

for robot_pos, target_pos in test_cases:
    prediction = predict_action(robot_pos, target_pos)
    print(
        f"Robot={robot_pos}, target={target_pos}, "
        f"predicted action={prediction}"
    )


with torch.no_grad():
    scores = model(x_train)
    predictions = scores.argmax(dim=1)
    accuracy = (predictions == y_train).float().mean()
    print(accuracy)

print(f"Training accuracy: {accuracy.item():.1%}")
