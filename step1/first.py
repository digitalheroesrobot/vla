""" Imagine a simpllified robot can see where the target object is :

    -1 means the target is to the left
    0 means the target is straight ahead
    1 means the target is to the right

"""
# A tiny demonstration dataset
training_data = [
    {"position": -1, "action": "left"},
    {"position": 0, "action": "forward"},
    {"position": 1, "action": "right"},
    {"position": -1, "action": "left"},
    {"position": 1, "action": "right"}
]

# A simple policy
def robot_policy(position):
    if position < 0:
        return "left"
    elif position > 0:
        return "right"
    else:
        return "forward"

# Test the policy
for example in training_data:
    prediction = robot_policy(example["position"])
    print("input:", example["position"], "expected:", example["action"], "predicted:", prediction)

