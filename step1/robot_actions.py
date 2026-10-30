
"""
STEP 9: ROBOT ACTION REPRESENTATIONS

Learn how a robot represents movement numerically.

Topics:
1. Discrete vs continuous actions
2. Joint positions and joint velocities
3. End-effector positions
4. Coordinate systems
5. Forward kinematics (FK)
6. Inverse kinematics (IK)
7. Numerical action vectors
8. Action sequences and action chunks

This exercise uses a simplified 2D robot arm with two revolute joints.

No external libraries are required.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

# Length of each arm segment, in meters.
LINK_1 = 0.5
LINK_2 = 0.4

# Simulation time step.
DT = 0.1

# Robot action definitions.
ACTIONS = {
    0: "MOVE_LEFT",
    1: "MOVE_RIGHT",
    2: "MOVE_FORWARD",
    3: "MOVE_BACKWARD",
    4: "STOP",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def deg(rad: float) -> float:
    """Convert radians to degrees."""
    return math.degrees(rad)


def rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return math.radians(degrees)


def print_vector(name: str, values: List[float], units: str = ""):
    """Print a numerical vector."""
    formatted = ", ".join(f"{v: .3f}" for v in values)
    print(f"{name} = [{formatted}] {units}")


# ============================================================
# 1. DISCRETE ACTIONS
# ============================================================

def demonstrate_discrete_actions():
    print("\n" + "=" * 65)
    print("1. DISCRETE ACTIONS")
    print("=" * 65)

    print("""
A discrete action comes from a finite set of choices.

For example, a grid-world robot might have:

    0 = MOVE_LEFT
    1 = MOVE_RIGHT
    2 = MOVE_FORWARD
    3 = MOVE_BACKWARD
    4 = STOP

The model predicts an action ID rather than an exact motor command.
""")

    predicted_action = 2

    print(f"Model prediction: {predicted_action}")
    print(f"Meaning:          {ACTIONS[predicted_action]}")

    print("""
Advantages:
    - Simple action space
    - Easy to train as a classification problem
    - Useful for navigation and high-level decisions

Limitations:
    - Movement choices are restricted
    - No direct control over movement magnitude
    - Not sufficient for precise motor control
""")


# ============================================================
# 2. CONTINUOUS ACTIONS
# ============================================================

def demonstrate_continuous_actions():
    print("\n" + "=" * 65)
    print("2. CONTINUOUS ACTIONS")
    print("=" * 65)

    print("""
A continuous action contains numerical values.

For example, a two-joint robot could receive:

    [0.2, -0.1]

If these are joint velocities, the meaning is:

    Joint 1 velocity =  0.2 rad/s
    Joint 2 velocity = -0.1 rad/s

Unlike discrete actions, the values can vary continuously.
""")

    action = [0.2, -0.1]

    print_vector("Continuous action", action, "rad/s")

    print("""
Other possible continuous action representations:

    Joint position target:  [1.0, 0.5] rad
    Joint velocity target:  [0.2, -0.1] rad/s
    Joint torque command:   [0.5, 0.3] Nm
    End-effector target:    [0.4, 0.3] meters

IMPORTANT:
The meaning of an action vector depends on the robot controller.
A vector of positions is not the same as a vector of velocities.
""")


# ============================================================
# 3. JOINT POSITIONS AND JOINT VELOCITIES
# ============================================================

@dataclass
class JointState:
    """
    State of a robot's joints.

    q:
        Joint positions in radians.

    dq:
        Joint velocities in radians per second.
    """

    q: List[float]
    dq: List[float]


def integrate_joint_velocity(
    state: JointState,
    velocity_command: List[float],
    dt: float,
) -> JointState:
    """
    Integrate joint velocities to update joint positions.

    q_new = q_old + velocity * dt
    """

    new_q = [
        q + velocity * dt
        for q, velocity in zip(state.q, velocity_command)
    ]

    return JointState(
        q=new_q,
        dq=velocity_command.copy(),
    )


def demonstrate_joint_state():
    print("\n" + "=" * 65)
    print("3. JOINT POSITIONS AND VELOCITIES")
    print("=" * 65)

    state = JointState(
        q=[rad(30), rad(45)],
        dq=[0.0, 0.0],
    )

    print("Initial robot joint state:")
    print_vector("Joint positions", state.q, "radians")
    print_vector("Joint positions", [deg(x) for x in state.q], "degrees")
    print_vector("Joint velocities", state.dq, "rad/s")

    velocity_command = [0.2, -0.1]

    print("\nApply a velocity command:")
    print_vector("Velocity command", velocity_command, "rad/s")

    new_state = integrate_joint_velocity(
        state,
        velocity_command,
        DT,
    )

    print("\nAfter one time step:")
    print_vector("New joint positions", new_state.q, "radians")
    print_vector(
        "New joint positions",
        [deg(x) for x in new_state.q],
        "degrees",
    )

    print(f"\nTime step: {DT} seconds")

    print("""
A physical robot has sensors and controllers that work with joint
positions and velocities.

A robot policy might predict a desired joint position, a desired
velocity, or another control command depending on its design.
""")


# ============================================================
# 4. FORWARD KINEMATICS
# ============================================================

def forward_kinematics(
    q1: float,
    q2: float,
) -> Tuple[float, float]:
    """
    Calculate the end-effector position of a 2-link planar arm.

    q1: First joint angle relative to the horizontal axis.
    q2: Second joint angle relative to the first link.

    Returns:
        (x, y) in the robot base coordinate frame.
    """

    x = (
        LINK_1 * math.cos(q1)
        + LINK_2 * math.cos(q1 + q2)
    )

    y = (
        LINK_1 * math.sin(q1)
        + LINK_2 * math.sin(q1 + q2)
    )

    return x, y


def demonstrate_forward_kinematics():
    print("\n" + "=" * 65)
    print("4. FORWARD KINEMATICS (FK)")
    print("=" * 65)

    print("""
Forward kinematics answers:

    Given the joint positions, where is the robot's end effector?

For a two-link planar robot:

    x = L1*cos(q1) + L2*cos(q1 + q2)
    y = L1*sin(q1) + L2*sin(q1 + q2)

L1 and L2 are link lengths.
q1 and q2 are joint angles.
""")

    q1 = rad(30)
    q2 = rad(45)

    x, y = forward_kinematics(q1, q2)

    print("Robot configuration:")
    print(f"  Link 1 length: {LINK_1} m")
    print(f"  Link 2 length: {LINK_2} m")
    print(f"  Joint 1 angle: {deg(q1):.1f} degrees")
    print(f"  Joint 2 angle: {deg(q2):.1f} degrees")

    print("\nCalculated end-effector position:")
    print_vector("End effector", [x, y], "meters")

    print("""
The end effector is the part of the robot that interacts with the
environment.

Examples:
    - A gripper
    - A robot hand
    - A suction cup
    - A tool attached to the robot wrist
""")


# ============================================================
# 5. INVERSE KINEMATICS
# ============================================================

def inverse_kinematics(
    x: float,
    y: float,
    elbow_up: bool = False,
) -> Tuple[float, float]:
    """
    Calculate joint angles for a desired end-effector position.

    This uses the analytical solution for a 2-link planar arm.

    elbow_up selects between the two possible configurations.

    Raises:
        ValueError if the requested position is unreachable.
    """

    distance_squared = x * x + y * y

    cos_q2 = (
        distance_squared
        - LINK_1 ** 2
        - LINK_2 ** 2
    ) / (2 * LINK_1 * LINK_2)

    if cos_q2 < -1.0 or cos_q2 > 1.0:
        raise ValueError(
            f"Target ({x:.3f}, {y:.3f}) is unreachable."
        )

    cos_q2 = max(-1.0, min(1.0, cos_q2))

    sin_magnitude = math.sqrt(
        max(0.0, 1.0 - cos_q2 * cos_q2)
    )

    sin_q2 = sin_magnitude if elbow_up else -sin_magnitude

    q2 = math.atan2(sin_q2, cos_q2)

    k1 = LINK_1 + LINK_2 * math.cos(q2)
    k2 = LINK_2 * math.sin(q2)

    q1 = (
        math.atan2(y, x)
        - math.atan2(k2, k1)
    )

    return q1, q2


def demonstrate_inverse_kinematics():
    print("\n" + "=" * 65)
    print("5. INVERSE KINEMATICS (IK)")
    print("=" * 65)

    print("""
Inverse kinematics answers the opposite question:

    Given a desired end-effector position, which joint angles
    should the robot use?

For example:

    Desired end-effector position = (0.5, 0.3) meters

The inverse kinematics algorithm calculates q1 and q2.
""")

    target_x = 0.5
    target_y = 0.3

    print_vector(
        "Desired end-effector position",
        [target_x, target_y],
        "meters",
    )

    q1, q2 = inverse_kinematics(
        target_x,
        target_y,
    )

    print("\nCalculated joint angles:")
    print_vector(
        "Joint angles",
        [q1, q2],
        "radians",
    )
    print_vector(
        "Joint angles",
        [deg(q1), deg(q2)],
        "degrees",
    )

    # Verify the result using forward kinematics.
    check_x, check_y = forward_kinematics(q1, q2)

    print("\nVerification using forward kinematics:")
    print_vector(
        "Calculated position",
        [check_x, check_y],
        "meters",
    )

    error = math.sqrt(
        (target_x - check_x) ** 2
        + (target_y - check_y) ** 2
    )

    print(f"Position error: {error:.8f} meters")

    print("""
Notice that inverse kinematics can have multiple solutions.

For a two-link arm, the elbow can bend in different directions
while reaching the same end-effector position.

Real robots have additional considerations:
    - Joint limits
    - Collision avoidance
    - Singularities
    - Multiple possible configurations
    - Motion constraints
""")


# ============================================================
# 6. COORDINATE SYSTEMS
# ============================================================

def local_to_world(
    x_local: float,
    y_local: float,
    origin_x: float,
    origin_y: float,
    rotation: float,
) -> Tuple[float, float]:
    """
    Convert a point from a local coordinate frame to a world frame.

    rotation is the local frame's rotation relative to the world,
    measured in radians.
    """

    x_world = (
        origin_x
        + x_local * math.cos(rotation)
        - y_local * math.sin(rotation)
    )

    y_world = (
        origin_y
        + x_local * math.sin(rotation)
        + y_local * math.cos(rotation)
    )

    return x_world, y_world


def world_to_local(
    x_world: float,
    y_world: float,
    origin_x: float,
    origin_y: float,
    rotation: float,
) -> Tuple[float, float]:
    """Convert a world-frame point into a local coordinate frame."""

    dx = x_world - origin_x
    dy = y_world - origin_y

    x_local = (
        dx * math.cos(rotation)
        + dy * math.sin(rotation)
    )

    y_local = (
        -dx * math.sin(rotation)
        + dy * math.cos(rotation)
    )

    return x_local, y_local


def demonstrate_coordinate_systems():
    print("\n" + "=" * 65)
    print("6. ROBOT COORDINATE SYSTEMS")
    print("=" * 65)

    print("""
A robot operates with multiple coordinate frames.

Common examples:

    WORLD FRAME:
        A global reference for the environment.

    BASE FRAME:
        A reference attached to the robot's base.

    JOINT FRAME:
        A reference associated with a robot joint.

    TOOL FRAME:
        A reference attached to the end effector or tool.

A point's coordinates depend on the frame used to describe it.
""")

    # Define the robot's local frame in the world.
    origin_x = 1.0
    origin_y = 2.0
    rotation = rad(90)

    local_point = (0.5, 0.0)

    world_point = local_to_world(
        local_point[0],
        local_point[1],
        origin_x,
        origin_y,
        rotation,
    )

    print("Robot local frame:")
    print_vector(
        "Frame origin in world",
        [origin_x, origin_y],
        "meters",
    )
    print(f"Frame rotation: {deg(rotation):.1f} degrees")

    print("\nA point in the robot's local frame:")
    print_vector(
        "Local point",
        list(local_point),
        "meters",
    )

    print("\nThe same point in the world frame:")
    print_vector(
        "World point",
        list(world_point),
        "meters",
    )

    recovered_point = world_to_local(
        world_point[0],
        world_point[1],
        origin_x,
        origin_y,
        rotation,
    )

    print("\nConvert back into the local frame:")
    print_vector(
        "Recovered local point",
        list(recovered_point),
        "meters",
    )

    print("""
Why coordinate frames matter:

A robot may see an object in camera coordinates, calculate a target
in world coordinates, and execute movement using its joint coordinates.

Those representations must be transformed consistently.
""")


# ============================================================
# 7. ROBOT ACTION VECTORS
# ============================================================

def demonstrate_action_vectors():
    print("\n" + "=" * 65)
    print("7. REPRESENTING ACTIONS AS NUMERICAL VECTORS")
    print("=" * 65)

    print("""
A robot action can be represented as a numerical vector.

For a two-joint arm, examples include:

    Joint position target:
        [q1_target, q2_target]

    Joint velocity command:
        [dq1, dq2]

    End-effector position target:
        [x, y]

For a real robot with six joints, a joint action might contain
six values instead of two.
""")

    joint_position_action = [
        rad(45),
        rad(30),
    ]

    joint_velocity_action = [
        0.2,
        -0.1,
    ]

    end_effector_action = [
        0.5,
        0.3,
    ]

    print_vector(
        "Joint position action",
        joint_position_action,
        "radians",
    )

    print_vector(
        "Joint velocity action",
        joint_velocity_action,
        "rad/s",
    )

    print_vector(
        "End-effector target",
        end_effector_action,
        "meters",
    )

    print("""
The action dimension depends on the robot.

Examples:

    2-DOF arm:
        2 joint values

    6-DOF arm:
        6 joint values

    7-DOF arm:
        7 joint values

A mobile manipulator may have even more action dimensions.

DOF means degrees of freedom: the number of independent
coordinates needed to describe the robot's configuration.
""")


# ============================================================
# 8. ACTION SEQUENCES
# ============================================================

def demonstrate_action_sequences():
    print("\n" + "=" * 65)
    print("8. ACTION SEQUENCES")
    print("=" * 65)

    print("""
A robot usually needs multiple actions to complete a task.

An action sequence is a time-ordered list of actions.

For example, a two-joint robot might execute:

    t=0: [ 0.1,  0.0]
    t=1: [ 0.1,  0.1]
    t=2: [ 0.0,  0.1]
    t=3: [-0.1,  0.0]

Each vector represents a joint velocity command.
""")

    action_sequence = [
        [0.1, 0.0],
        [0.1, 0.1],
        [0.0, 0.1],
        [-0.1, 0.0],
    ]

    state = JointState(
        q=[0.0, 0.0],
        dq=[0.0, 0.0],
    )

    print("Executing action sequence:")

    for index, action in enumerate(action_sequence):
        state = integrate_joint_velocity(
            state,
            action,
            DT,
        )

        print(
            f"  Step {index + 1}: "
            f"action={action}, "
            f"joint_positions="
            f"[{state.q[0]:.3f}, {state.q[1]:.3f}]"
        )


# ============================================================
# 9. ACTION CHUNKS
# ============================================================

def demonstrate_action_chunks():
    print("\n" + "=" * 65)
    print("9. ACTION CHUNKS")
    print("=" * 65)

    print("""
An action chunk is a group of future actions predicted together.

Instead of predicting only one action:

    action = [0.1, 0.0]

A model can predict several future actions:

    action_chunk = [
        [ 0.1,  0.0],
        [ 0.1,  0.1],
        [ 0.0,  0.1],
        [-0.1,  0.0],
    ]

The robot executes the chunk over time.

This is useful because a model can plan a short movement sequence
rather than making every decision independently.
""")

    action_chunk = [
        [0.10, 0.00],
        [0.10, 0.05],
        [0.05, 0.10],
        [0.00, 0.10],
        [-0.05, 0.10],
    ]

    print("Predicted action chunk:")
    print("  Shape: (5, 2)")
    print("  Meaning: 5 future actions, each with 2 joint commands.")

    for index, action in enumerate(action_chunk):
        print(f"  t+{index}: {action}")

    print("""
A common approach is receding-horizon execution:

    1. Observe the robot and environment.
    2. Predict a chunk of future actions.
    3. Execute only part of the chunk.
    4. Observe the environment again.
    5. Predict a new chunk.

This allows the robot to correct its plan when the environment
or robot state changes.
""")


# ============================================================
# 10. MINI EXERCISE
# ============================================================

def mini_exercise():
    print("\n" + "=" * 65)
    print("10. MINI EXERCISE")
    print("=" * 65)

    print("""
Try to answer these questions:

1. What is the difference between a discrete and continuous action?

2. If a robot has six joints, how many values are needed for a
   joint-position action vector?

3. What does forward kinematics calculate?

4. What does inverse kinematics calculate?

5. Why can the same point have different coordinates in the
   robot base frame and world frame?

6. What is the difference between an action sequence and an
   action chunk?

7. Why might a robot execute only part of a predicted action chunk
   before observing the environment again?

EXPERIMENT:

Change the target in demonstrate_inverse_kinematics():

    target_x = 0.5
    target_y = 0.3

Try different reachable positions.

Then try:

    target_x = 1.5
    target_y = 0.0

The program should report that this position is unreachable.

Remember: the maximum reach of this arm is LINK_1 + LINK_2.
""")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "#" * 65)
    print("ROBOT ACTION REPRESENTATIONS")
    print("From machine-learning predictions to robot control")
    print("#" * 65)

    demonstrate_discrete_actions()
    demonstrate_continuous_actions()
    demonstrate_joint_state()
    demonstrate_forward_kinematics()
    demonstrate_inverse_kinematics()
    demonstrate_coordinate_systems()
    demonstrate_action_vectors()
    demonstrate_action_sequences()
    demonstrate_action_chunks()
    mini_exercise()

    print("\n" + "#" * 65)
    print("LESSON COMPLETE!")
    print("#" * 65)

    print("""
You have now connected high-level robot actions with numerical
representations used in robot control.

NEXT:
A robot policy must learn to predict these action representations
from visual observations and language instructions.
""")


if __name__ == "__main__":
    main()
