# VLA Crash Course

A hands-on course for robot developers. This repository's `step1` is the
starting point: it introduces the deep-learning and robot-control ideas that
later VLA lessons build on.

## Why does a VLA course start here?

A Vision-Language-Action (VLA) system ultimately needs to turn what a robot
**sees** and what a person **asks** into an **action**. Before learning how to
connect images and language to actions, it helps to understand how a model
learns a task and how its predictions can control a robot.

That is what Step 1 teaches, in small, inspectable examples. We first teach a
model a simple numeric relationship, then move from predictions to robot
actions, navigation, and finally obstacle avoidance. The grid-world is a
teaching environment: it makes the action-learning loop visible without
requiring a camera or a physical robot.

By the end of Step 1, you should be able to explain how a model learns from
examples, train a small PyTorch model, use its predictions to choose robot
movements, and test those movements in a simulation. This step does not yet
build a complete VLA system: it does not process images or language
instructions.

## Step 1 learning path

Follow the files in this order. Each exercise adds one idea needed to
understand how a learned model can eventually become part of a robot's
perception-to-action pipeline.

1. **First, understand the basic idea of learning.** In
   [`zero.py`](./step1/zero.py), we give a neural network example input-output
   pairs for the relationship `y = 2x + 1`. The model starts with parameters it
   has not learned yet; training adjusts them so its predictions match the
   examples. This is a small introduction to deep learning: the model learns a
   relationship from data instead of having the answer explicitly programmed
   as a rule.
2. **Next, connect a robot's situation to an action.** In
   [`first.py`](./step1/first.py), a hand-written policy maps the target's
   position to `left`, `forward`, or `right`. This gives us a simple example of
   the kind of decision a robot must make, and a comparison point for a model
   that learns decisions from examples.
3. **Then, see how model training works.** [`second.py`](./step1/second.py)
   shows the prediction, loss, gradient, and optimizer steps that teach a
   model. [`second_exercise.py`](./step1/second_exercise.py) applies this
   training pattern to a basic position-to-action relationship.
4. **Use a model to navigate.** [`third.py`](./step1/third.py) trains a model
   to predict whether a robot should move left, stay, or move right.
   [`assessment.py`](./step1/assessment.py) and
   [`assessment2.py`](./step1/assessment2.py) extend the idea to two
   dimensions: given the robot and target positions on a 5 × 5 grid, predict
   movement along both axes.
5. **Run the learned policy and observe what it does.**
   [`simulator.py`](./step1/simulator.py) puts the model in a Pygame grid-world.
   Press **Space** for one AI step, **A** to toggle automatic movement, **R** to
   reset, click a cell to move the target, and **Esc** to quit. This demonstrates
   the control loop: the model predicts an action, the environment changes,
   and the model acts again.
6. **Finally, introduce planning around obstacles.**
   [`BFS.md`](./step1/BFS.md) explains breadth-first search, a path-finding
   algorithm. Use [`fourth_terminal.py`](./step1/fourth_terminal.py) to see
   obstacle-aware navigation in the terminal, then
   [`fourth.py`](./step1/fourth.py) for a visual exercise.

## Technical setup

Run scripts from the repository root with Python and the required packages
installed in your active environment. For example:

```bash
python step1/zero.py
python step1/third.py
python step1/fourth_terminal.py
```

The exercises use PyTorch; the obstacle-navigation examples also use NumPy.
Both are listed in [`requirements.txt`](./requirements.txt). The visual demos
also require Pygame, which is not currently listed there.
[`simulator.md`](./step1/simulator.md) includes offline Pygame installation
commands.

## The connection to the rest of VLA

In the examples here, the model's input is a simple numeric description of the
robot and its target, and its output is a movement. In a later VLA system, the
input can include camera images and a language instruction, while the output
is still an action the robot can take. The scale and inputs become more
complex, but the basic question introduced in Step 1 remains: **how can a
model use what it is given to choose an appropriate action?**
