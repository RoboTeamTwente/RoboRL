# RoboRL

## What is RoboRL?
RoboRL is RoboTeamTwente's attempt to create a simple, easy-to-use, and scalable reinforcement learning environment for the small size league (SSL).
We use [MuJoCo](https://github.com/deepmind/mujoco) as our physics engine and [Brax](https://github.com/google/brax) as our RL framework. More specifically, we use the GPU accelerated version of mujoco called MujocoMJX.
This is a framework that can be used to train deep reinforcement learning algorithms [MujocoMJX](https://mujoco.readthedocs.io/en/stable/overview.html).

## Getting started
Pull the repo:

```bash
git pull https://github.com/RoboTeamTwente/RoboRL.git
```

Train the model by running the train.py file.
Be sure to modify the hyperparameters, and environment difficulty in RoboRLEnv.py.

### Running the visual simulator
To run the simulation go to /mujoco-3.3.0/bin and type ./simulate

## Current progress
Currently we have finished the environment and have validated the physics validity to some extent. It is not glitching out or doing things it is not supposed to be doing.
We also have trained a single agent in a simple environment which can move to the ball, dribble and kick it into an empty goal from whereever on the field.

The environment itself is a 1:1 in terms of dimensions with respect to the real thing.

## Codebase 
robot_model contains the xml files, as well as the python functions to easily generate these xml files (WARNING: the python XML generators are not updated).
mujoco-3.3.0 contains the Mujoco source code, to run the sim in /mujoco-3.3.0/bin
customPPO contains the files for a CTDE (centralized training, decentralized execution) framework. This essentially means that all robots are collectively training and running the same policy network, which is trained using a centralized value network, but each robot does execute it's own actions.

## Modelling choices
To maintain a high SPS (Steps per second) rate, it is vital to simplify the physics tremendously. 
The omnidirectional drive for example, is not modelled precisely, but instead estimated using sliding joints.

For some reason Mujoco MJX is much slower (comparatively) than Isaac Labs (another RL framework), averaging about 15.000 sps.

## In hindsight
In hindsight Mujoco MJX was perhaps not the right tool for the job. While it is much much simpler to use than IsaacLabs, it has some severe limitations.
It is slow when there are a high number of possible collissions (like our Robocup soccer is, especially with many robots).
It also does not look like it's made for driving robots, but more for modelling joints/ robotics.

Authors from ETH Zurich have demonstrated a similar implementation, but in IsaacLabs which is successful. [Paper](https://arxiv.org/abs/2409.20326)

