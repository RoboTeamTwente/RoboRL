# RoboRL

## What is RoboRL?
RoboRL is RoboTeamTwente's attempt to create a simple, easy-to-use, and scalable reinforcement learning environment for the small size league (SSL).
We use [MuJoCo](https://github.com/deepmind/mujoco) as our physics engine and [Brax](https://github.com/google/brax) as our RL framework. More specifically, we use the GPU accelerated version of mujoco called MujocoMJX.
This is a framework that can be used to train deep reinforcement learning algorithms [MujocoMJX](https://mujoco.readthedocs.io/en/stable/overview.html). The goal of this project is to introduce reinforcement learning to the small size league and set a foundation to adopt this technology in the future.

## Getting started
You need a couple of things to get started:

### Installing dependencies
You have to install the following packages:
```bash
pip install mujoco
pip install mujoco_mjx
pip install brax
```

### Pulling repository
```bash
git pull https://github.com/RoboTeamTwente/RoboRL.git
```

Install[MujocoMJX](https://mujoco.readthedocs.io/en/stable/overview.html).

Train the model by running the train.py file.
Be sure to modify the hyperparameters, and environment difficulty in RoboRLEnv.py.

### Running the visual simulator
To run the simulation go to /mujoco-3.3.0/bin and type ./simulate

### Troubleshooting
If the video is not saving try running export MUJOCO_GL=osmesa

#### RL
If the agent is not learning, try to play with the reward functions. Generally we found that we had to make it really easy for the agent to get any meaningful dense reward.
Also tune the hyperparameters. The ones currently in train.py should provide a good baseline.
To just run inference or restart from a checkpoint, fill in the checkpoint number into: restore_checkpoint_path=ckpt_path / '0' as well as model_path = ''

## Current progress
Currently we have finished the environment and have validated the physics validity to some extent. It is not glitching out or doing things it is not supposed to be doing.
We also have trained a single agent in a simple environment which can move to the ball, dribble and kick it into an empty goal from whereever on the field.

The environment itself is a 1:1 in terms of dimensions with respect to the real world soccer field.

## Codebase 
robot_model contains the xml files, as well as the python functions to easily generate these xml files (WARNING: the python XML generators are not updated).
mujoco-3.3.0 contains the Mujoco source code, to run the sim in /mujoco-3.3.0/bin
customPPO contains the files for a CTDE (centralized training, decentralized execution) framework. This essentially means that all robots are collectively training and running the same policy network, which is trained using a centralized value network, but each robot does execute it's own actions. We did not test this out, but it atleast runs without blatant errors.

## Modelling choices
To maintain a high SPS (Steps per second) rate, it is vital to simplify the physics tremendously. 
The omnidirectional drive for example, is not modelled precisely, but instead estimated using sliding joints.

## In hindsight
In hindsight Mujoco MJX was perhaps not the right tool for the job. While it is much much simpler to use than IsaacLabs, it has some severe limitations.
It is slow when there are a high number of possible collissions (like our Robocup soccer is, especially with many robots).
It also does not look like it's made for driving robots, but more for modelling joints/ robotics.

Authors from ETH Zurich have demonstrated a similar implementation, but in IsaacLabs which is successful. [Paper](https://arxiv.org/abs/2409.20326).

This project was a bit ad-hoc put together and done in three months, and we have not had the time to properly implement it for a more complex environment. 
And getting a version of IsaacLabs to work was definately going to take a lot of time.

## Tips and tricks

### Training
Training a continuous control policy is a bit tricky. The observation space is huge, which makes learning a challenge.
We had the best success by using curriculum learning. We started with putting a robot very close (like 0.5m) to the ball and gave it a dense reward for moving to the ball. Then we gradually increased the distance to the ball, and the reward for moving to the ball. Then we added a reward for kicking. This proved to be the only way to get a decent result.

### Debugging
If you have issues with GPU, you need to troubleshoot CUDA. It might help to do a conda install of the mujoco mjx package too.
