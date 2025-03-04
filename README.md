# DS 598 Final Report: Team Rocket

This repository contains the code, experiments, and documentation for our DS 598 project on the Lux AI Challenge. In this competition, our goal was to develop an intelligent agent capable of thriving in a dynamic 1v1 arena by effectively managing resources, building cities, and outmaneuvering an opponent. Our approach evolved from an initial Proximal Policy Optimization (PPO) strategy to an enhanced Imitation Learning (IL) method that leverages a UNet architecture with spatial and channel attention mechanisms.

## Project Overview

The Lux AI Challenge provided a platform to explore advanced reinforcement learning and imitation learning techniques. Our solution comprises three key phases:

- **Initial Strategy with PPO:**  
  We started with a PPO-based agent, designing a reward system to incentivize efficient resource collection, strategic resource transformation, and research advancements.

- **Enhanced Modeling with Imitation Learning:**  
  By incorporating imitation learning, we leveraged expert strategies from top competitors. Our input representation included both local features (capturing immediate surroundings) and global features (aggregating overall game state). This approach significantly improved our agent's performance.

- **Final Submission with UNet and Attention:**  
  For our final submission, we built upon our midterm results by integrating enhanced spatial and channel attention mechanisms into a UNet model. These modifications allowed our agent to more effectively capture strategic spatial relationships and resource distributions.

## Installation Requirements
Please ensure the following packages are installed:
- `luxai2021`
- `stable_baselines3`
- `torch`

You can install these packages using pip:
```bash
pip install luxai2021 stable_baselines3 torch
```
## PPO Folder Contents
- PPO Training Notebook: Use this Jupyter notebook to train the agent. It will also save the trained model upon completion.
- `PPO/PPO Training.ipynb`
- Agent Policy File: This script contains the agents policy. Note that it will be overwritten each time the PPO Training notebook is run.
- `PPO/agent_policy.py`
- Saved Model: After running the training for 12 million timesteps, the model is saved in this directory.
- `PPO/rl_model_1_12000000_steps.zip`
- Main.py: Use this script to run the model against other competitors. The script references the agent policy and model file, which can be modified as required.
- `PPO/main.py`

## IL Midterm Folder Contents
- Agent Policy
- `Imitation Learning Midterm/agent.py`
- Trained Model
- `Imitation Learning Midterm/model_il.pth`

## IL Final Folder Contents
- File to train the model
- `Imitation Learning Final/train.py`
- Agent Policy
- `Imitation Learning Final/agent.py`
- UNet Model File (Class)
- `Imitation Learning Final/unet_model.py`
- UNet Parts (all blocks of the UNet model, i.e, Conv Blocks, Decoder, Encoder, Attention Maps etc.)
- `Imitation Learning Final/unet_parts.py`

## Experimental Results

Our ablation studies highlight the effectiveness of our various strategies:

| Model                              | Opponent                    | Win Rate (%) |
| ---------------------------------- | --------------------------- | ------------ |
| PPO                                | Midterm Champion            | 12           |
| Imitation Learning (IL)            | Midterm Champion            | 54           |
| Imitation Learning (IL)            | PPO                         | 100          |
| Final IL (with Attention)          | Midterm Champion            | 94           |
| Final IL (with Attention)          | Previous IL Version         | 87           |

These results demonstrate significant improvements from our baseline PPO approach to our enhanced IL model.


## References and Credits

- [PPO Implementation on Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#)  
- [Kaggle Notebook by @glmcdona on PPO](https://www.kaggle.com/code/glmcdona/reinforcement-learning-openai-ppo-with-python-game#Define-the-RL-agent-logic)  
- [Lux AI Replay Dataset](https://www.kaggle.com/datasets/bomac1/luxai-replay-dataset/data)  
- [UNet with Imitation Learning on Kaggle](https://www.kaggle.com/code/bachngoh/luxai-unet-immitationlearning/notebook)  



