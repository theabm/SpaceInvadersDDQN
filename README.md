# SpaceInvadersDDQN

An implementation of Double Deep Q-Learning for the course of Reinforcement Learning @ DSSC AA 2021-2022

This respository aims at solving the space invaders game using DDQN with some minor tweaks such as soft updating and using some skipping in order to increase learning.

The notebook for the code can be found under the name 'AtariRLProject.ipynb' and can be run using google colab.

# Environment setup:

from the termminal line, run:

  - pip3 install gym

  - pip3 install pytorch

  - pip3 install scikit-image

  - pip3 install numpy

  - pip3 install atary_py

  - pip3 install gym[atari]

The roms for space invaders are found under Utils. The link with all the roms from atari can be found in the python notebook.

# Code Folder
Although the code was initially run in google colab, it didnt have enough RAM to run over 400 episodes (with 1 million transitions stored in memory).
To be able to run in colab for more than 400 episodes, decreasing the size of the experience replay memory is highly beneficial.

Since we initially wanted to replicate the results from the DeepMind paper, we decided to use a supercomputer at our university. 
So the folder Code contains all the appropriate folders to run the code. 

Be careful of the absolute paths where the checkpoints are saved for the memory class and for the DQNagent class. 


# Run

To run, either run the notebook in google colab ensuring that the folder structure is as outlined in the notebook. 
Alternatively, using a local maching simply run using:

  - python3 DQN.py 

# Results

The results can be found in the results folder. They are also outline in the notebook. 

