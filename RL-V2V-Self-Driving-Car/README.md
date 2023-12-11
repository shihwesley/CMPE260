# CMPE 260 - Reinforcement Learning
# Team : Petabyte Transformers
# Project : Training a Self Driving Car agent to use V2V data along with vision data to learn to achieve higher speeds on freeways


This guide demonstrates how to make use of V2V data to help a self driving car agent using DQN.

## Initialize

The following instructions demonstrate how to initialize your system to access the HPC.


### Configure Conda environment

If you haven't installed conda, you may need to do that first.

Run the following to activate conda environment:

conda activate <environment-name>

### Requirements

Install the packages from requirements.txt

python 3.7
tensorflow 1.15


## Run gui.py

Now, run the code via:

python gui.py


### Track via tensorboard

In order to track the training process locally:

conda activate environment-name

tensorboard --logdir= "location-of-the-log-directory"

## Demo


https://github.com/shihwesley/CMPE260/assets/112594201/eb8b0beb-f305-47c5-a5c8-97f144062859



Project Report : https://docs.google.com/document/d/1FWF8hmMQhUjBLOTVy9CuVcU04-b4yQs9/edit?usp=sharing&ouid=103663966554730930942&rtpof=true&sd=true
