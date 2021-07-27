# Queue-Length-Estimation-using-RSSI-Deep-Learning

This github repository contains project component related to final year project of Department of Electronic and Telecommunication and Engineering, University of Moratuwa.

The big picture of the project is that build large scale intteligent transportation system. In this project it used technology related to wireless communication, Artificial Intelligence, Big data and Advanced Deployment Techniques.

# Project Components
The project contain several components which really courages to develop modern transportaion systems.
1. Vehicle Localization using Signal Strength.
2. Queue length estimation.
3. Inference of Deep Learning and Machine Learning Models on power constrained devices like Raspberry PI.
# Methodology
  - Using Deep Learning classfication models like MLP, 1D convolution first predict the current state of a vehicle.
  - Using that prediction estimate the queue length at a given time step.
  - Using Signal strength and Capture state information cluster pedestrian to analyze current state a road crossing.
  - All Machine learning and Deep Learning models deploy on Raspberry pi modules.
# Techniques
  - Convolutional Neural Networks
  - Multi layer perceptron Netwroks
  - Deep Autoencoder Networks
  - Tiny Machine Learning
# Tools
* TensorFlow - Deep Learning Model
* pandas - Data Extraction and Preprocessing
* numpy - numerical computations
* scikit learn - Advanced preprocessing and Machine Learning Models
* Omnet++ and Sumo - Wireless Simulations

### Installation

Install the dependencies and conda environment

```sh
$ conda create -n envname python=3.7
$ activate envname 
$ conda install -c anaconda tensorflow-gpu
$ conda install -c anaconda pandas
$ conda install -c anaconda matplotlib
$ conda install -c anaconda scikit-learn
$ conda install -c anaconda sqlite
```
