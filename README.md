# Three-Layer Neural Network Implementations for Non-Linear Regression

This repository presents a comprehensive exploration of building a three-layer neural network for a non-linear regression task using various frameworks and techniques. The goal of the project was to demonstrate how to implement the same basic neural network using multiple approaches, ensuring a deep understanding of model construction, data processing, and training mechanisms across different libraries.


## What Was Accomplished

- **Data Generation and Visualization:**  
  A synthetic dataset was created using a defined non-linear function involving three input features. The data is visualized in 3D, with an additional dimension represented by the target value, providing a clear insight into the structure and variability of the generated data.

- **NumPy-Only Implementation:**  
  A fully from-scratch implementation was developed using only NumPy. This version includes manual computation of the forward pass, loss evaluation, and backpropagation with gradient descent—highlighting the underlying mechanics of neural networks.

https://github.com/intimanjunath/Neural-network-using-numpy-keras-pytorch-and-jax/blob/main/1_NumPy_neural_network.ipynb 

- **PyTorch Implementations:**  
  Multiple approaches were explored:
  - A from-scratch implementation using PyTorch tensors and manual backpropagation without relying on built-in layers.
  - https://github.com/intimanjunath/Neural-network-using-numpy-keras-pytorch-and-jax/blob/main/2_PyTorch_3layer_neural_network.ipynb 
  - A class-based implementation utilizing PyTorch's `nn.Module` for a modular, object-oriented design.
  - https://github.com/intimanjunath/Neural-network-using-numpy-keras-pytorch-and-jax/blob/main/3_PyTorch_neural_network_builtin_modules.ipynb 
  - A PyTorch Lightning version to demonstrate a cleaner, higher-level abstraction for model training, logging, and checkpointing.
  - https://github.com/intimanjunath/Neural-network-using-numpy-keras-pytorch-and-jax/blob/main/4_PyTorch_Lightning.ipynb

- **TensorFlow Implementations:**  
  Various TensorFlow approaches were implemented:
  - **Low-Level:** A model constructed using low-level TensorFlow operations (tf.Variable, tf.matmul, tf.GradientTape) to manually handle the training loop.
  - https://github.com/intimanjunath/Neural-network-using-numpy-keras-pytorch-and-jax/blob/main/6_tensorflow_from_scratch.ipynb
  - **Built-In Layers:** A solution using tf.keras built-in layers (via the Sequential API) to simplify model development.
  - https://github.com/intimanjunath/Neural-network-using-numpy-keras-pytorch-and-jax/blob/main/7_tensorflow_builtin_layers.ipynb
  - **Functional API:** A model built using the Functional API to provide flexibility in defining complex architectures.
  - https://github.com/intimanjunath/Neural-network-using-numpy-keras-pytorch-and-jax/blob/main/8_functional_api_tensorflow.ipynb
  - **High-Level API:** A straightforward implementation using the high-level tf.keras.Sequential API, emphasizing ease of use and rapid prototyping.
  - https://github.com/intimanjunath/Neural-network-using-numpy-keras-pytorch-and-jax/blob/main/9_tensorflow_high_level_api_.ipynb

- **TensorFlow ALL varients:**
- https://github.com/intimanjunath/Neural-network-using-numpy-keras-pytorch-and-jax/blob/main/5_Tensorflow_various_variants.ipynb

## Educational Objectives

The project was designed to:
- Illustrate how the same neural network architecture can be implemented across different programming environments and libraries.
- Enhance understanding of low-level operations such as forward propagation and backpropagation.
- Compare the advantages of using high-level APIs that abstract away many of the implementation details.
- Demonstrate proper data handling techniques including normalization, train/test splitting, and visualization.
- Provide a basis for further exploration and experimentation with alternative frameworks such as JAX.

This repository serves as both a learning tool and a reference for building and understanding neural network models from the ground up as well as using high-level libraries for rapid model development.


## YouTube link : https://youtu.be/Vyj3I8uByEk
---
