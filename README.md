# Three-Layer Neural Network Implementations for Non-Linear Regression

This repository presents a comprehensive exploration of building a three-layer neural network for a non-linear regression task using various frameworks and techniques. The goal of the project was to demonstrate how to implement the same basic neural network using multiple approaches, ensuring a deep understanding of model construction, data processing, and training mechanisms across different libraries.

## Project Overview

The project focuses on constructing a neural network that consists of an input layer, two hidden layers, and one output layer, designed to learn a non-linear mapping from three input variables to a target value. The target is generated from a non-linear equation with added noise, and the input data is visualized with a 3D scatter plot (using color to represent the target).

## What Was Accomplished

- **Data Generation and Visualization:**  
  A synthetic dataset was created using a defined non-linear function involving three input features. The data is visualized in 3D, with an additional dimension represented by the target value, providing a clear insight into the structure and variability of the generated data.

- **NumPy-Only Implementation:**  
  A fully from-scratch implementation was developed using only NumPy. This version includes manual computation of the forward pass, loss evaluation, and backpropagation with gradient descent—highlighting the underlying mechanics of neural networks.

- **PyTorch Implementations:**  
  Multiple approaches were explored:
  - A from-scratch implementation using PyTorch tensors and manual backpropagation without relying on built-in layers.
  - A class-based implementation utilizing PyTorch's `nn.Module` for a modular, object-oriented design.
  - A PyTorch Lightning version to demonstrate a cleaner, higher-level abstraction for model training, logging, and checkpointing.

- **TensorFlow Implementations:**  
  Various TensorFlow approaches were implemented:
  - **Low-Level:** A model constructed using low-level TensorFlow operations (tf.Variable, tf.matmul, tf.GradientTape) to manually handle the training loop.
  - **Built-In Layers:** A solution using tf.keras built-in layers (via the Sequential API) to simplify model development.
  - **Functional API:** A model built using the Functional API to provide flexibility in defining complex architectures.
  - **High-Level API:** A straightforward implementation using the high-level tf.keras.Sequential API, emphasizing ease of use and rapid prototyping.

- **JAX Implementation (Planned/Optional):**  
  An additional implementation using JAX was considered to further extend the exploration of different computational frameworks for neural network training.

## Educational Objectives

The project was designed to:
- Illustrate how the same neural network architecture can be implemented across different programming environments and libraries.
- Enhance understanding of low-level operations such as forward propagation and backpropagation.
- Compare the advantages of using high-level APIs that abstract away many of the implementation details.
- Demonstrate proper data handling techniques including normalization, train/test splitting, and visualization.
- Provide a basis for further exploration and experimentation with alternative frameworks such as JAX.

This repository serves as both a learning tool and a reference for building and understanding neural network models from the ground up as well as using high-level libraries for rapid model development.

---
