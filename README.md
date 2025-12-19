Neural Network from Scratch in C++
Overview

This project implements a fully connected neural network from scratch in C++, without using any machine learning libraries.
The goal was to deeply understand how neural networks work internally by manually implementing the mathematical and algorithmic components.

Motivation

Most ML frameworks abstract away the learning process.
This project was built to:

Understand forward propagation and backpropagation mathematically

Implement gradient descent manually

Gain confidence in building ML systems at a low level using C++

**Features**

Fully connected feedforward neural network

Manual implementation of:

Forward propagation

Backpropagation

Gradient descent optimization

Configurable network architecture (layers, neurons)

Trained and tested on logical datasets such as XOR
**
Tech Stack**

Language: C++

Libraries: Standard C++ STL only

No external ML frameworks used

Network Architecture

Input Layer

One or more Hidden Layers

Output Layer

Activation functions implemented manually

Loss computed and minimized using gradient descent

Training Process

Initialize weights and biases randomly

Perform forward propagation to compute predictions

Calculate error using loss function

Apply backpropagation to compute gradients

Update weights using gradient descent

Repeat over multiple epochs

Example Output
0 XOR 0 = 0.04
0 XOR 1 = 0.85
1 XOR 0 = 0.85
1 XOR 1 = 0.06


(Values approximate correct XOR behavior)

What I Learned

How neural networks learn at a mathematical level

Importance of weight initialization and learning rate

Numerical stability and floating-point precision in C++

Writing modular, readable code for complex systems




**How to Run**
g++ nn_xor.cpp -o nn_xor
./nn_xor
**Helpful Links**

https://home.agh.edu.pl/~vlsi/AI/xor_t/en/main.htm?source=post_page-----83e35a22c96f---------------------------------------

Project Status

 Completed core implementation
 Open to future extensions

