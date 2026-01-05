# XOR Neural Network in C++  

A **feedforward neural network implemented from scratch in C++** to solve the classic XOR problem using **sigmoid activation**, **backpropagation**, and **gradient descent**.

---

## Project Overview

The XOR problem is **non-linearly separable**, making it a canonical example for neural network learning. This implementation demonstrates how a network can approximate a function \( f: \mathbb{R}^2 \to \mathbb{R} \) mapping binary inputs to outputs.  

**Network architecture:**

- Input layer: \( x \in \mathbb{R}^2 \)  
- Hidden layer: 2 neurons (\( h_1, h_2 \))  
- Output layer: 1 neuron (\( \hat{y} \in [0,1] \))  

The network uses **sigmoid activation**:  

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]  

to introduce non-linearity and allow modeling of XOR.

---

## Mathematical Formulation

1. **Forward Pass:**

\[
h_j = \sigma\left(\sum_{i=1}^{2} w_{ij}^{(1)} x_i + b_j^{(1)}\right), \quad j = 1,2
\]  

\[
\hat{y} = \sigma\left(\sum_{j=1}^{2} w_j^{(2)} h_j + b^{(2)}\right)
\]  

where \( w_{ij}^{(1)}, w_j^{(2)} \) are weights and \( b_j^{(1)}, b^{(2)} \) are biases.

2. **Loss Function (Binary Cross-Entropy):**

\[
\mathcal{L}(y, \hat{y}) = - \big[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \big]
\]  

3. **Backpropagation (Gradient Descent):**

\[
w \gets w - \eta \frac{\partial \mathcal{L}}{\partial w}, \quad
b \gets b - \eta \frac{\partial \mathcal{L}}{\partial b}
\]  

where \( \eta \) is the **learning rate**.

4. **Training Objective:**  

Minimize the total loss over all 4 XOR inputs:

\[
\min_{w,b} \sum_{(x,y)\in \text{XOR}} \mathcal{L}(y, \hat{y})
\]

---

## Example Output

