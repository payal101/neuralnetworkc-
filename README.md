# XOR Neural Network in C++

A **feedforward neural network implemented from scratch in C++** to solve the XOR problem using **sigmoid activation**, **backpropagation**, and **gradient descent**.

---

## Project Overview

The XOR problem is **non-linearly separable**, requiring a neural network with at least one hidden layer to approximate the mapping:

$$
f: \mathbb{R}^2 \to \mathbb{R}, \quad (x_1, x_2) \mapsto y \in \{0,1\}.
$$

The network architecture used:

- **Input layer:** 2 neurons  
- **Hidden layer:** 2 neurons  
- **Output layer:** 1 neuron  

The network uses the **sigmoid activation function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

which introduces non-linearity, allowing the network to model XOR.

---

## Forward Pass

Let the input vector be 
$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$, weights $$\mathbf{W}^{(1)} \in \mathbb{R}^{2\times2}$, biases $\mathbf{b}^{(1)} \in \mathbb{R}^{2}$$ for the hidden layer, and $$\mathbf{W}^{(2)} \in \mathbb{R}^{1\times2}$$, $$b^{(2)} \in \mathbb{R}$$for the output layer.  

Hidden layer activations:

$$
\mathbf{h} = \sigma(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)})
$$

Output:

$$
\hat{y} = \sigma(\mathbf{W}^{(2)} \mathbf{h} + b^{(2)})
$$

---

## Loss Function

We use **binary cross-entropy**:

$$
\mathcal{L}(y, \hat{y}) = - \big[ y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \big]
$$

Training objective over all 4 XOR inputs $(\mathbf{x}^{(i)}, y^{(i)})$:

$$
\min_{\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \mathbf{W}^{(2)}, b^{(2)}} \sum_{i=1}^{4} \mathcal{L}(y^{(i)}, \hat{y}^{(i)})
$$

---

## Backpropagation

**1. Output layer gradient:**

$$
\delta^{(2)} = \hat{y} - y
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(2)}} = \delta^{(2)} \mathbf{h}^\top, \quad
\frac{\partial \mathcal{L}}{\partial b^{(2)}} = \delta^{(2)}
$$

**2. Hidden layer gradient:**

$$
\delta^{(1)} = (\mathbf{W}^{(2)})^\top \delta^{(2)} \odot \sigma'(\mathbf{h}), \quad
\sigma'(\mathbf{h}) = \mathbf{h} \odot (1-\mathbf{h})
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \delta^{(1)} \mathbf{x}^\top, \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(1)}} = \delta^{(1)}
$$

**3. Weight updates (Gradient Descent):**

$$
\mathbf{W}^{(l)} \gets \mathbf{W}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}, \quad
\mathbf{b}^{(l)} \gets \mathbf{b}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}, \quad l = 1,2
$$

where $\eta$ is the **learning rate**.

---

## Example Output
0 XOR 0 ≈ 0.01
0 XOR 1 ≈ 0.99
1 XOR 0 ≈ 0.98
1 XOR 1 ≈ 0.02


The network successfully approximates the XOR function.


