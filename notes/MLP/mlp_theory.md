# Mathematical Foundations of the Perceptron

This document provides a detailed derivation of the Perceptron Convergence Theorem and a solution to the XOR problem using a multi-layer architecture.

## 1. Perceptron Convergence Theorem Derivation

The goal is to prove that for any linearly separable dataset, the Perceptron algorithm will converge in a finite number of steps.

### 1.1 Assumptions and Definitions

Given a dataset $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with $y_i \in \{-1, 1\}$:

1. **Bounded Input:** $\|\mathbf{x}_i\| \leq r$ for all $i$.

2. **Linear Separability:** There exists an optimal vector $\mathbf{w}^*$ and bias $b^*$ such that $y_i(\mathbf{x}_i^\top \mathbf{w}^* + b^*) \geq \rho$ where $\rho > 0$.

3. **Normalization:** $\|\mathbf{w}^*\|^2 + (b^*)^2 \leq 1$.

**Augmented Notation:**
Let $\hat{\mathbf{w}} = [\mathbf{w}; b]$ and $\hat{\mathbf{x}}_i = [\mathbf{x}_i; 1]$.
The conditions become:

* $y_i(\hat{\mathbf{w}}^{*\top} \hat{\mathbf{x}}_i) \geq \rho$

* $\|\hat{\mathbf{w}}^*\|^2 \leq 1$

* $\|\hat{\mathbf{x}}_i\|^2 = \|\mathbf{x}_i\|^2 + 1 \leq r^2 + 1$

### 1.2 The Proof

Let $\hat{\mathbf{w}}_k$ be the weight vector after $k$ updates. The update rule on a mistake is:

$$
\hat{\mathbf{w}}_k = \hat{\mathbf{w}}_{k-1} + y_i \hat{\mathbf{x}}_i
$$

#### Part A: Lower Bound on the Projection

We calculate the growth of the projection of $\hat{\mathbf{w}}_k$ onto the direction of the optimal solution $\hat{\mathbf{w}}^*$:

$$
\hat{\mathbf{w}}_k^\top \hat{\mathbf{w}}^* = (\hat{\mathbf{w}}_{k-1} + y_i \hat{\mathbf{x}}_i)^\top \hat{\mathbf{w}}^* = \hat{\mathbf{w}}_{k-1}^\top \hat{\mathbf{w}}^* + y_i(\hat{\mathbf{x}}_i^\top \hat{\mathbf{w}}^*)
$$

Since $y_i(\hat{\mathbf{x}}_i^\top \hat{\mathbf{w}}^*) \geq \rho$, we have:

$$
\hat{\mathbf{w}}_k^\top \hat{\mathbf{w}}^* \geq \hat{\mathbf{w}}_{k-1}^\top \hat{\mathbf{w}}^* + \rho
$$

Starting from $\hat{\mathbf{w}}_0 = \mathbf{0}$, after $k$ updates:

$$
\hat{\mathbf{w}}_k^\top \hat{\mathbf{w}}^* \geq k\rho
$$

#### Part B: Upper Bound on the Norm

We calculate the growth of the squared norm $\|\hat{\mathbf{w}}_k\|^2$:

$$
\|\hat{\mathbf{w}}_k\|^2 = \|\hat{\mathbf{w}}_{k-1} + y_i \hat{\mathbf{x}}_i\|^2 = \|\hat{\mathbf{w}}_{k-1}\|^2 + 2y_i(\hat{\mathbf{w}}_{k-1}^\top \hat{\mathbf{x}}_i) + \|\hat{\mathbf{x}}_i\|^2
$$

Since an update occurs only if $y_i(\hat{\mathbf{w}}_{k-1}^\top \hat{\mathbf{x}}_i) \leq 0$:

$$
\|\hat{\mathbf{w}}_k\|^2 \leq \|\hat{\mathbf{w}}_{k-1}\|^2 + \|\hat{\mathbf{x}}_i\|^2
$$

Since $\|\hat{\mathbf{x}}_i\|^2 \leq r^2 + 1$:

$$
\|\hat{\mathbf{w}}_k\|^2 \leq k(r^2 + 1)
$$

#### Part C: Combining with Cauchy-Schwarz

From the Cauchy-Schwarz inequality:

$$
(\hat{\mathbf{w}}_k^\top \hat{\mathbf{w}}^*)^2 \leq \|\hat{\mathbf{w}}_k\|^2 \|\hat{\mathbf{w}}^*\|^2
$$

Substituting our results:

$$
(k\rho)^2 \leq k(r^2 + 1) \cdot 1
$$

$$
k^2 \rho^2 \leq k(r^2 + 1)
$$

$$
k \leq \frac{r^2 + 1}{\rho^2}
$$

## 2. Two-Layer Perceptron for XOR

The XOR function is defined as:

* (0,0) $\to$ 0

* (0,1) $\to$ 1

* (1,0) $\to$ 1

* (1,1) $\to$ 0

### 2.1 The Construction

We use a hidden layer with two neurons ($h_1, h_2$) and an output layer ($y$). We use the Step Function: $H(z) = 1$ if $z > 0$ else $0$.

| **Layer**      | **Type** | **Weights (w1​,w2​)** | **Bias (b)** |
| :------------- | :------- | :-------------------- | :----------- |
| Hidden ($h_1$) | OR       | $[1, 1]$              | $-0.5$       |
| Hidden ($h_2$) | NAND     | $[-1, -1]$            | $1.5$        |
| Output ($y$)   | AND      | $[1, 1]$              | $-1.5$       |

### 2.2 Mathematical Verification

Let's test the most difficult case: **Input (1, 1)**.

1. $h_1$ **(OR):** $H(1(1) + 1(1) - 0.5) = H(1.5) = 1$

2. $h_2$ **(NAND):** $H(-1(1) - 1(1) + 1.5) = H(-0.5) = 0$

3. $y$ **(AND):** $H(1(h_1) + 1(h_2) - 1.5) = H(1 + 0 - 1.5) = H(-0.5) = 0$

**Result:** Input (1, 1) $\to$ Output 0. (Correct)

### 2.3 Geometric Intuition

The hidden layer creates a new feature space $(h_1, h_2)$. In this new space, the points $(0,1)$ and $(1,0)$ both map to $(1,1)$, while $(0,0)$ and $(1,1)$ map to $(0,1)$ and $(1,0)$ respectively. This new representation is linearly separable by a single line (the AND gate).