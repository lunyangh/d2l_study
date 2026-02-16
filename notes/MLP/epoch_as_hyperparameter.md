# Formal Notes: Optimization as Iterative Regularization

## 1. The Procedural Definition of a Model
In classical theory, a hypothesis space $\mathcal{H}$ is defined by the architecture. In a procedural framework, we redefine the "model" to include the optimization process:
- **The Estimator:** $G(S, \eta, t, w_0)$
  - $S$: Training data
  - $\eta$: Learning rate
  - $t$: Number of iterations (Epochs)
  - $w_0$: Initialization (Starting point)

**Key Insight:** The "Effective Hypothesis Space" $\mathcal{H}_t$ is the set of all weight configurations reachable from $w_0$ in exactly $t$ steps.

## 2. Epochs vs. VC Complexity
There is a direct correlation between the number of training iterations and the model's capacity to shatter data points (VC Dimension).

| Training Phase             | Complexity ($d_{VC}$)       | Generalization Gap | Description                                                                           |
| :------------------------- | :-------------------------- | :----------------- | :------------------------------------------------------------------------------------ |
| **Early ($t \ll \infty$)** | Low (Small $\mathcal{H}_t$) | Small              | The optimizer has only captured coarse, global features. The model is "stable."       |
| **Late ($t \to \infty$)**  | High (Full $\mathcal{H}$)   | Large              | The optimizer has tracked high-frequency noise and outliers. The model is "unstable." |

**Early Stopping** is therefore a form of **Structural Risk Minimization (SRM)**, where we choose the index $t$ that minimizes the bound:
$$R(h_t) \leq \hat{R}(h_t) + \mathcal{C}(\mathcal{H}_t, n)$$
where $\mathcal{C}$ is a complexity penalty that grows monotonically with $t$.

## 3. Regularization: The "Tax" on Trajectory
While epochs limit complexity by stopping the search early, explicit regularization (Weight Decay) limits complexity by "warping" the search space.

### L2 Regularization (Weight Decay)
- **Math:** $J(\theta) + \lambda \|w\|^2$
- **Effect:** Acts as a "restoring force" toward the origin.
- **Complexity Link:** It shrinks the radius of the reachable hypothesis space. Mathematically, for linear models, there is a formal **Equivalence Theorem** stating that for every early stopping point $t$, there exists a corresponding $\lambda$ that produces a nearly identical solution.



## 4. Algorithmic Stability
Modern theory often replaces VC Dimension with **Uniform Stability**.
- An algorithm is $\beta$-stable if changing one training point changes the output by at most $\beta$.
- **SGD Stability:** It has been proven that SGD is stable for a small number of epochs. As $t$ increases, the stability $\beta$ increases, which directly leads to a larger generalization error.

## 5. The Modern Phenomenon: Double Descent
While the U-shaped curve (VC Bound) suggests that increasing epochs indefinitely leads to catastrophic overfitting, modern deep learning observes a "second descent."
- **Phase 1:** Classical U-shaped curve (Overfitting occurs).
- **Phase 2:** The "Interpolation Threshold" (Training error hits zero).
- **Phase 3:** Second Descent (The model finds a "simplest" smooth solution among all that fit the data perfectly).