# Deep Dive: MLP Hyperparameter Tuning & Architectural Logic

## 1. Architectural Design Patterns (Refining your Notes)

When working with an input dimension of **128**, the structural choice determines the "Information Capacity" and "Gradient Stability" of the model.

### A. The "Wide" Strategy (Width over Depth)
* **Mechanics:** Uses a single, large hidden layer.
* **Why it works:** According to the *Universal Approximation Theorem*, a single hidden layer with enough neurons can approximate any continuous function.
* **The 128-Input Context:** * Setting units to **256** (2x input) allows the model to learn complex combinations of features in a high-dimensional space before mapping to the output.
    * **Pros:** Easier to train (no vanishing gradients); fewer hyperparameters to manage.
    * **Cons:** Prone to memorization (overfitting) if not regularized.

### B. The "Deep & Tapered" Strategy (Information Bottleneck)
* **Mechanics:** Progressively reducing layer width (e.g., $128 \to 64 \to 32 \to 16 \to 8$).
* **The "Initial Expansion" Exception:** As you noted, starting with $128 \to 256$ is often beneficial. This is a **feature projection** step—it creates more "room" for the model to disentangle features before the compression begins.
* **The Tapering Rule:** Once compression starts (Layer 2 onwards), we never increase width again.
    * *Mathematical Intuition:* If you compress data into a 16-dimensional space, you have discarded information. Expanding to 64 in the next layer adds "empty" parameters that cannot reconstruct the discarded data, leading to wasted computation and potential noise.

---
Another note

## 2. Search Methodologies: Why Random beats Grid

### The "Grid Search" Trap
Grid search treats all hyperparameters as equally important. In an MLP, the **Learning Rate ($\eta$)** is usually 10x more important than the **Activation Function**.
* In a $10 \times 10$ grid, you spend 90% of your time testing different activations on "bad" learning rates.
* You only ever test 10 unique values of $\eta$.



### The "Random Search" Advantage
* **Effective Sample Size:** If you run 100 random trials, you test **100 unique learning rates** and **100 unique architectures**.
* **The "Peak" Discovery:** Because random search samples continuously, it is far more likely to hit the specific "sweet spot" (e.g., $\eta = 0.0074$) that a fixed grid (0.01, 0.001) would skip over.

---

## 3. Post-Search Analysis: Finding the Signal in the Noise

Since you aren't holding variables constant, you use statistical visualization to "review" results:

### A. Parallel Coordinates Plot
This is the "Flight Path" of your trials.
* **Visualization:** Each vertical axis is a hyperparameter; each line is a trial.
* **Analysis:** If all high-performing trials (colored gold) pass through a tiny window on the "Learning Rate" axis but are scattered across the "Layer 1 Width" axis, you conclude: *Learning Rate is critical; Layer 1 Width is flexible.*



### B. Slice/Contour Plots
These show the "Objective Value" (Accuracy) against 1 or 2 parameters.
* **Goal:** To find the **Optimal Range**.
* **Detection:** You look for a "U-shape" (in loss) or a "Mountain" (in accuracy). If the mountain is wide, the parameter is "robust." If the mountain is a sharp needle, the parameter is "volatile" and needs careful tuning.

---

## 4. Modern "Smart" Tuning (Bayesian Optimization)

Instead of pure randomness, tools like **Optuna** use the results of Trial 1-10 to "guess" where Trial 11 should be.
1.  **Exploitation:** It sees that "Option 2" (Tapered) is performing well, so it tries slight variations of that taper.
2.  **Exploration:** It occasionally tries a "Wide" architecture to ensure it hasn't missed a better global solution.
3.  **Pruning (Early Stopping):** It monitors the learning curve. If a random trial starts with 10% accuracy and doesn't move, Optuna kills the trial at Epoch 5 to save resources.

---

## 5. Summary Checklist for Your 128-Input MLP
1.  **Priority 1:** Find the **Learning Rate** range (usually $10^{-5}$ to $10^{-1}$ on a log scale).
2.  **Priority 2:** Compare **Option 1 (Wide)** vs **Option 2 (Tapered)**.
3.  **Priority 3:** If overfitting occurs, tune **Dropout** (range: 0.1 to 0.5) or **Weight Decay**.
4.  **Priority 4:** Finalize **Batch Size** (typically powers of 2: 32, 64, 128, 256).
5.


# Deep Learning Insights: Architecture & Activations

## 1. Wide vs. Deep Architectures

### The "Wide" Approach (Single Hidden Layer)
* **Theoretical Power:** Possesses **Universal Approximation** capabilities. A single hidden layer can represent any continuous function *if* it is wide enough.
* **The Problem (Efficiency):** To match the performance of a deep network, a wide network often requires an **exponential number of neurons**.
* **Training Dynamics:**
    * **Harder to Train:** Lacks the "scaffolding" of multiple layers, making the optimization landscape flatter and more difficult to navigate.
    * **Overfitting Risk:** Tends to act like a giant **lookup table**, memorizing specific input-output pairs (noise) rather than learning rules.

### The "Deep" Approach (Multi-Layer)
* **Inductive Bias:** Depth introduces a structural bias that assumes the world is **compositional** (complex things are built from simple things).
* **Hierarchical Learning:**
    * *Lower Layers:* Detect simple patterns (e.g., edges).
    * *Middle Layers:* Combine patterns into parts (e.g., shapes).
    * *Higher Layers:* Assemble parts into whole concepts (e.g., objects).
* **Efficiency:** Can represent complex functions using significantly fewer parameters ($O(n)$ vs $O(2^n)$) by reusing features across the hierarchy.

**Key Insight:** While both have similar *theoretical* representation power, Deep Learning prevails because its structure aligns with how information is naturally organized (hierarchy), making it easier to learn generalizable rules.

---

## 2. Activation Functions

### The Rule of Thumb
* **"Choice typically does not matter much":** This usually applies to choosing between **modern** non-saturating functions (e.g., ReLU, Leaky ReLU, GELU, Swish). The performance difference between these is often marginal (<1-2%).

### The Exception (History)
* **Legacy Functions (Sigmoid, Tanh):** The choice *does* matter if you compare modern functions to older ones like Sigmoid. Using Sigmoid in deep networks causes the **Vanishing Gradient Problem**, making training impossible.

**Takeaway:** Focus on architecture and data quality rather than obsessing over which specific variant of ReLU to use.

---

Another notes on how to tune hyperparameter in an example. And this discussion is expanded by gemini below

## example of tuning 128 input layer

Try following things in order: (graudally increase architecture complexity)
1. try without hidden layer
2. adding one hidden layer, try dimension {16, 32,64, 128}
    1. see how many dimension in hidden layer does better
        1. is 16 too simple to capture
        2. is 128 too rich to overfit
3. try two hidden layer
    1. based on experiment 2, if 16,128 is bad, we can do first hidden 32, second hidden 16
    2. we can also try other combination
        1. first hidden 64, second hidden {16,32}
    3. intuition is using experiment 2 to pin down reasonable range of n_dim for first hidden layer and second layer should further reduce dim based on n_hidden_1.


# Neural Network Tuning Strategy: A Constructive Approach

This document outlines a systematic, "greedy" approach to finding the optimal neural network architecture. The core philosophy is **Constructive Architecture Search**: start with the simplest possible model (Occam's Razor) and only add complexity (width and depth) when the performance specifically justifies it.

---

## Phase 1: The Linear Baseline
**Action:** Train a model with **0 Hidden Layers** (Input $\to$ Output).

### The Rationale: The "Linearity Check"
Before deploying a complex neural network, you must determine if the problem is actually simple. A network with zero hidden layers is mathematically equivalent to **Logistic Regression** (for classification) or **Linear Regression** (for regression).

### Interpreting Results
* **High Accuracy (e.g., >90%):** The problem is **Linearly Separable**.
    * *Conclusion:* You do not need a neural network. Using deep learning here is a waste of resources and increases the risk of overfitting.
* **Low Accuracy (e.g., 50-60%):** The problem is non-linear.
    * *Conclusion:* You have established a **Baseline**. Any future model with hidden layers must significantly beat this score to justify its computational cost.

---

## Phase 2: Tuning Width (Capacity)
**Action:** Add **1 Hidden Layer**.
**Search Space:** Try hidden dimensions in powers of 2: $\{16, 32, 64, 128\}$.

### The Rationale: Finding the "Goldilocks Zone"
We need to find the balance between a model that is "too stupid to learn" and one that is "smart enough to memorize."

### 1. The Lower Bound (e.g., 16 Neurons)
* **Scenario:** Compressing 128 input features into 16 hidden neurons (8x compression).
* **Risk (Underfitting/High Bias):** This creates a severe **Information Bottleneck**. If the data contains 50 dimensions of relevant signal, forcing it through 16 neurons causes the model to "forget" critical patterns.
* **Symptom:** Both Training Error and Test Error remain high.

### 2. The Upper Bound (e.g., 128 Neurons)
* **Scenario:** Mapping 128 inputs to 128 neurons (No compression).
* **Risk (Overfitting/High Variance):** With excessive capacity, the model may act as a **Lookup Table**, memorizing noise in the training data rather than learning general rules.
* **Symptom:** Training Error is very low, but Test Error is high.

### 3. The Sweet Spot (e.g., 32 or 64 Neurons)
* The ideal dimension captures the underlying pattern while discarding the noise.

---

## Phase 3: Tuning Depth (Hierarchy)
**Action:** Add **2 Hidden Layers**.
**Strategy:** Use the "winner" from Phase 2 to prune the search space.

### The Rationale: The "Funnel" Architecture
Instead of blindly guessing dimensions for two layers (e.g., `128 -> 128` or `16 -> 16`), we uses the insights from Phase 2 to design a **Gradual Compression** architecture.

### The Pivot Strategy
If Phase 2 showed that `64` neurons performed best (balancing under/overfitting), we use that as our anchor for the first layer ($h_1$).

* **Proposed Architecture:** Input (128) $\to$ $h_1$ (64) $\to$ $h_2$ (16 or 32) $\to$ Output.

### Why this works: Gradual Feature Extraction
This "Funnel" shape forces the network to organize information hierarchically:
1.  **Layer 1 (64):** Extracts broad, low-level features and filters out obvious noise.
2.  **Layer 2 (32):** Combines those features into higher-level concepts.

By compressing gradually ($128 \to 64 \to 32$) rather than suddenly ($128 \to 16$), we allow the network to refine the data step-by-step, preserving signal while reducing dimensionality.

---

## Summary Workflow

| Step  | Architecture                         | Goal                | Logical Check                                                    |
| :---- | :----------------------------------- | :------------------ | :--------------------------------------------------------------- |
| **1** | Input $\to$ Output                   | **Linearity Check** | Can a simple straight line solve this?                           |
| **2** | Input $\to$ H1 $\to$ Output          | **Capacity Check**  | How much "memory" does the model need? (Finding the bottleneck). |
| **3** | Input $\to$ H1 $\to$ H2 $\to$ Output | **Hierarchy Check** | Does breaking the problem into two steps improve compression?    |



---

# other discussion

* NN vs SVM for classification task:
    * svm is not very sensitive to hyperparameter. many hyperparameter give similar results.
        * but NN has flexiblity to tune architecture (cover wide range of model compexity spectrum)
    * kernel SVM is very costly to scale up for samples with 1 million or 10 million samples.
        * NN based on SGD can be easily scaled up.
    * though svm has good math theory, NN does not.
    * NN is more of a programming language, it preseves flexiblity for you to express your achitecture/inductive bias to better fit data
    * NN is good at extracting implicit features from unstructured data.


* do we do k-fold CV in deep learning
    * deep learning training is too costly, typically we do just 1 fold CV on very large dataset.

* what approach for tuning hyperparameters:
    * grid vs random search vs bayes
    * LI's view:
        * based on expert's domain knowledge.
            * sequential search based on prior results, on a relative smaller set
        * random grid search. recommend
        * bayes: require 100 or 10000 times search to get better results.


# weight decay L2 regularization

* common value for l2 norm is 1e-2, 1e-3, 1e-4. (assume relative to mean of sample loss)
    * those value in SGD has the nice interpretation of relative scaling to weight itself
    * thus there are common values suitable for many different dataset.
* for tuning NN, slight tuning around those values probably is enough.

# dropout
* dropout is a common technique used to control model complexity for MLP
* dropout commonly used in output of hidden layer in MLP.
    * it's not used in convolution layer, typically used only in fully connected layer.
* dropout probability is the hyperparameter for regularization effect
    * larger value means stronger stronger regularization.
* dropout has a issue with reproducibility (because of random selection)
    * this can be fixed via setting random seed.

* we don't enable dropout during inference:
    * dropout is only needed in training for regularization effect
    * enable dropout in inference adds randomness, you will need to ensemble multiple runs to get average results. but that's more or less equivalent to disable dropout (hinton's ensemble view)
* during training, dropout transform output value by 1/(1-p) to adjust for expected value taking into account number of output dimension disabled.



why dropout is preferred compared to l2 regularization:
    * typically feels easier to tune than l2 regularization.
    * common values are 0.1, 0.5, 0.9.
    * A common choice is to set a lower dropout probability for hidden layer closer to the input layer.
example hyperparaemeter test path:
    * hidden 64
    * -> hidden 128 + dropout 0.5
    * -> hidden 128 + dropout 0.1
    * allow larger model capacity (increase hidden dim) but use regularization to control search space.

* dropout might slows down training. intuitively due to the randomness
    * but there is no consensus on if increasing learning rate helps with speeding up convergence to combat this issue.

# numerical stabilty

* how nan or inf generated in training:
    * inf is due to overflow gradient etc (especially when training with GPU on float16). typically too large learning rate.
    * nna is due to division of zero.
    * solution:
        * proper initialization
            * xavier intialization.
            * control std of weight distribution to be small. such that probabiliy of overflow is smaller. tune this until the program can output proper loss then tune other parameters.
        * correct scale of learning rate.

* gradient explosion typically is not related to activation function, mostly due to large values in weights/backpropagation.
* gradient diminishing might be caused by sigmoid activation function or some other reasons.

* initialization distribution does not matter much as long as the variance is fixed by xavier initialization
    * we typically use normal and uniform because those are very common distribution and well behaved?
    * I think as long as you don't go with some heavy tail strange distribution, probably is fine.



---

# Summary: Gradient Explosion (Inf) to NaN Conversion in PyTorch

This document summarizes the technical transition from infinite gradient values (**Inf**) to undefined numerical values (**NaN**) during the training of neural networks in PyTorch.

---

## 1. The Core Mechanism: IEEE 754 Arithmetic
The transition from `Inf` to `NaN` is rarely a single event but a chain reaction governed by floating-point standard rules.

* **Inf (Infinity):** Occurs when a value exceeds the maximum representable float (e.g., ~3.4e38 for `float32`).
* **NaN (Not a Number):** Occurs when an operation is mathematically undefined.

### Common Conversion Triggers
| Operation             | Input                      | Result |
| :-------------------- | :------------------------- | :----- |
| **Subtraction**       | `Inf - Inf`                | `NaN`  |
| **Multiplication**    | `0 * Inf`                  | `NaN`  |
| **Division**          | `Inf / Inf`                | `NaN`  |
| **Power/Square Root** | `sqrt(-1)` or `(-Inf)^0.5` | `NaN`  |

---

## 2. Primary Transition Cases

### Case A: The Optimizer Update
The most common point of conversion. After gradients explode to `Inf`, the optimizer attempts to update the model weights ($w$):
$$w_{new} = w_{old} - (\eta \cdot \text{Inf})$$
The weight becomes `Inf`. In the subsequent forward pass, if this weight is multiplied by a zero-valued activation (common in ReLU layers), the result of $0 \cdot \text{Inf}$ becomes `NaN`.

### Case B: Normalization Breakdown
In `LayerNorm` or `BatchNorm`, the statistics (mean $\mu$ and variance $\sigma^2$) are calculated across the batch or features.
1.  If one activation is `Inf`, the variance $\sigma^2$ becomes `Inf`.
2.  The normalization formula $\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$ then attempts to divide by `Inf`.
3.  Any non-finite numerator divided by an infinite denominator leads to a total collapse into `NaN`.

### Case C: Softmax and Cross-Entropy
PyTorch's `CrossEntropyLoss` is numerically stabilized, but internal overflows can still occur:
* An `Inf` logit passed into an exponential function results in `Inf`.
* The Softmax normalization (summing exponentials) results in `Inf / Inf`, creating `NaN` probabilities.
* The loss function then calculates $\log(NaN)$, which propagates `NaN` through the entire backpropagation chain.

---

## 3. Visualization of the Failure Chain

1.  **Explosion:** High learning rate or deep architectures cause gradients to grow exponentially.
2.  **Saturation:** Gradients hit the `float32` limit and are stored as `Inf`.
3.  **Contamination:** Weights become `Inf` during the `optimizer.step()`.
4.  **Invalidation:** Mathematical operations in the next forward pass (like `0 * Inf`) turn the weights or activations into `NaN`.
5.  **Persistence:** Once a weight is `NaN`, any further calculation involving that weight will result in `NaN` indefinitely.

---

## 4. Prevention and Debugging

* **Gradient Clipping:** Use `torch.nn.utils.clip_grad_norm_` to cap the norm of gradients before they reach the `Inf` threshold.
* **Anomaly Detection:** Use `torch.autograd.set_detect_anomaly(True)` to identify the exact forward-pass operation that first generated a non-finite value.
* **Weight Initialization:** Ensure weights are initialized with small variances (e.g., Xavier or Kaiming initialization) to prevent early-epoch explosions.