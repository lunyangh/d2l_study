# Generalized Linear Models: A Unified Framework for Poisson, Logistic, and Softmax Regression

This notes discuesses about generalized linear model. it unifies softmax + cross entropy loss, logistic regression + cross entropy loss and mean squared loss for regression problem.


## 1. Introduction to GLMs
The Generalized Linear Model (GLM) is a flexible generalization of ordinary linear regression that allows for response variables that have error distribution models other than a normal distribution.

Every GLM consists of three core components:
1.  **Random Component:** The probability distribution of the response variable $Y$ (e.g., Normal, Poisson, Binomial).
2.  **Systematic Component:** A linear predictor $\eta$ formed by a linear combination of predictors $X$ and coefficients $\beta$.
    $$\eta = X\beta$$
3.  **Link Function:** A function $g(\cdot)$ that connects the expected value (mean) of the distribution ($\mu$) to the linear predictor.
    $$\eta = g(\mu) \iff \mu = g^{-1}(\eta)$$

---

## 2. The Mathematical Engine: The Exponential Family

For a model to be a GLM, the distribution must belong to the **Exponential Family**. The probability density function (PDF) or probability mass function (PMF) takes the following form:

$$f(y|\theta) = h(y) \exp\left( \langle y, \theta \rangle - b(\theta) \right)$$

* $\theta$: The **Canonical Parameter** (or Natural Parameter).
* $b(\theta)$: The **Cumulant Function** (Log-Partition Function).
* $\langle y, \theta \rangle$: The interaction between the data and the parameter.

### Key Properties
1.  **The Mean:** The first derivative of the cumulant is the mean. (conditional expectation of y given theta)
    $$\mu = b'(\theta)$$
2.  **The Variance:** The second derivative is the variance.
    $$\text{Var}(Y) = b''(\theta)$$

---

## 3. The "Canonical Link" and Optimization

While many link functions are possible, every distribution has a unique **Canonical Link Function**. This occurs when we set the linear predictor $\eta$ directly equal to the canonical parameter $\theta$.

$$\eta = \theta$$

### The "Universal" Gradient
When using the canonical link, the derivative of the Log-Likelihood (the Score Equation) simplifies to a "Linear minus Non-Linear" structure. This results in the same gradient form for **all** GLMs:

$$\nabla_\beta \mathcal{L} = X^T (y - \mu)$$

Or, for a single observation $i$ and coefficient $j$:
$$\frac{\partial \mathcal{L}}{\partial \beta_j} = (y_i - \hat{y}_i) x_{ij}$$

### The Orthogonality Condition
Setting the gradient to zero yields the **Normal Equations** for GLMs. This implies that the Maximum Likelihood Estimate (MLE) is found where the raw residuals are orthogonal to the predictors:

$$\sum_{i=1}^n (y_i - \mu_i)x_{ij} = 0$$

This is why GLM estimation (via Iteratively Reweighted Least Squares) is mathematically equivalent to solving a weighted OLS problem repeatedly.

---

## 4. Case Studies

### A. Poisson Regression (Count Data)
Used for modeling rates or counts (e.g., traffic accidents, call center volume).
* **Distribution:** Poisson
* **Canonical Link:** Log
    $$\eta = \ln(\mu) \implies \mu = e^\eta$$
* **Interpretation:** Coefficients represent multiplicative effects on the rate.
* **Gradient:** $(y - e^{X\beta})x$

### B. Logistic Regression (Binary Classification)
Used for binary outcomes (0/1).
* **Distribution:** Bernoulli (Binomial with $n=1$)
* **Canonical Link:** Logit
    $$\eta = \ln\left(\frac{\mu}{1-\mu}\right) \implies \mu = \frac{1}{1+e^{-\eta}} = \sigma(\eta)$$
* **Interpretation:** Coefficients represent changes in the log-odds.
* **Gradient:** $(y - \sigma(X\beta))x$

### C. Softmax Regression (Multi-Class Classification)
Used for categorical outcomes with $K$ classes. This is the multivariate extension of the GLM.
* **Distribution:** Multinomial (Categorical)
* **Canonical Link:** Softmax (Inverse Link)
    $$\mu_k = \frac{e^{\eta_k}}{\sum_{j} e^{\eta_j}}$$
* **Loss Function:** Categorical Cross Entropy (equivalent to Negative Log Likelihood).
* **Gradient:** $(y - \text{softmax}(X\beta))x$

---

## 5. Summary Table

| Model        | Distribution      | Canonical Link Function $g(\mu)$       | Mean Function $\mu = g^{-1}(\eta)$ | Range of $Y$         |
| :----------- | :---------------- | :------------------------------------- | :--------------------------------- | :------------------- |
| **OLS**      | Normal (Gaussian) | Identity: $\mu = \eta$                 | $\eta$                             | $(-\infty, \infty)$  |
| **Poisson**  | Poisson           | Log: $\ln(\mu) = \eta$                 | $e^\eta$                           | $\{0, 1, 2, \dots\}$ |
| **Logistic** | Bernoulli         | Logit: $\ln(\frac{\mu}{1-\mu}) = \eta$ | $\frac{1}{1+e^{-\eta}}$            | $\{0, 1\}$           |
| **Softmax**  | Multinomial       | (Multivariate Logit)                   | $\frac{e^{\eta_k}}{\sum e^{\eta}}$ | One-hot Vector       |