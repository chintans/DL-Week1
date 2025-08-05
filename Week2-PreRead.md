**Quick orientation (why these five pillars matter)**
Activation functions inject non-linearity, multilayer perceptrons (MLPs) stack those activations to model complex patterns, loss functions tell us *how wrong* the network is, forward/back-propagation compute errors and send them backward, and optimizers like SGD, Momentum, and Adam translate those gradients into weight updates. Mastering how these pieces fit together is enough to implement and troubleshoot most modern deep-learning pipelines.

---

## 1 Activation Functions

### 1.1 Why we need them

Without a non-linear activation after each affine transformation (**w·x + b**), any depth-N network collapses to one big linear map; it would be no more expressive than a single perceptron — and it would still suffer from vanishing or exploding gradients in deep stacks. ([Wikipedia][1], [Wikipedia][2])

### 1.2 ReLU (Rectified Linear Unit)

| Formula | $\text{ReLU}(z)=\max(0,z)$ |
| ------- | -------------------------- |

* **Cheap to compute** (no exponentials). ([Built In][3])
* **Sparse outputs** — negative activations are clamped to 0, so only a subset of neurons fire at any given time, reducing inter-neuron interference. ([Cross Validated][4])
* **Helps mitigate vanishing gradients** because its derivative is 1 for positive inputs. ([KDnuggets][5])
* **Caveat:** neurons can “die” if their inputs stay negative; Leaky-ReLU, ELU, and GELU are common fixes.

### 1.3 Sigmoid (Logistic)

| Formula | $\sigma(z)=\dfrac{1}{1+e^{-z}}$ |
| ------- | ------------------------------- |

* Smooth, bounded in (0, 1); ideal when you want probabilistic interpretation, e.g., output layer for binary classification. ([Wikipedia][6])
* Gradients shrink quickly for $|z|\gt4$, which can slow training in deep or poorly initialized networks (classic *vanishing-gradient* issue). ([KDnuggets][5])

### 1.4 Tanh (Hyperbolic Tangent)

| Formula | $\tanh(z)=\dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$ |
| ------- | --------------------------------------------- |

* Output range (−1, 1) centers activations around zero, often leading to faster convergence than sigmoid. ([GeeksforGeeks][7])
* Still suffers from gradient saturation at extreme inputs, though less severely than sigmoid because its derivative peaks at 1 rather than 0.25. ([KDnuggets][5])

### 1.5 When to choose what

* **Hidden layers:** ReLU by default; switch to Leaky-ReLU/ELU if many neurons die or if negative information is important.
* **Binary-prob outputs:** Sigmoid.
* **Zero-centered outputs (e.g., RNN gates):** Tanh.
* **Very deep nets / transformer sub-layers:** GELU or SiLU dominate in practice; they offer smoothness without hard saturation.

---

## 2 Multilayer Perceptron (MLP)

A vanilla MLP is a *feed-forward* network that arranges neurons in layers **Input → Hidden(1…k) → Output**; every neuron in layer *L* connects to every neuron in *L + 1* (fully connected). ([Wikipedia][2])

### 2.1 Why multiple layers?

Even a single hidden layer with a non-polynomial activation is a **universal approximator** — it can approximate any continuous function given enough neurons. ([Wikipedia][2]) But depth often beats width: stacking layers can represent certain functions exponentially more efficiently than a single wide layer.

### 2.2 Solving non-linear separation

Consider the classic XOR problem: no straight line separates outputs in input space, but an MLP with just one hidden layer (two ReLU units) can carve out the correct decision regions.

---

## 3 Loss Functions

| Task type                 | Typical loss                        | Formula                                  | Intuition                                                                                                                                            |
| ------------------------- | ----------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Binary classification** | **Binary Cross-Entropy (Log-Loss)** | $L=-[y\log p + (1-y)\log(1-p)]$          | Penalizes confident but wrong predictions; diverges as $p\rightarrow0$ or $1$ opposite the label. ([GeeksforGeeks][8], [peterroelants.github.io][9]) |
| **Regression**            | **Mean Squared Error (MSE)**        | $L=\dfrac{1}{N}\sum_{i}(y_i-\hat y_i)^2$ | Measures average squared distance; symmetric, heavily punishes large errors. ([Built In][10], [Wikipedia][11])                                       |

> **Why not use accuracy as a loss?** Accuracy is non-differentiable; gradient-based optimizers need smooth objectives.

---

## 4 Forward & Backpropagation

### 4.1 Forward pass

1. Feed input vector $x$ through each layer: $z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}$, $a^{(l)}=f(z^{(l)})$.
2. Produce prediction $\hat y$. ([DataCamp][12])

### 4.2 Backpropagation (gradient back-flow)

1. Compute loss $L(\hat y, y)$.
2. Starting at the output layer, compute gradients of $L$ w\.r.t. activations, then w\.r.t. weights and biases via chain rule.
3. Propagate error back layer-by-layer, accumulating partial derivatives.
4. Update parameters using an optimizer (next section). ([NVIDIA Developer][13])

> **Key point:** Backprop is just systematic application of calculus; automatic-differentiation libraries (PyTorch, TensorFlow, JAX) generate these gradient formulas for you.

---

## 5 Optimization Algorithms

### 5.1 Stochastic Gradient Descent (SGD)

* Updates after each mini-batch (or even single example), giving noisy but cheap gradient estimates. ([GeeksforGeeks][14], [Wikipedia][15])
* Learning rate $\eta$ controls step size; too high → divergence, too low → crawl.

### 5.2 Momentum & Nesterov Momentum

* Maintains an exponentially decaying *velocity* $v$ that accumulates past gradients:
  $v_t = \gamma v_{t-1} + \eta \nabla L$.
  Weights move by $-v_t$, smoothing out zig-zags in ravines. ([Medium][16], [DeepAI][17])
* **Nesterov variant** looks ahead one step before measuring the gradient, often yielding faster convergence on convex surfaces.

### 5.3 Adam (Adaptive Moment Estimation)

* Combines **RMSProp-style** per-parameter learning rates (second-moment estimate) with **momentum** (first-moment estimate).
* Default hyper-parameters ($\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$) work well for most problems. ([MarketMuse Blog][18], [GeeksforGeeks][19])
* Widely used in NLP and vision because it handles sparse gradients and decays learning rate automatically.

---

### Cheat-sheet of typical choices

| Component         | Common default                                                                |
| ----------------- | ----------------------------------------------------------------------------- |
| Hidden activation | ReLU / GELU                                                                   |
| Output activation | Sigmoid (binary), Softmax (multi-class), Identity (regression)                |
| Loss              | BCE for binary, Categorical Cross-Entropy for multi-class, MSE for regression |
| Optimizer         | Adam at $1\text{e-}3$ → decay or switch to SGD with momentum for fine-tuning  |

---

## Take-away messages

1. **Activation functions** decide the expressiveness and trainability of each neuron.
2. **MLPs** leverage stacked activations to approximate any reasonable function.
3. A **loss function** must be differentiable and matched to the prediction target.
4. **Backpropagation** + **gradients** turn losses into informative weight updates.
5. **Optimizers** translate gradients into motion; Adam is the go-to, but plain SGD with momentum remains a strong baseline.

With these foundations internalized, you can read modern deep-learning code bases, spot implementation bugs, and explain *why* a network behaves the way it does.

[1]: https://en.wikipedia.org/wiki/Vanishing_gradient_problem?utm_source=chatgpt.com "Vanishing gradient problem - Wikipedia"
[2]: https://en.wikipedia.org/wiki/Universal_approximation_theorem?utm_source=chatgpt.com "Universal approximation theorem - Wikipedia"
[3]: https://builtin.com/machine-learning/relu-activation-function?utm_source=chatgpt.com "ReLU Activation Function Explained - Built In"
[4]: https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks?utm_source=chatgpt.com "What are the advantages of ReLU over sigmoid function in deep ..."
[5]: https://www.kdnuggets.com/2022/02/vanishing-gradient-problem.html?utm_source=chatgpt.com "Vanishing Gradient Problem: Causes, Consequences, and Solutions"
[6]: https://en.wikipedia.org/wiki/Sigmoid_function?utm_source=chatgpt.com "Sigmoid function - Wikipedia"
[7]: https://www.geeksforgeeks.org/deep-learning/tanh-activation-in-neural-network/?utm_source=chatgpt.com "Tanh Activation in Neural Network - GeeksforGeeks"
[8]: https://www.geeksforgeeks.org/deep-learning/binary-cross-entropy-log-loss-for-binary-classification/?utm_source=chatgpt.com "Binary Cross Entropy/Log Loss for Binary Classification"
[9]: https://peterroelants.github.io/posts/cross-entropy-logistic/?utm_source=chatgpt.com "Logistic classification with cross-entropy (1/2) - Peter Roelants"
[10]: https://builtin.com/machine-learning/loss-functions?utm_source=chatgpt.com "Loss Functions in Neural Networks & Deep Learning | Built In"
[11]: https://en.wikipedia.org/wiki/Mean_squared_error?utm_source=chatgpt.com "Mean squared error - Wikipedia"
[12]: https://www.datacamp.com/tutorial/forward-propagation-neural-networks?utm_source=chatgpt.com "Forward Propagation in Neural Networks: A Complete Guide"
[13]: https://developer.nvidia.com/blog/a-data-scientists-guide-to-gradient-descent-and-backpropagation-algorithms/?utm_source=chatgpt.com "A Data Scientist's Guide to Gradient Descent and Backpropagation ..."
[14]: https://www.geeksforgeeks.org/machine-learning/ml-stochastic-gradient-descent-sgd/?utm_source=chatgpt.com "ML - Stochastic Gradient Descent (SGD) - GeeksforGeeks"
[15]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent?utm_source=chatgpt.com "Stochastic gradient descent - Wikipedia"
[16]: https://medium.com/%40giorgio.martinez1926/nesterov-momentum-explained-with-examples-in-tensorflow-and-pytorch-4673dbf21998?utm_source=chatgpt.com "Nesterov Momentum Explained with examples in TensorFlow and ..."
[17]: https://deepai.org/machine-learning-glossary-and-terms/nesterovs-momentum?utm_source=chatgpt.com "Nesterov's Momentum Definition | DeepAI"
[18]: https://blog.marketmuse.com/glossary/adaptive-moment-estimation-adam-definition/?utm_source=chatgpt.com "What is Adaptive Moment Estimation (ADAM) - MarketMuse Blog"
[19]: https://www.geeksforgeeks.org/machine-learning/adam-adaptive-moment-estimation-optimization-ml/?utm_source=chatgpt.com "ML | ADAM (Adaptive Moment Estimation) Optimization"
