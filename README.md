# MATH 3700 Project: From Chain Rule to Gradient Vanishing

**Author:** Shimo Wang (ID: 1306043)  
**Course:** MATH 3700 - Big Data Computing  

##  Project Overview
Deep networks are trained via backpropagation, which is fundamentally the repeated application of the multivariate **Chain Rule**. As network depth increases, the gradients are multiplied layer by layer, creating a product of Jacobians. This calculus-driven process determines whether a network can learn or fail due to vanishing/exploding gradients. This project connects theoretical calculus with computational experiments on the `make_moons` dataset.

##  Mathematical Mainline

The backpropagated gradient for an \(L\)-layer network is a product of per-layer Jacobians. The gradient signal from the last layer (\(h_L\)) to the first layer weights (\(W_1\)) is given by:

$$
\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial h_L}\prod_{k=2}^{L}\frac{\partial h_k}{\partial h_{k-1}}\frac{\partial h_1}{\partial W_1}
$$

By the sub-multiplicative property of spectral norms, the magnitude of this product is bounded:

$$
\left\| \prod_{k=1}^{L} J_k \right\|_2 \le \prod_{k=1}^{L} \|D_k\|_2 \|W_k\|_2
$$

**Vanishing/Exploding Conditions:**
- If \(\|J_k\|_2 < 1\) for all \(k\), the product decays geometrically, leading to **Vanishing Gradients**. (e.g., Sigmoid derivative \(\sigma'(z) \in (0, 0.25]\)).
- If \(\|J_k\|_2 > 1\) for all \(k\), the product grows exponentially, leading to **Exploding Gradients**.

##  Experimental Setup
- **Dataset:** Synthetic `make_moons` dataset (500 samples, 2 features).
- **Network:** Fully connected feed-forward network with **10 hidden layers** (16 neurons each).
- **Task:** Binary classification using MSE Loss and vanilla Gradient Descent.

##  Key Results

### 1. Vanishing Gradients (Sigmoid vs. ReLU)
Sigmoid's derivative is bounded by 0.25, causing exponential decay over 10 layers (Dead Neurons). ReLU's derivative is \(\{0, 1\}\), which preserves the gradient signal without suppressing it.

![Cumulative Derivative Product](figures/cumulative_derivative_product.png)
![Gradient Norm During Training](figures/gradient_norm_training.png)

### 2. Depth Effect & Exploding Gradients
As network depth increases, Sigmoid fails to learn (loss plateaus at 0.25). Furthermore, scaling weights too large (e.g., Scale=1.5) causes the cumulative Jacobian norm to explode to \(10^{13}\) over 20 layers, proving the necessity of strict weight initialization.

![Weight Scale vs Jacobian Norm](figures/weight_scale_jacobian.png)

##  Discussion: The Optimizer & Initialization Perspective

The chain rule's multiplicative structure is the root cause of numerical instability in deep networks. Our experiments align with recent literature addressing these issues:
- **Activation Gradients & Optimizers:** As discussed by *Liu et al. (2023)*, simply changing the activation function (like ReLU) regulates the gradient magnitudes. Furthermore, modern optimizers (Adam, Momentum) adaptively scale learning rates to prevent weights from growing out of bounds during backpropagation.
- **Self-Normalization & Orthogonal Filters:** To fundamentally solve the exploding gradient problem without relying purely on optimizers, *Lu et al. (2023)* propose bidirectionally self-normalizing networks, and *Ceni (2025)* introduces Random Orthogonal Additive Filters. These methods mathematically control the spectral norm \(\|W_k\|_2\) to keep the Jacobian product near 1, ensuring stable signal propagation.

##  Conclusion & Future Work

**Conclusion:**
Training deep networks is fundamentally a **numerical stability problem**. While ReLU mitigates the vanishing gradient caused by Sigmoid's \(0.25\) derivative bound, the risk of explosion remains if weights are not properly scaled. The mathematical structure of the chain rule demands careful architectural design.

**Future Work:**
Building on our findings and recent literature, future work should investigate how modern architectures—such as **ResNet skip connections**—provide an additive shortcut that bypasses the multiplicative Jacobian chain entirely, allowing for the successful training of 100+ layer networks.

##  References

1. Lu, Y., Gould, S., & Ajanthan, T. (2023). Bidirectionally self-normalizing neural networks. *Neural Networks*, 167, 283-291.
2. Liu, M., Chen, L., Du, X., Jin, L., & Shang, M. (2023). Activated Gradients for Deep Neural Networks. *IEEE TNNLS*, 34(4), 2156-2168.
3. Ceni, A. (2025). Random Orthogonal Additive Filters: A Solution to the Vanishing/Exploding Gradient of Deep Neural Networks. *IEEE TNNLS*, 36(6), 10794-10807.
