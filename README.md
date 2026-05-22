# Week 1 — MNIST Classification

**Mentee:** Aryan Prasad
**SoC Track:** Diffusion Models from Scratch — SoC 2026

## Final results
- **Test accuracy:** 97.53%
- **Best validation accuracy:** 97.42% at epoch 10
- **Final train loss:** 0.0262

## Design choices
- **Architecture:** MLP with 2 layers, Linear(784→128) → ReLU → Linear(128→10), no dropout
- **Optimizer:** Adam with lr=0.001
- **Batch size:** 64
- **Epochs trained:** 10
- **Validation split:** 10% of training data, no fixed seed

## What I learned
i learned how Pytorch traning loop works, zero.grad(), forward pass, loss , back prop, weights and bias update. learned another loss funtion crossentropy and how it handle softmax on its own.

## What I'd do differently
i would add a fixed seed and will try to increase the hidden layer to see if more accuracy is possible or not

## How to reproduce
1. Open `week1_mnist.ipynb` in Colab with a T4 GPU runtime.
2. Run all cells top to bottom.
3. Checkpoint will be saved to `best_model.pt`.
```

---