# NeuroEngine-CPP — Neural Networks in C++

This repository is part of my project to implement **Deep Learning from scratch in C++**, using **Armadillo** as the matrix computation backend.

---

## Phase 1: Tensor Abstraction
In this phase, I implemented a basic `Tensor` class that supports:

- storing data in a **1-D contiguous `vector`**
- keeping track of **shape and strides**
- safe **multi-dimensional indexing**
- math operations like:
  - `+  -  /  %`
  - `dot()` for matrix multiplication
  - `transpose()`
  - `reshape()`
  - `sum(), mean(), max()` along rows/columns
- a `view()` method to convert storage into an Armadillo matrix **without copying data**
- simple `print()` for debugging

This is the base I’ll build everything else on.

---

## Why Tensor Abstraction First?
Because every neural network eventually becomes:
> numbers stored in memory + math operations + gradients flowing backward

If the tensor is not built well, nothing built on top of it will be clean or efficient.

---

## Small Example
```cpp
fmat A = {{1, 3}, {2, 4}};
Tensor<float> a(A);
a.print();

Tensor<float> t = a.transpose();
t.print();

Tensor<float> c = a + t;
c.print();

Tensor<float> d = a.dot(t);
d.print();

---

## **DyNet Comparison & Roadmap**

This repository has been analyzed and compared with [DyNet](https://github.com/clab/dynet), a mature neural network toolkit from Carnegie Mellon University. Comprehensive documentation has been created:

📋 **[DYNET_COMPARISON.md](DYNET_COMPARISON.md)** - Detailed architectural comparison
- Lists all architectural gaps between this repo and DyNet
- Compares system design and forward/backward propagation approaches
- Identifies critical missing features (computational graphs, autograd, parameters, optimizers)
- Provides prioritized recommendations

🗺️ **[ARCHITECTURE_ROADMAP.md](ARCHITECTURE_ROADMAP.md)** - Implementation guide
- Detailed Phase 2 implementation plan (Computational Graph & Autograd)
- Complete code examples with forward and backward propagation
- File organization and testing strategy
- Week-by-week implementation checklist

### Current Status
**Phase 1 Complete**: Basic Tensor abstraction ✅  
**Next**: Phase 2 - Computational Graph & Automatic Differentiation

---

## **Next After This**

### Critical Foundation (Phase 2)
- **Computational Graph** - Record operations in a DAG structure
- **Automatic Differentiation** - Backpropagation through the graph
- **Expression API** - User-friendly operator overloading

### Training Infrastructure (Phase 3)
- **Parameter Management** - ParameterCollection for model weights
- **Optimizers** - SGD, Adam, RMSprop implementations

### Neural Network Layers (Phase 4)
- **Layer Abstractions** - Linear, activation functions, loss functions
- **Model Building** - High-level API for network construction

### Advanced Features (Phase 5+)
- **Recurrent Networks** - LSTM, GRU implementations
- **CUDA Support** - GPU acceleration
- **Auto-batching** - DyNet's signature optimization feature

---
