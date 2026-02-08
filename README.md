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

## **Next After This**

- **Backpropagation / Autograd**
- **Neural Network Layers**
- **Optimizers**
- **CUDA (maybe later!)**

---

## 📚 **DyNet Comparison & Implementation Guides**

This repository now includes comprehensive architectural analysis and implementation guides inspired by DyNet:

### 📖 **Documentation**

1. **[DYNET_COMPARISON.md](DYNET_COMPARISON.md)** - Complete architectural comparison
   - Side-by-side analysis of NeuroEngine-CPP vs DyNet
   - Detailed gap analysis for each component
   - System architecture and forward pass approach differences
   - Development roadmap with 6 phases
   - Code examples and best practices
   - Estimated timelines and effort

2. **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - Step-by-step implementation guide
   - Detailed instructions for Phase 2 (Computation Graph & Autograd)
   - Complete code examples for each component
   - Node base class and operation implementations
   - Gradient checking and testing strategies
   - Integration with existing Tensor code
   - Build and compilation instructions

3. **[ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)** - Visual architecture guide
   - ASCII diagrams of current vs target architecture
   - Execution flow comparisons (eager vs graph-based)
   - Data flow in forward/backward passes
   - Memory management strategies
   - Component interaction diagrams
   - Complete roadmap visualization

### 🎯 **Key Findings**

**Current State**: Phase 1 (Tensor Abstraction) ✅ Complete

**Critical Next Steps**:
- Phase 2: Computation Graph & Autograd (START HERE)
- Phase 3: Parameter Management & Training
- Phase 4: Neural Network Layers

See [DYNET_COMPARISON.md](DYNET_COMPARISON.md) for the complete analysis and roadmap.

---
