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

## 📚 **Architectural Analysis & Implementation Guides**

This repository includes comprehensive architectural analysis comparing NeuroEngine-CPP with industry-leading frameworks:

### 🚀 **Start Here**

**New!** [PYTORCH_QUICK_START.md](PYTORCH_QUICK_START.md) - Quick overview of gaps with PyTorch

### 📖 **Complete Documentation**

#### **PyTorch Comparison** (Industry Standard)

1. **[PYTORCH_COMPARISON.md](PYTORCH_COMPARISON.md)** - Comprehensive PyTorch architectural comparison
   - Complete gap analysis across 14 major components
   - Tensor system, autograd, neural modules, optimizers
   - GPU support, data loading, production features
   - Side-by-side code examples (PyTorch vs NeuroEngine-CPP)
   - Detailed implementation roadmap with timelines
   - **800+ lines of in-depth analysis**

2. **[PYTORCH_QUICK_START.md](PYTORCH_QUICK_START.md)** - Quick reference guide
   - Top 5 critical gaps at a glance
   - Current capabilities vs missing features
   - Phase-by-phase roadmap summary
   - Key concepts explained simply

#### **DyNet Comparison** (Research Framework)

3. **[DYNET_COMPARISON.md](DYNET_COMPARISON.md)** - DyNet framework comparison
   - Dynamic computation graph architecture
   - Research-oriented perspective
   - Complements PyTorch analysis

4. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - DyNet quick reference

#### **Implementation Guides**

5. **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - Step-by-step implementation
   - Phase 2 (Autograd) detailed instructions
   - Complete code examples for all components
   - Build and testing strategies

6. **[ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)** - Visual architecture guide
   - ASCII diagrams of system architecture
   - Data flow illustrations

### 🎯 **Key Findings**

**Current State**: Phase 1 (Tensor Abstraction) ✅ Complete (~1% of PyTorch)

**Critical Gaps** (Cannot function without):
- ❌ No Autograd (automatic differentiation)
- ❌ No Neural Network Modules
- ❌ No Optimizers (SGD, Adam)
- ❌ No Loss Functions
- ❌ No GPU Support

**Critical Next Steps**:
- Phase 2: Computation Graph & Autograd (START HERE) - 4-6 weeks
- Phase 3: Neural Network Modules - 6-8 weeks
- Phase 4: Optimizers & Training - 4-6 weeks

**Timeline to MVP**: 4-5 months | **Production-Ready**: 12-15 months

See [PYTORCH_COMPARISON.md](PYTORCH_COMPARISON.md) for complete architectural analysis.

---
