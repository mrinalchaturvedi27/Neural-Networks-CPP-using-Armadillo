# NeuroEngine-CPP — Neural Networks in C++

This repository is part of my project to implement **Deep Learning from scratch in C++**, using **Armadillo** as the matrix computation backend.

---

## Phase 1: Tensor Abstraction
In this phase, I implemented a basic `Tensor` class that supports:

- storing data in a **1-D contiguous `std::vector`**
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

## **Next After This**

- **Backpropagation / Autograd**
- **Neural Network Layers**
- **Optimizers**
- **CUDA (maybe later!)**
