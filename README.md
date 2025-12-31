# NeuroEngine-CPP — Neural Networks in C++ (Phase 1)

This repository is part of my journey to implement **Deep Learning from scratch in C++**, using **Armadillo** as the matrix computation backend.

### 🎯 Goal of Phase 1
Build a simple **Tensor abstraction layer** that allows me to:
- Store data in a **flat contiguous 1-D array**
- Track tensor **shape and compute strides**
- Support safe **multi-dimensional indexing**
- Implement common math operations like:
  - `+` `-` `/` `%`
  - `dot()` (matrix multiplication)
  - `transpose()`
  - `reshape()`
  - `sum()` `mean()` `max()`
  - basic **row broadcasting**

This phase is the foundation for future steps like **backpropagation, neural network layers, and GPU acceleration**.

---

## 🧠 Why Tensor Abstraction First?
Because every neural network eventually becomes:
> numbers stored in memory + math operations + gradients flowing backward

If the tensor is not built well, nothing built on top of it will be clean or efficient.

---

## 📌 Example

```cpp
fmat A = {{1, 3}, {2, 4}};
Tensor<float> a(A);
Tensor<float> t = a.transpose();
a.print();
t.print();
