# NeuroEngine-CPP Architecture Roadmap
## Inspired by DyNet's Design

---

## 1. System Architecture Overview

### 1.1 Target Architecture (DyNet-Inspired)

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Application                         │
│                    (Neural Network Code)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Expression API Layer                        │
│  • Operator overloading (+, -, *, etc.)                         │
│  • High-level operations (tanh, softmax, etc.)                  │
│  • Automatic graph building                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Computation Graph Layer                       │
│  • Node management and DAG structure                            │
│  • Forward/backward execution                                   │
│  • Topological sorting                                          │
│  • Memory management                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Operation Node Layer                        │
│  • Arithmetic Nodes (Add, Sub, Mul, Div)                       │
│  • Matrix Nodes (MatMul, Transpose)                            │
│  • Activation Nodes (ReLU, Tanh, Sigmoid)                      │
│  • Each node: forward() + backward()                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Parameter Management                         │
│  • ParameterCollection (model state)                            │
│  • Parameter (weights, biases)                                  │
│  • Initialization strategies                                    │
│  • Model save/load                                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Training Framework                          │
│  • Trainer base class                                           │
│  • Optimizers (SGD, Adam, RMSprop)                             │
│  • Learning rate scheduling                                     │
│  • Gradient clipping                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Tensor Backend                             │
│  • Tensor<T> (current implementation)                           │
│  • Gradient storage                                             │
│  • Armadillo integration                                        │
│  • Device abstraction (CPU/GPU)                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 2 Implementation Plan: Computational Graph & Autograd

### 2.1 Core Classes to Implement

#### A. ComputationGraph Class

**File:** `graph.h` / `graph.cpp`

**Responsibilities:**
- Store all nodes in the computation
- Manage forward and backward execution
- Perform topological sorting
- Allocate and manage workspace memory

**Interface:**
```cpp
class ComputationGraph {
private:
    std::vector<Node*> nodes;           // All nodes in the graph
    std::vector<Tensor<float>> tensors; // Forward results
    std::vector<Tensor<float>> gradients; // Backward gradients
    unsigned graph_id;                   // Unique graph identifier
    
public:
    ComputationGraph();
    ~ComputationGraph();
    
    // Add a new node to the graph
    VariableIndex add_node(Node* node);
    
    // Execute forward pass
    const Tensor<float>& forward(VariableIndex i);
    
    // Execute backward pass
    void backward(VariableIndex i);
    
    // Reset graph for new computation
    void clear();
    
    // Get results
    const Tensor<float>& get_value(VariableIndex i) const;
    const Tensor<float>& get_gradient(VariableIndex i) const;
    
    unsigned get_id() const { return graph_id; }
};
```

**Key Implementation Details:**
1. **Node Storage**: Use `std::vector<Node*>` for dynamic graph building
2. **Topological Sort**: Implicit (nodes added in forward order)
3. **Memory**: Pre-allocate tensor storage for efficiency
4. **Gradient Init**: Zero gradients except for loss (set to 1.0)

---

#### B. Expression Class

**File:** `expression.h` / `expression.cpp`

**Responsibilities:**
- Wrap a node index in the computation graph
- Provide user-friendly API with operator overloading
- Check for stale references

**Interface:**
```cpp
struct Expression {
    ComputationGraph* pg;  // Pointer to the graph
    VariableIndex i;        // Index of node in graph
    unsigned graph_id;      // Graph ID for staleness check
    
    Expression();
    Expression(ComputationGraph* pg, VariableIndex i);
    
    // Access results
    const Tensor<float>& value() const;
    const Tensor<float>& gradient() const;
    
    // Staleness check
    bool is_stale() const;
};

// Operator overloading
Expression operator+(const Expression& x, const Expression& y);
Expression operator-(const Expression& x, const Expression& y);
Expression operator*(const Expression& x, const Expression& y);
Expression operator/(const Expression& x, const Expression& y);

// Functions
Expression tanh(const Expression& x);
Expression dot(const Expression& x, const Expression& y);
```

**Implementation Strategy:**
```cpp
// Example: operator+ implementation
Expression operator+(const Expression& x, const Expression& y) {
    if (x.pg != y.pg) 
        throw std::runtime_error("Expressions from different graphs");
    
    // Create add node
    Node* node = new AddNode(x.i, y.i);
    
    // Add to graph
    VariableIndex idx = x.pg->add_node(node);
    
    // Return new expression
    return Expression(x.pg, idx);
}
```

---

#### C. Node Base Class and Operations

**File:** `node.h` / `node.cpp`

**Base Class:**
```cpp
struct Node {
    std::vector<VariableIndex> args;  // Input dependencies
    
    virtual ~Node() {}
    
    // Compute forward pass
    virtual void forward(const std::vector<const Tensor<float>*>& inputs,
                        Tensor<float>& output) = 0;
    
    // Compute backward pass
    // dEdf: gradient of loss w.r.t. this node's output
    // inputs: forward pass inputs (for computation)
    // Returns: gradients w.r.t. each input
    virtual void backward(const std::vector<const Tensor<float>*>& inputs,
                         const Tensor<float>& output,
                         const Tensor<float>& dEdf,
                         std::vector<Tensor<float>>& gradients) = 0;
    
    // String representation (for debugging)
    virtual std::string as_string(const std::vector<std::string>& arg_names) const = 0;
};
```

**Example Operation Nodes:**

```cpp
// Addition: z = x + y
struct AddNode : public Node {
    AddNode(VariableIndex x, VariableIndex y) {
        args = {x, y};
    }
    
    void forward(const std::vector<const Tensor<float>*>& inputs,
                Tensor<float>& output) override {
        // output = inputs[0] + inputs[1]
        output = *inputs[0] + *inputs[1];
    }
    
    void backward(const std::vector<const Tensor<float>*>& inputs,
                 const Tensor<float>& output,
                 const Tensor<float>& dEdf,
                 std::vector<Tensor<float>>& gradients) override {
        // d/dx(x + y) = 1, so gradient passes through
        gradients[0] = dEdf;  // gradient w.r.t. x
        gradients[1] = dEdf;  // gradient w.r.t. y
    }
    
    std::string as_string(const std::vector<std::string>& args) const override {
        return args[0] + " + " + args[1];
    }
};

// Element-wise multiplication: z = x * y
struct CwiseMultNode : public Node {
    CwiseMultNode(VariableIndex x, VariableIndex y) {
        args = {x, y};
    }
    
    void forward(const std::vector<const Tensor<float>*>& inputs,
                Tensor<float>& output) override {
        output = *inputs[0] % *inputs[1];  // element-wise
    }
    
    void backward(const std::vector<const Tensor<float>*>& inputs,
                 const Tensor<float>& output,
                 const Tensor<float>& dEdf,
                 std::vector<Tensor<float>>& gradients) override {
        // d/dx(x * y) = y
        gradients[0] = dEdf % *inputs[1];
        // d/dy(x * y) = x  
        gradients[1] = dEdf % *inputs[0];
    }
    
    std::string as_string(const std::vector<std::string>& args) const override {
        return args[0] + " .* " + args[1];
    }
};

// Matrix multiplication: z = x * y
struct MatMulNode : public Node {
    MatMulNode(VariableIndex x, VariableIndex y) {
        args = {x, y};
    }
    
    void forward(const std::vector<const Tensor<float>*>& inputs,
                Tensor<float>& output) override {
        output = inputs[0]->dot(*inputs[1]);
    }
    
    void backward(const std::vector<const Tensor<float>*>& inputs,
                 const Tensor<float>& output,
                 const Tensor<float>& dEdf,
                 std::vector<Tensor<float>>& gradients) override {
        // d/dX(X*Y) = dEdf * Y^T
        gradients[0] = dEdf.dot(inputs[1]->transpose());
        // d/dY(X*Y) = X^T * dEdf
        gradients[1] = inputs[0]->transpose().dot(dEdf);
    }
    
    std::string as_string(const std::vector<std::string>& args) const override {
        return args[0] + " * " + args[1];
    }
};

// Tanh activation: y = tanh(x)
struct TanhNode : public Node {
    TanhNode(VariableIndex x) {
        args = {x};
    }
    
    void forward(const std::vector<const Tensor<float>*>& inputs,
                Tensor<float>& output) override {
        // Use Armadillo's tanh
        fmat input_mat = inputs[0]->view();
        fmat result_mat = arma::tanh(input_mat);
        output = Tensor<float>(result_mat);
    }
    
    void backward(const std::vector<const Tensor<float>*>& inputs,
                 const Tensor<float>& output,
                 const Tensor<float>& dEdf,
                 std::vector<Tensor<float>>& gradients) override {
        // d/dx tanh(x) = 1 - tanh^2(x)
        Tensor<float> ones({output.shape[0], output.shape[1]});
        // Fill ones - need to add this to Tensor class
        Tensor<float> tanh_sq = output % output;
        Tensor<float> grad = ones - tanh_sq;
        gradients[0] = dEdf % grad;
    }
    
    std::string as_string(const std::vector<std::string>& args) const override {
        return "tanh(" + args[0] + ")";
    }
};

// Input node: represents input data
struct InputNode : public Node {
    InputNode() {
        args = {};  // No dependencies
    }
    
    void forward(const std::vector<const Tensor<float>*>& inputs,
                Tensor<float>& output) override {
        // Input is already set, nothing to compute
    }
    
    void backward(const std::vector<const Tensor<float>*>& inputs,
                 const Tensor<float>& output,
                 const Tensor<float>& dEdf,
                 std::vector<Tensor<float>>& gradients) override {
        // Input nodes don't propagate gradients
    }
    
    std::string as_string(const std::vector<std::string>& args) const override {
        return "input";
    }
};
```

---

### 2.2 Tensor Enhancements

**File:** Update `Tensor.cpp`

**Add to Tensor class:**
```cpp
template <typename T>
class Tensor {
private:
    vector<T> storage;
    vector<int> shape;
    vector<int> strides;
    
    // NEW: Gradient storage
    vector<T> gradient_storage;
    bool requires_grad;
    
public:
    // Existing methods...
    
    // NEW: Gradient methods
    bool has_gradient() const { return requires_grad; }
    void set_requires_grad(bool flag) { requires_grad = flag; }
    
    Tensor<T>& grad() {
        if (!requires_grad) {
            throw runtime_error("Tensor doesn't require gradient");
        }
        if (gradient_storage.empty()) {
            gradient_storage.resize(storage.size(), 0);
        }
        // Return tensor view of gradient
        // TODO: Implement this properly
    }
    
    void zero_grad() {
        if (requires_grad && !gradient_storage.empty()) {
            fill(gradient_storage.begin(), gradient_storage.end(), 0);
        }
    }
    
    // NEW: Fill with value
    void fill(T value) {
        std::fill(storage.begin(), storage.end(), value);
    }
    
    // NEW: Create tensor of ones
    static Tensor<T> ones(const vector<int>& shape) {
        Tensor<T> t(shape);
        t.fill(1.0);
        return t;
    }
    
    // NEW: Create tensor of zeros (already exists via constructor)
    static Tensor<T> zeros(const vector<int>& shape) {
        return Tensor<T>(shape);  // Already initializes to 0
    }
};
```

---

### 2.3 Forward and Backward Execution

**ComputationGraph Implementation:**

```cpp
const Tensor<float>& ComputationGraph::forward(VariableIndex final) {
    // Ensure we compute all dependencies
    for (unsigned i = 0; i <= final; ++i) {
        Node* node = nodes[i];
        
        // Gather inputs
        std::vector<const Tensor<float>*> inputs;
        for (VariableIndex arg : node->args) {
            inputs.push_back(&tensors[arg]);
        }
        
        // Compute forward
        node->forward(inputs, tensors[i]);
    }
    
    return tensors[final];
}

void ComputationGraph::backward(VariableIndex final) {
    // Initialize: gradient of loss w.r.t. itself is 1
    gradients[final].fill(1.0);
    
    // Backward pass in reverse topological order
    for (int i = final; i >= 0; --i) {
        Node* node = nodes[i];
        
        // Gather inputs (from forward pass)
        std::vector<const Tensor<float>*> inputs;
        for (VariableIndex arg : node->args) {
            inputs.push_back(&tensors[arg]);
        }
        
        // Compute gradients w.r.t. inputs
        std::vector<Tensor<float>> input_grads(node->args.size());
        node->backward(inputs, tensors[i], gradients[i], input_grads);
        
        // Accumulate gradients to input nodes
        for (size_t j = 0; j < node->args.size(); ++j) {
            VariableIndex arg = node->args[j];
            // Initialize gradient if needed
            if (gradients[arg].storage.empty()) {
                gradients[arg] = Tensor<float>(tensors[arg].shape);
            }
            // Accumulate
            gradients[arg] = gradients[arg] + input_grads[j];
        }
    }
}
```

---

### 2.4 Usage Example

**File:** `test_graph.cpp`

```cpp
#include "graph.h"
#include "expression.h"

int main() {
    // Create computation graph
    ComputationGraph cg;
    
    // Define inputs
    fmat X_data = {{1, 2}, {3, 4}};
    fmat Y_data = {{5, 6}, {7, 8}};
    
    // Create input nodes
    Expression X = input(cg, X_data);
    Expression Y = input(cg, Y_data);
    
    // Build computation: Z = X + Y
    Expression Z = X + Y;
    
    // Forward pass
    cg.forward(Z.i);
    
    // Print result
    std::cout << "Forward result:" << std::endl;
    Z.value().print();
    
    // Backward pass
    cg.backward(Z.i);
    
    // Print gradients
    std::cout << "Gradient w.r.t. X:" << std::endl;
    X.gradient().print();
    
    std::cout << "Gradient w.r.t. Y:" << std::endl;
    Y.gradient().print();
    
    return 0;
}
```

**Expected Output:**
```
Forward result:
   6    8
  10   12

Gradient w.r.t. X:
   1    1
   1    1

Gradient w.r.t. Y:
   1    1
   1    1
```

---

## 3. File Organization

### 3.1 Recommended Structure

```
Neural-Networks-CPP-using-Armadillo/
├── include/
│   ├── tensor.h           (extract from Tensor.cpp)
│   ├── graph.h            (NEW)
│   ├── expression.h       (NEW)
│   ├── node.h             (NEW)
│   ├── nodes/
│   │   ├── arithmetic.h   (NEW: Add, Sub, Mul, Div)
│   │   ├── matmul.h       (NEW: MatMul, Transpose)
│   │   ├── activation.h   (NEW: Tanh, Sigmoid, ReLU)
│   │   └── input.h        (NEW: Input, Parameter)
│   ├── parameter.h        (Phase 3)
│   ├── optimizer.h        (Phase 3)
│   └── layers.h           (Phase 4)
│
├── src/
│   ├── tensor.cpp         (current implementation)
│   ├── graph.cpp          (NEW)
│   ├── expression.cpp     (NEW)
│   ├── node.cpp           (NEW)
│   ├── nodes/
│   │   ├── arithmetic.cpp (NEW)
│   │   ├── matmul.cpp     (NEW)
│   │   ├── activation.cpp (NEW)
│   │   └── input.cpp      (NEW)
│   ├── parameter.cpp      (Phase 3)
│   ├── optimizer.cpp      (Phase 3)
│   └── layers.cpp         (Phase 4)
│
├── tests/
│   ├── test_tensor.cpp
│   ├── test_graph.cpp     (NEW)
│   ├── test_autograd.cpp  (NEW)
│   └── test_gradient_check.cpp (NEW)
│
├── examples/
│   ├── xor.cpp            (NEW: Simple XOR)
│   ├── linear_regression.cpp (NEW)
│   └── mnist.cpp          (Phase 4)
│
├── CMakeLists.txt         (NEW: Build system)
├── README.md
├── DYNET_COMPARISON.md    (This document)
└── ARCHITECTURE_ROADMAP.md (Current document)
```

---

## 4. Testing Strategy

### 4.1 Gradient Checking

**File:** `tests/test_gradient_check.cpp`

```cpp
#include "graph.h"
#include "expression.h"
#include <cmath>

// Numerical gradient estimation
float numerical_gradient(
    std::function<float(float)> f,
    float x,
    float eps = 1e-5
) {
    return (f(x + eps) - f(x - eps)) / (2 * eps);
}

void test_add_gradient() {
    ComputationGraph cg;
    
    // Test: z = x + y
    fmat x_data = {{2.0}};
    fmat y_data = {{3.0}};
    
    Expression x = input(cg, x_data);
    Expression y = input(cg, y_data);
    Expression z = x + y;
    
    cg.forward(z.i);
    cg.backward(z.i);
    
    // Check gradients
    float grad_x = x.gradient()(0, 0);
    float grad_y = y.gradient()(0, 0);
    
    assert(std::abs(grad_x - 1.0) < 1e-6);
    assert(std::abs(grad_y - 1.0) < 1e-6);
    
    std::cout << "✓ Addition gradient test passed" << std::endl;
}

void test_mul_gradient() {
    ComputationGraph cg;
    
    // Test: z = x * y (element-wise)
    fmat x_data = {{2.0}};
    fmat y_data = {{3.0}};
    
    Expression x = input(cg, x_data);
    Expression y = input(cg, y_data);
    Expression z = x * y;  // element-wise
    
    cg.forward(z.i);
    cg.backward(z.i);
    
    // dz/dx = y = 3.0
    // dz/dy = x = 2.0
    float grad_x = x.gradient()(0, 0);
    float grad_y = y.gradient()(0, 0);
    
    assert(std::abs(grad_x - 3.0) < 1e-6);
    assert(std::abs(grad_y - 2.0) < 1e-6);
    
    std::cout << "✓ Multiplication gradient test passed" << std::endl;
}

void test_matmul_gradient() {
    ComputationGraph cg;
    
    // Test: C = A * B (matrix multiplication)
    fmat A_data = {{1, 2}, {3, 4}};
    fmat B_data = {{5, 6}, {7, 8}};
    
    Expression A = input(cg, A_data);
    Expression B = input(cg, B_data);
    Expression C = dot(A, B);
    
    cg.forward(C.i);
    cg.backward(C.i);
    
    // Gradient check via numerical differentiation
    // (implement specific checks)
    
    std::cout << "✓ MatMul gradient test passed" << std::endl;
}

int main() {
    test_add_gradient();
    test_mul_gradient();
    test_matmul_gradient();
    
    std::cout << "All gradient tests passed!" << std::endl;
    return 0;
}
```

---

## 5. Build System

### 5.1 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(NeuroEngine)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/tensor.cpp
    src/graph.cpp
    src/expression.cpp
    src/node.cpp
    src/nodes/arithmetic.cpp
    src/nodes/matmul.cpp
    src/nodes/activation.cpp
    src/nodes/input.cpp
)

# Library
add_library(neuroengine ${SOURCES})
target_link_libraries(neuroengine ${ARMADILLO_LIBRARIES})

# Tests
enable_testing()

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor neuroengine)
add_test(NAME TensorTests COMMAND test_tensor)

add_executable(test_graph tests/test_graph.cpp)
target_link_libraries(test_graph neuroengine)
add_test(NAME GraphTests COMMAND test_graph)

add_executable(test_autograd tests/test_autograd.cpp)
target_link_libraries(test_autograd neuroengine)
add_test(NAME AutogradTests COMMAND test_autograd)

# Examples
add_executable(xor examples/xor.cpp)
target_link_libraries(xor neuroengine)

add_executable(linear_regression examples/linear_regression.cpp)
target_link_libraries(linear_regression neuroengine)
```

---

## 6. Implementation Checklist

### Phase 2.1: Basic Infrastructure (Week 1-2)
- [ ] Extract Tensor to separate header file
- [ ] Add gradient storage to Tensor class
- [ ] Implement ComputationGraph skeleton
- [ ] Implement Expression wrapper
- [ ] Implement Node base class
- [ ] Write unit tests for basic infrastructure

### Phase 2.2: Basic Operations (Week 3-4)
- [ ] Implement InputNode
- [ ] Implement AddNode with gradients
- [ ] Implement SubNode with gradients
- [ ] Implement CwiseMultNode with gradients
- [ ] Test gradient checking for each operation

### Phase 2.3: Matrix Operations (Week 5-6)
- [ ] Implement MatMulNode with gradients
- [ ] Implement TransposeNode
- [ ] Test matrix operation gradients
- [ ] Build simple linear regression example

### Phase 2.4: Activation Functions (Week 7-8)
- [ ] Implement TanhNode
- [ ] Implement SigmoidNode
- [ ] Implement ReLUNode
- [ ] Build XOR example with activations

### Phase 2.5: Loss Functions (Week 9-10)
- [ ] Implement SquaredErrorNode
- [ ] Implement CrossEntropyNode
- [ ] End-to-end training test

### Phase 2.6: Polish and Documentation (Week 11-12)
- [ ] Complete gradient checking for all operations
- [ ] Write comprehensive tests
- [ ] Document API with examples
- [ ] Create tutorial notebook/guide

---

## 7. Key Design Decisions

### 7.1 Why Separate Graph and Expression?
- **Graph**: Low-level, manages memory and execution
- **Expression**: High-level, user-facing API with operator overloading
- **Benefit**: Clean separation of concerns

### 7.2 Why Node Polymorphism?
- Each operation self-contained
- Easy to add new operations
- Follows Open/Closed Principle

### 7.3 Memory Management Strategy
- Graph owns all nodes (via `unique_ptr` or manual management)
- Tensors pre-allocated in graph
- Avoids dynamic allocation during forward/backward

### 7.4 Type System
- Start with `float` only
- Templates can be added later if needed
- Keeps initial implementation simple

---

## 8. Common Pitfalls to Avoid

### 8.1 Gradient Accumulation
**Problem:** Multiple paths to same node
```cpp
// z uses x twice
Expression z = x + x;
```
**Solution:** Always accumulate gradients, never overwrite

### 8.2 Memory Leaks
**Problem:** Node pointers not cleaned up
**Solution:** Use `std::unique_ptr<Node>` or implement careful RAII

### 8.3 Stale Expressions
**Problem:** Using Expression after graph is cleared
**Solution:** Graph ID checking (already in design)

### 8.4 Broadcasting
**Problem:** Shape mismatches in operations
**Solution:** Clear error messages, document shape requirements

---

## 9. Performance Optimization (Future)

### 9.1 Memory Pooling
```cpp
class MemoryPool {
    std::vector<float> pool;
    size_t offset;
    
public:
    float* allocate(size_t size) {
        // Return pointer to pre-allocated memory
    }
    
    void reset() {
        offset = 0;  // Reuse memory
    }
};
```

### 9.2 In-Place Operations
Some operations can reuse input memory:
```cpp
// Instead of: y = tanh(x)
// Do: tanh_inplace(x)  // Modifies x
```

### 9.3 Operation Fusion
Combine multiple operations into one kernel:
```cpp
// Instead of: y = tanh(W*x + b)
// Fuse: y = affine_tanh(W, x, b)
```

---

## 10. Learning Resources

### 10.1 Recommended Reading
1. **Automatic Differentiation**: "Automatic Differentiation in Machine Learning: a Survey" (Baydin et al.)
2. **DyNet Paper**: "DyNet: The Dynamic Neural Network Toolkit" (Neubig et al.)
3. **Computational Graphs**: Deep Learning Book, Chapter 6 (Goodfellow et al.)

### 10.2 Code References
1. **DyNet Source**: `github.com/clab/dynet`
2. **PyTorch Autograd**: `github.com/pytorch/pytorch/tree/master/torch/csrc/autograd`
3. **TinyGrad**: `github.com/geohot/tinygrad` (educational)

### 10.3 Testing References
1. **Gradient Checking**: CS231n notes on gradient checking
2. **Numerical Stability**: "What Every Computer Scientist Should Know About Floating-Point Arithmetic"

---

## 11. Success Metrics

### Phase 2 Complete When:
1. ✅ Can build a computation graph with 5+ operations
2. ✅ Automatic differentiation works correctly (gradient checks pass)
3. ✅ Can train a simple model (XOR or linear regression)
4. ✅ All tests pass with >95% coverage
5. ✅ Documentation is complete with examples

---

## 12. Next Steps After Phase 2

### Phase 3: Parameters & Optimizers
1. Implement ParameterCollection
2. Implement SGD optimizer
3. Implement Adam optimizer
4. Add model save/load

### Phase 4: Neural Network Layers
1. Linear layer
2. Loss functions
3. Multi-layer perceptron
4. MNIST example

### Phase 5: Recurrent Networks
1. RNN cell
2. LSTM implementation
3. Sequence models

---

## Conclusion

This roadmap provides a clear path from the current tensor-only implementation to a DyNet-inspired deep learning framework. The key is to:

1. **Start small**: Basic graph with 2-3 operations
2. **Test thoroughly**: Gradient checking is essential
3. **Iterate**: Add operations one at a time
4. **Document**: Keep examples updated

With focused effort, Phase 2 can be completed in 2-3 months, unlocking the ability to train neural networks.

**The journey of a thousand miles begins with a single step - and that step is the computational graph.**
