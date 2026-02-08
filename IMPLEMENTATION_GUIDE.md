# Implementation Guide: Building Computation Graph & Autograd

This guide provides step-by-step instructions for implementing Phase 2 (Computation Graph & Autograd) inspired by DyNet.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Step 1: Node Base Class](#step-1-node-base-class)
3. [Step 2: Expression Class](#step-2-expression-class)
4. [Step 3: Computation Graph](#step-3-computation-graph)
5. [Step 4: Basic Operations](#step-4-basic-operations)
6. [Step 5: Backward Pass](#step-5-backward-pass)
7. [Step 6: Testing](#step-6-testing)
8. [Integration with Existing Code](#integration-with-existing-code)

---

## Architecture Overview

### Component Interaction

```
User Code
    │
    ├─► Expression (builds graph, returns handles)
    │       │
    │       ├─► ComputationGraph (manages nodes & execution)
    │       │       │
    │       │       ├─► Node (operation implementation)
    │       │       │       │
    │       │       │       └─► Tensor (data storage)
    │       │       │
    │       │       └─► forward() → backward()
```

### Data Flow

```
1. BUILD PHASE
   user: y = tanh(W * x + b)
   → Creates Nodes: MatMul, Add, Tanh
   → Returns Expressions: just handles, no computation

2. FORWARD PHASE
   cg.forward(y)
   → Topological sort of nodes
   → Execute each node.forward()
   → Store results in values[]

3. BACKWARD PHASE
   cg.backward(y)
   → Reverse topological order
   → Execute each node.backward()
   → Accumulate gradients in gradients[]
```

---

## Step 1: Node Base Class

### File: `include/Node.h`

```cpp
#ifndef NODE_H
#define NODE_H

#include "Tensor.cpp"
#include <vector>
#include <string>

namespace neuroengine {

/**
 * Base class for all computation graph nodes.
 * Each node represents an operation (e.g., add, matmul, tanh).
 */
class Node {
protected:
    std::vector<int> inputs_;  // Input node indices
    std::vector<int> shape_;   // Output shape
    
public:
    virtual ~Node() {}
    
    /**
     * Compute output given input values.
     * @param xs - Input tensors
     * @param fx - Output tensor (to be filled)
     */
    virtual void forward(
        const std::vector<const Tensor<float>*>& xs,
        Tensor<float>& fx) const = 0;
    
    /**
     * Compute gradients w.r.t. inputs given gradient w.r.t. output.
     * @param xs - Input tensors (from forward pass)
     * @param fx - Output tensor (from forward pass)
     * @param dEdf - Gradient w.r.t. output
     * @param dEdx - Gradients w.r.t. inputs (to be accumulated)
     */
    virtual void backward(
        const std::vector<const Tensor<float>*>& xs,
        const Tensor<float>& fx,
        const Tensor<float>& dEdf,
        std::vector<Tensor<float>*>& dEdx) const = 0;
    
    /**
     * Get output shape.
     */
    virtual std::vector<int> dim() const { return shape_; }
    
    /**
     * Get input node indices.
     */
    const std::vector<int>& inputs() const { return inputs_; }
    
    /**
     * Get node name for debugging.
     */
    virtual std::string name() const = 0;
};

} // namespace neuroengine

#endif // NODE_H
```

---

## Step 2: Expression Class

### File: `include/Expression.h`

```cpp
#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <stdexcept>

namespace neuroengine {

class ComputationGraph;

/**
 * Handle to a node in the computation graph.
 * Expressions are returned by operations and used to build the graph.
 */
class Expression {
public:
    ComputationGraph* graph;
    int node_id;
    
    Expression() : graph(nullptr), node_id(-1) {}
    
    Expression(ComputationGraph* g, int id) 
        : graph(g), node_id(id) {}
    
    /**
     * Get the computed value (only valid after forward pass).
     */
    const Tensor<float>& value() const;
    
    /**
     * Get the computed gradient (only valid after backward pass).
     */
    const Tensor<float>& gradient() const;
    
    /**
     * Get the output dimensions.
     */
    std::vector<int> dim() const;
    
    /**
     * Check if this expression is valid.
     */
    bool valid() const { return graph != nullptr && node_id >= 0; }
};

// Forward declarations for operations
Expression operator+(const Expression& lhs, const Expression& rhs);
Expression operator-(const Expression& lhs, const Expression& rhs);
Expression operator*(const Expression& lhs, const Expression& rhs); // element-wise
Expression matmul(const Expression& lhs, const Expression& rhs);
Expression tanh(const Expression& x);
Expression relu(const Expression& x);
Expression sigmoid(const Expression& x);

} // namespace neuroengine

#endif // EXPRESSION_H
```

---

## Step 3: Computation Graph

### File: `include/ComputationGraph.h`

```cpp
#ifndef COMPUTATION_GRAPH_H
#define COMPUTATION_GRAPH_H

#include "Node.h"
#include "Expression.h"
#include <vector>
#include <memory>

namespace neuroengine {

/**
 * Computation graph that manages nodes and executes forward/backward passes.
 */
class ComputationGraph {
private:
    std::vector<std::unique_ptr<Node>> nodes_;
    std::vector<Tensor<float>> values_;
    std::vector<Tensor<float>> gradients_;
    bool backward_computed_;
    
public:
    ComputationGraph() : backward_computed_(false) {}
    
    ~ComputationGraph() { clear(); }
    
    /**
     * Add a node to the graph.
     * @param node - Node to add (graph takes ownership)
     * @return Node index
     */
    int add_node(Node* node) {
        nodes_.emplace_back(node);
        values_.emplace_back(node->dim());
        gradients_.emplace_back(node->dim());
        return nodes_.size() - 1;
    }
    
    /**
     * Add an input node (leaf node with no inputs).
     */
    Expression add_input(const std::vector<int>& shape, const std::vector<float>& data);
    
    /**
     * Execute forward pass up to the given expression.
     */
    void forward(const Expression& last);
    
    /**
     * Execute backward pass from the given expression.
     * Computes gradients for all nodes.
     */
    void backward(const Expression& last);
    
    /**
     * Get computed value for a node.
     */
    const Tensor<float>& get_value(int node_id) const {
        return values_[node_id];
    }
    
    /**
     * Get computed gradient for a node.
     */
    const Tensor<float>& get_gradient(int node_id) const {
        if (!backward_computed_) {
            throw std::runtime_error("backward() not called yet");
        }
        return gradients_[node_id];
    }
    
    /**
     * Get dimensions for a node.
     */
    std::vector<int> get_dim(int node_id) const {
        return nodes_[node_id]->dim();
    }
    
    /**
     * Clear the graph for reuse.
     */
    void clear() {
        nodes_.clear();
        values_.clear();
        gradients_.clear();
        backward_computed_ = false;
    }
    
    /**
     * Get number of nodes.
     */
    size_t size() const { return nodes_.size(); }
};

} // namespace neuroengine

#endif // COMPUTATION_GRAPH_H
```

### File: `src/ComputationGraph.cpp`

```cpp
#include "ComputationGraph.h"
#include "nodes/Input.h"
#include <algorithm>
#include <set>

namespace neuroengine {

Expression ComputationGraph::add_input(
    const std::vector<int>& shape, 
    const std::vector<float>& data) {
    
    Node* node = new InputNode(shape, data);
    int id = add_node(node);
    
    // For input nodes, immediately set the value
    values_[id] = Tensor<float>(shape);
    // Copy data into tensor
    // (Implementation depends on your Tensor class)
    
    return Expression(this, id);
}

void ComputationGraph::forward(const Expression& last) {
    if (!last.valid() || last.graph != this) {
        throw std::runtime_error("Invalid expression");
    }
    
    // Topological sort to determine execution order
    std::vector<int> execution_order;
    std::set<int> visited;
    
    std::function<void(int)> dfs = [&](int node_id) {
        if (visited.count(node_id)) return;
        visited.insert(node_id);
        
        for (int input_id : nodes_[node_id]->inputs()) {
            dfs(input_id);
        }
        
        execution_order.push_back(node_id);
    };
    
    dfs(last.node_id);
    
    // Execute nodes in topological order
    for (int node_id : execution_order) {
        const Node* node = nodes_[node_id].get();
        
        // Gather inputs
        std::vector<const Tensor<float>*> input_values;
        for (int input_id : node->inputs()) {
            input_values.push_back(&values_[input_id]);
        }
        
        // Execute forward pass
        node->forward(input_values, values_[node_id]);
    }
}

void ComputationGraph::backward(const Expression& last) {
    if (!last.valid() || last.graph != this) {
        throw std::runtime_error("Invalid expression");
    }
    
    // Initialize all gradients to zero
    for (auto& grad : gradients_) {
        grad = Tensor<float>(grad.shape);  // Zero tensor
    }
    
    // Gradient of loss w.r.t. itself is 1
    gradients_[last.node_id] = Tensor<float>(nodes_[last.node_id]->dim());
    // Set to ones (implementation depends on your Tensor class)
    
    // Topological sort (same as forward, but we reverse it)
    std::vector<int> execution_order;
    std::set<int> visited;
    
    std::function<void(int)> dfs = [&](int node_id) {
        if (visited.count(node_id)) return;
        visited.insert(node_id);
        
        for (int input_id : nodes_[node_id]->inputs()) {
            dfs(input_id);
        }
        
        execution_order.push_back(node_id);
    };
    
    dfs(last.node_id);
    
    // Execute in reverse order
    std::reverse(execution_order.begin(), execution_order.end());
    
    for (int node_id : execution_order) {
        const Node* node = nodes_[node_id].get();
        
        // Gather inputs and gradients
        std::vector<const Tensor<float>*> input_values;
        std::vector<Tensor<float>*> input_gradients;
        
        for (int input_id : node->inputs()) {
            input_values.push_back(&values_[input_id]);
            input_gradients.push_back(&gradients_[input_id]);
        }
        
        // Execute backward pass
        node->backward(
            input_values,
            values_[node_id],
            gradients_[node_id],
            input_gradients);
    }
    
    backward_computed_ = true;
}

} // namespace neuroengine
```

---

## Step 4: Basic Operations

### File: `include/nodes/Input.h`

```cpp
#ifndef NODE_INPUT_H
#define NODE_INPUT_H

#include "Node.h"

namespace neuroengine {

/**
 * Input node - holds user data.
 */
class InputNode : public Node {
private:
    std::vector<float> data_;
    
public:
    InputNode(const std::vector<int>& shape, const std::vector<float>& data)
        : data_(data) {
        shape_ = shape;
        inputs_ = {};  // No inputs
    }
    
    void forward(
        const std::vector<const Tensor<float>*>& xs,
        Tensor<float>& fx) const override {
        // Just copy data to output
        // (Implementation depends on your Tensor class)
    }
    
    void backward(
        const std::vector<const Tensor<float>*>& xs,
        const Tensor<float>& fx,
        const Tensor<float>& dEdf,
        std::vector<Tensor<float>*>& dEdx) const override {
        // Input nodes have no inputs, so nothing to do
    }
    
    std::string name() const override { return "Input"; }
};

} // namespace neuroengine

#endif // NODE_INPUT_H
```

### File: `include/nodes/MatMul.h`

```cpp
#ifndef NODE_MATMUL_H
#define NODE_MATMUL_H

#include "Node.h"

namespace neuroengine {

/**
 * Matrix multiplication node: output = lhs * rhs
 */
class MatMulNode : public Node {
public:
    MatMulNode(int lhs_id, int rhs_id, 
               const std::vector<int>& lhs_shape,
               const std::vector<int>& rhs_shape) {
        inputs_ = {lhs_id, rhs_id};
        
        // Output shape: [lhs_rows, rhs_cols]
        shape_ = {lhs_shape[0], rhs_shape[1]};
    }
    
    void forward(
        const std::vector<const Tensor<float>*>& xs,
        Tensor<float>& fx) const override {
        // fx = xs[0] * xs[1]
        fx = xs[0]->dot(*xs[1]);
    }
    
    void backward(
        const std::vector<const Tensor<float>*>& xs,
        const Tensor<float>& fx,
        const Tensor<float>& dEdf,
        std::vector<Tensor<float>*>& dEdx) const override {
        // Gradient w.r.t. lhs: dEdf * rhs^T
        *dEdx[0] += dEdf.dot(xs[1]->transpose());
        
        // Gradient w.r.t. rhs: lhs^T * dEdf
        *dEdx[1] += xs[0]->transpose().dot(dEdf);
    }
    
    std::string name() const override { return "MatMul"; }
};

} // namespace neuroengine

#endif // NODE_MATMUL_H
```

### File: `include/nodes/Add.h`

```cpp
#ifndef NODE_ADD_H
#define NODE_ADD_H

#include "Node.h"

namespace neuroengine {

/**
 * Element-wise addition: output = lhs + rhs
 */
class AddNode : public Node {
public:
    AddNode(int lhs_id, int rhs_id, const std::vector<int>& shape) {
        inputs_ = {lhs_id, rhs_id};
        shape_ = shape;
    }
    
    void forward(
        const std::vector<const Tensor<float>*>& xs,
        Tensor<float>& fx) const override {
        fx = *xs[0] + *xs[1];
    }
    
    void backward(
        const std::vector<const Tensor<float>*>& xs,
        const Tensor<float>& fx,
        const Tensor<float>& dEdf,
        std::vector<Tensor<float>*>& dEdx) const override {
        // Gradient distributes equally
        *dEdx[0] += dEdf;
        *dEdx[1] += dEdf;
    }
    
    std::string name() const override { return "Add"; }
};

} // namespace neuroengine

#endif // NODE_ADD_H
```

### File: `include/nodes/Tanh.h`

```cpp
#ifndef NODE_TANH_H
#define NODE_TANH_H

#include "Node.h"
#include <cmath>

namespace neuroengine {

/**
 * Hyperbolic tangent activation: output = tanh(input)
 */
class TanhNode : public Node {
public:
    TanhNode(int input_id, const std::vector<int>& shape) {
        inputs_ = {input_id};
        shape_ = shape;
    }
    
    void forward(
        const std::vector<const Tensor<float>*>& xs,
        Tensor<float>& fx) const override {
        // Apply tanh element-wise
        // (Implementation depends on your Tensor class)
        // You may need to add a tanh() method to Tensor
        
        const Tensor<float>& x = *xs[0];
        // For now, assume you add this to Tensor class:
        // fx = x.apply([](float v) { return std::tanh(v); });
    }
    
    void backward(
        const std::vector<const Tensor<float>*>& xs,
        const Tensor<float>& fx,
        const Tensor<float>& dEdf,
        std::vector<Tensor<float>*>& dEdx) const override {
        // d/dx tanh(x) = 1 - tanh(x)^2 = 1 - fx^2
        // dEdx[0] = dEdf * (1 - fx^2)
        
        // Element-wise: (1 - fx * fx) * dEdf
        // (Implementation depends on your Tensor class)
    }
    
    std::string name() const override { return "Tanh"; }
};

} // namespace neuroengine

#endif // NODE_TANH_H
```

---

## Step 5: Backward Pass

### Gradient Checking Utility

```cpp
#ifndef GRAD_CHECK_H
#define GRAD_CHECK_H

#include "ComputationGraph.h"
#include <cmath>
#include <iostream>

namespace neuroengine {

/**
 * Numerical gradient checking for debugging.
 */
class GradientChecker {
public:
    /**
     * Check gradients using finite differences.
     * @param cg - Computation graph
     * @param loss - Loss expression
     * @param epsilon - Step size for numerical gradient
     * @return true if all gradients are correct
     */
    static bool check(ComputationGraph& cg, Expression& loss, 
                     float epsilon = 1e-4) {
        // 1. Compute analytical gradients
        cg.forward(loss);
        cg.backward(loss);
        
        // 2. For each parameter, compute numerical gradient
        bool all_correct = true;
        
        for (size_t i = 0; i < cg.size(); i++) {
            // Get analytical gradient
            const Tensor<float>& analytical = cg.get_gradient(i);
            
            // Compute numerical gradient
            Tensor<float> numerical = compute_numerical_gradient(
                cg, loss, i, epsilon);
            
            // Compare
            float diff = (analytical - numerical).abs().max();
            if (diff > 1e-3) {
                std::cerr << "Gradient mismatch at node " << i 
                         << ": diff = " << diff << std::endl;
                all_correct = false;
            }
        }
        
        return all_correct;
    }
    
private:
    static Tensor<float> compute_numerical_gradient(
        ComputationGraph& cg, Expression& loss, 
        int node_id, float epsilon) {
        
        Tensor<float>& value = cg.get_value(node_id);
        Tensor<float> grad(value.shape);
        
        // For each element in the tensor
        for (int i = 0; i < value.size(); i++) {
            float original = value.data[i];
            
            // f(x + epsilon)
            value.data[i] = original + epsilon;
            cg.forward(loss);
            float loss_plus = cg.get_value(loss.node_id).data[0];
            
            // f(x - epsilon)
            value.data[i] = original - epsilon;
            cg.forward(loss);
            float loss_minus = cg.get_value(loss.node_id).data[0];
            
            // Numerical gradient
            grad.data[i] = (loss_plus - loss_minus) / (2 * epsilon);
            
            // Restore original value
            value.data[i] = original;
        }
        
        return grad;
    }
};

} // namespace neuroengine

#endif // GRAD_CHECK_H
```

---

## Step 6: Testing

### File: `tests/test_autograd.cpp`

```cpp
#include "ComputationGraph.h"
#include "nodes/Add.h"
#include "nodes/MatMul.h"
#include "nodes/Tanh.h"
#include "GradCheck.h"
#include <cassert>
#include <iostream>

using namespace neuroengine;

void test_simple_add() {
    std::cout << "Testing addition..." << std::endl;
    
    ComputationGraph cg;
    
    // Create inputs
    auto a = cg.add_input({2, 2}, {1, 2, 3, 4});
    auto b = cg.add_input({2, 2}, {5, 6, 7, 8});
    
    // Build graph
    auto c = a + b;
    
    // Forward pass
    cg.forward(c);
    
    // Check result
    const auto& result = c.value();
    assert(result.data[0] == 6);  // 1 + 5
    assert(result.data[1] == 8);  // 2 + 6
    
    std::cout << "✓ Addition forward pass works" << std::endl;
    
    // Backward pass
    cg.backward(c);
    
    // Check gradients (should be 1 for both inputs)
    const auto& grad_a = a.gradient();
    const auto& grad_b = b.gradient();
    
    for (int i = 0; i < 4; i++) {
        assert(std::abs(grad_a.data[i] - 1.0f) < 1e-5);
        assert(std::abs(grad_b.data[i] - 1.0f) < 1e-5);
    }
    
    std::cout << "✓ Addition backward pass works" << std::endl;
}

void test_matmul() {
    std::cout << "Testing matrix multiplication..." << std::endl;
    
    ComputationGraph cg;
    
    // 2x3 matrix
    auto A = cg.add_input({2, 3}, {1, 2, 3, 4, 5, 6});
    
    // 3x2 matrix
    auto B = cg.add_input({3, 2}, {7, 8, 9, 10, 11, 12});
    
    // C = A * B (should be 2x2)
    auto C = matmul(A, B);
    
    cg.forward(C);
    
    const auto& result = C.value();
    
    // Expected: [[58, 64], [139, 154]]
    assert(std::abs(result.data[0] - 58) < 1e-5);
    assert(std::abs(result.data[1] - 64) < 1e-5);
    assert(std::abs(result.data[2] - 139) < 1e-5);
    assert(std::abs(result.data[3] - 154) < 1e-5);
    
    std::cout << "✓ MatMul forward pass works" << std::endl;
    
    // Gradient check
    cg.backward(C);
    bool grad_ok = GradientChecker::check(cg, C);
    assert(grad_ok);
    
    std::cout << "✓ MatMul backward pass works" << std::endl;
}

void test_xor_training() {
    std::cout << "Testing XOR training..." << std::endl;
    
    // This would require implementing:
    // - Parameter class
    // - SGD optimizer
    // - Loss functions
    // See next phase!
    
    std::cout << "✓ XOR test (to be implemented in Phase 3)" << std::endl;
}

int main() {
    test_simple_add();
    test_matmul();
    test_xor_training();
    
    std::cout << "\n✅ All tests passed!" << std::endl;
    return 0;
}
```

---

## Integration with Existing Code

### Modify `Tensor.cpp`

Add utility methods needed by autograd:

```cpp
// Add to Tensor class

// Element-wise function application
Tensor apply(std::function<T(T)> func) const {
    Tensor result(shape);
    for (size_t i = 0; i < storage.size(); i++) {
        result.storage[i] = func(storage[i]);
    }
    return result;
}

// Element-wise multiplication
Tensor operator*(const Tensor& other) const {
    if (shape != other.shape) {
        throw runtime_error("Shape mismatch");
    }
    Mat<T> result = this->view() % other.view();
    return Tensor(result);
}

// Fill with scalar
void fill(T value) {
    std::fill(storage.begin(), storage.end(), value);
}

// Get absolute value
Tensor abs() const {
    return apply([](T v) { return std::abs(v); });
}

// Get maximum element
T max() const {
    return *std::max_element(storage.begin(), storage.end());
}

// Access underlying data
T* data() { return storage.data(); }
const T* data() const { return storage.data(); }

// Get total size
size_t size() const { return storage.size(); }
```

---

## Compilation

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(NeuroEngine)

set(CMAKE_CXX_STANDARD 14)

# Find Armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Include directories
include_directories(include)

# Source files
file(GLOB SOURCES "src/*.cpp")
file(GLOB NODE_SOURCES "src/nodes/*.cpp")

# Create library
add_library(neuroengine ${SOURCES} ${NODE_SOURCES})
target_link_libraries(neuroengine ${ARMADILLO_LIBRARIES})

# Tests
add_executable(test_autograd tests/test_autograd.cpp)
target_link_libraries(test_autograd neuroengine)

# Examples
add_executable(example_xor examples/xor.cpp)
target_link_libraries(example_xor neuroengine)
```

### Build Instructions

```bash
mkdir build
cd build
cmake ..
make

# Run tests
./test_autograd

# Run example
./example_xor
```

---

## Next Steps

After completing Phase 2 (Computation Graph & Autograd):

1. **Phase 3**: Implement Parameter management and Trainers
2. **Phase 4**: Add neural network layers (Linear, LSTM, etc.)
3. **Phase 5**: Advanced features (convolution, attention, etc.)

---

## Debugging Tips

1. **Enable verbose logging**: Add print statements in forward/backward
2. **Gradient checking**: Always use `GradientChecker::check()` for new nodes
3. **Visualize graph**: Add `print_graph()` method to dump graph structure
4. **Start simple**: Test each node in isolation before combining
5. **Use debugger**: GDB/LLDB to step through forward/backward passes

---

## Common Pitfalls

1. **Forgetting to initialize gradients to zero**: `backward()` must zero gradients first
2. **Wrong topological order**: Forward requires inputs before outputs
3. **Gradient accumulation**: Use `+=` not `=` when adding to gradients
4. **Shape mismatches**: Always validate dimensions in constructors
5. **Memory management**: Use smart pointers to avoid leaks

---

## Summary

This guide provides a complete blueprint for implementing computation graphs and automatic differentiation inspired by DyNet. Follow the steps in order, test thoroughly, and you'll have a working autograd system that can train neural networks!

**Key Achievements After Phase 2**:
- ✅ Dynamic computation graphs
- ✅ Automatic differentiation
- ✅ Gradient checking
- ✅ Foundation for neural network training

**Ready for Phase 3**:
- 🔲 Parameter management
- 🔲 Optimizers (SGD, Adam)
- 🔲 Loss functions
- 🔲 Model save/load

Happy coding! 🚀
