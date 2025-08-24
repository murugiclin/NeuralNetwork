# ML-CPP-Core

ML-CPP-Core is a lightweight C++ library implementing fundamental machine learning algorithms from scratch: **Linear Regression**, **Decision Trees**, **K-Means Clustering**, and a **Feedforward Neural Network**. This repository is designed for educational purposes, offering clean, minimal implementations to help you understand the core mechanics of these algorithms. Ideal for students, hobbyists, and developers learning machine learning concepts in C++.

---

## Features

- **Linear Regression:** Fits a line to data using the least squares method for regression tasks.
- **Decision Tree:** Implements a binary classification tree with Gini impurity-based splitting.
- **K-Means Clustering:** Groups data points into *k* clusters based on Euclidean distance.
- **Neural Network:** Feedforward neural network with one hidden layer, trained via backpropagation (demonstrates XOR solution).
- **Example Usage:** The main program demonstrates each algorithm with practical examples.

---

## Getting Started

### Prerequisites

- C++ compiler supporting C++11 or later (e.g., `g++`, `clang++`)
- Standard Template Library (STL)
- **No external dependencies required**

### Installation

```sh
git clone https://github.com/murugiclin/ML-CPP-Core.git
cd ML-CPP-Core
g++ MachineLearning.cpp -o ml_core
./ml_core
```

---

## Usage

The `main()` function in `MachineLearning.cpp` demonstrates example usage for each algorithm:

- **Linear Regression:** Fits a line to sample data (e.g., y = 2x) and predicts a new value.
- **Decision Tree:** Classifies 2D points (e.g., binary classification on a simple dataset).
- **K-Means:** Clusters 2D points into two groups.
- **Neural Network:** Trains on the XOR problem to demonstrate non-linear classification.

To experiment, modify the `main()` function with your own datasets or parameters.

---

## Code Structure

- **LinearRegression:** Class implementing linear regression with `fit` and `predict` methods.
- **DecisionTree:** Class for binary classification with configurable max depth.
- **KMeans:** Class for clustering with configurable cluster count.
- **NeuralNetwork:** Feedforward neural network with one hidden layer, trained via backpropagation.
- **All implementations:** Contained in `MachineLearning.cpp`.

---

## Example Output

```
Linear Regression: y = 2x + 0
Prediction for x=6: 12
Decision Tree prediction for [0.5, 0.5]: 0
K-Means cluster assignments: 0 0 1 1 1
Epoch 0 Error: 0.25
...
Predictions:
Input: [0, 0] Output: 0.03
Input: [0, 1] Output: 0.95
Input: [1, 0] Output: 0.94
Input: [1, 1] Output: 0.06
```

---

## Notes

- These are simplified implementations for educational purposes. They lack optimizations like batch processing, advanced regularization, or vectorized operations.
- The Decision Tree uses a basic Gini impurity criterion and does not handle continuous outputs.
- K-Means uses random centroid initialization, which may lead to varying results.
- The Neural Network is a basic feedforward model with sigmoid activation and mean squared error loss.
- For production use, consider established libraries like [MLpack](https://www.mlpack.org/), [Eigen](https://eigen.tuxfamily.org/), or [TensorFlow C++ API](https://www.tensorflow.org/install/lang_c).

---

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for bug reports, suggestions, or improvements. Potential areas for enhancement:

- Adding more algorithms (e.g., SVM, KNN)
- Optimizing performance with vectorization
- Supporting additional activation functions or loss metrics

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by the need for clear, educational machine learning implementations in C++.
- Thanks to the open-source community for resources on algorithm fundamentals.
