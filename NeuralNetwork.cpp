#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <cassert>

// Utility functions
namespace nn_utils {
    // Sigmoid activation function
    inline double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Derivative of sigmoid given output of sigmoid
    inline double sigmoid_derivative(double sig) {
        return sig * (1.0 - sig);
    }

    // Matrix-vector multiplication (matrix: n x m, vector: n)
    std::vector<double> matmul(const std::vector<std::vector<double>>& matrix,
                               const std::vector<double>& vec) {
        assert(matrix.size() == vec.size());
        std::vector<double> result(matrix[0].size(), 0.0);
        for (size_t j = 0; j < matrix[0].size(); ++j)
            for (size_t i = 0; i < matrix.size(); ++i)
                result[j] += matrix[i][j] * vec[i];
        return result;
    }

    // Vector addition
    std::vector<double> add(const std::vector<double>& a, const std::vector<double>& b) {
        assert(a.size() == b.size());
        std::vector<double> result(a.size());
        for (size_t i = 0; i < a.size(); ++i)
            result[i] = a[i] + b[i];
        return result;
    }

    // Apply function elementwise
    std::vector<double> apply(const std::vector<double>& vec, double(*func)(double)) {
        std::vector<double> result(vec.size());
        std::transform(vec.begin(), vec.end(), result.begin(), func);
        return result;
    }
}

class NeuralNetwork {
private:
    int input_size, hidden_size, output_size;
    std::vector<std::vector<double>> weights_ih; // [input_size][hidden_size]
    std::vector<std::vector<double>> weights_ho; // [hidden_size][output_size]
    std::vector<double> bias_h; // [hidden_size]
    std::vector<double> bias_o; // [output_size]
    double learning_rate;

    // Initialize weights and biases with Xavier initialization
    void initialize_weights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        // Xavier/Glorot initialization
        double limit_ih = std::sqrt(6.0 / (input_size + hidden_size));
        double limit_ho = std::sqrt(6.0 / (hidden_size + output_size));
        std::uniform_real_distribution<double> dist_ih(-limit_ih, limit_ih);
        std::uniform_real_distribution<double> dist_ho(-limit_ho, limit_ho);

        weights_ih.resize(input_size, std::vector<double>(hidden_size));
        for (int i = 0; i < input_size; ++i)
            for (int j = 0; j < hidden_size; ++j)
                weights_ih[i][j] = dist_ih(gen);

        weights_ho.resize(hidden_size, std::vector<double>(output_size));
        for (int i = 0; i < hidden_size; ++i)
            for (int j = 0; j < output_size; ++j)
                weights_ho[i][j] = dist_ho(gen);

        bias_h.resize(hidden_size, 0.0);
        bias_o.resize(output_size, 0.0);
    }

public:
    NeuralNetwork(int input, int hidden, int output, double lr = 0.1)
        : input_size(input), hidden_size(hidden), output_size(output), learning_rate(lr) {
        initialize_weights();
    }

    // Forward propagation, returns tuple: {hidden_activations, output_activations}
    std::pair<std::vector<double>, std::vector<double>> forward(const std::vector<double>& inputs) const {
        using namespace nn_utils;
        auto hidden_pre = add(matmul(weights_ih, inputs), bias_h);
        auto hidden = apply(hidden_pre, sigmoid);

        auto output_pre = add(matmul(weights_ho, hidden), bias_o);
        auto output = apply(output_pre, sigmoid);
        return { hidden, output };
    }

    std::vector<double> predict(const std::vector<double>& inputs) const {
        return forward(inputs).second;
    }

    // Train with backpropagation (batch gradient descent)
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<std::vector<double>>& y,
               int epochs) {
        using namespace nn_utils;
        assert(X.size() == y.size());

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;
            // Shuffle data for stochasticity
            std::vector<size_t> indices(X.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);

            for (size_t idx : indices) {
                const auto& input = X[idx];
                const auto& target = y[idx];

                // --- Forward pass ---
                auto hidden_pre = add(matmul(weights_ih, input), bias_h);
                auto hidden = apply(hidden_pre, sigmoid);

                auto output_pre = add(matmul(weights_ho, hidden), bias_o);
                auto output = apply(output_pre, sigmoid);

                // --- Error ---
                std::vector<double> output_errors(output_size);
                for (int k = 0; k < output_size; ++k) {
                    output_errors[k] = target[k] - output[k];
                    total_error += output_errors[k] * output_errors[k];
                }

                // --- Backpropagation ---
                // Output gradients
                std::vector<double> output_gradients(output_size);
                for (int k = 0; k < output_size; ++k) {
                    output_gradients[k] = output_errors[k] * sigmoid_derivative(output[k]);
                }

                // Hidden to output weights update
                for (int j = 0; j < hidden_size; ++j)
                    for (int k = 0; k < output_size; ++k)
                        weights_ho[j][k] += learning_rate * output_gradients[k] * hidden[j];

                for (int k = 0; k < output_size; ++k)
                    bias_o[k] += learning_rate * output_gradients[k];

                // Hidden errors
                std::vector<double> hidden_errors(hidden_size, 0.0);
                for (int j = 0; j < hidden_size; ++j)
                    for (int k = 0; k < output_size; ++k)
                        hidden_errors[j] += output_gradients[k] * weights_ho[j][k];

                // Hidden gradients
                std::vector<double> hidden_gradients(hidden_size);
                for (int j = 0; j < hidden_size; ++j)
                    hidden_gradients[j] = hidden_errors[j] * sigmoid_derivative(hidden[j]);

                // Input to hidden weights update
                for (int i = 0; i < input_size; ++i)
                    for (int j = 0; j < hidden_size; ++j)
                        weights_ih[i][j] += learning_rate * hidden_gradients[j] * input[i];

                for (int j = 0; j < hidden_size; ++j)
                    bias_h[j] += learning_rate * hidden_gradients[j];
            }

            if (epoch % 100 == 0 || epoch == epochs - 1) {
                std::cout << "Epoch " << epoch << " Error: " << total_error / X.size() << "\n";
            }
        }
    }
};

// Example usage
int main() {
    // XOR dataset
    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> y = {{0}, {1}, {1}, {0}};

    NeuralNetwork nn(2, 4, 1, 0.2);

    nn.train(X, y, 2000);

    std::cout << "\nPredictions:\n";
    for (const auto& input : X) {
        auto output = nn.predict(input);
        std::cout << "Input: [" << input[0] << ", " << input[1]
                  << "] Output: " << output[0] << "\n";
    }
    return 0;
}
