#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>

// Neural Network class
class NeuralNetwork {
private:
    int input_size;
    int hidden_size;
    int output_size;
    std::vector<std::vector<double>> weights_ih; // Input to hidden weights
    std::vector<std::vector<double>> weights_ho; // Hidden to output weights
    std::vector<double> bias_h; // Hidden layer bias
    std::vector<double> bias_o; // Output layer bias
    double learning_rate;

    // Sigmoid activation function
    double sigmoid(double x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Derivative of sigmoid
    double sigmoid_derivative(double x) const {
        double sig = sigmoid(x);
        return sig * (1.0 - sig);
    }

    // Initialize weights and biases randomly
    void initialize_weights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 1.0);

        // Initialize input to hidden weights
        weights_ih.resize(input_size, std::vector<double>(hidden_size));
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                weights_ih[i][j] = dist(gen);
            }
        }

        // Initialize hidden to output weights
        weights_ho.resize(hidden_size, std::vector<double>(output_size));
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                weights_ho[i][j] = dist(gen);
            }
        }

        // Initialize biases
        bias_h.resize(hidden_size);
        bias_o.resize(output_size);
        for (int i = 0; i < hidden_size; ++i) {
            bias_h[i] = dist(gen);
        }
        for (int i = 0; i < output_size; ++i) {
            bias_o[i] = dist(gen);
        }
    }

    // Matrix-vector multiplication
    std::vector<double> matmul(const std::vector<std::vector<double>>& matrix,
                              const std::vector<double>& vector) const {
        std::vector<double> result(matrix[0].size(), 0.0);
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            for (size_t i = 0; i < matrix.size(); ++i) {
                result[j] += matrix[i][j] * vector[i];
            }
        }
        return result;
    }

    // Vector addition
    std::vector<double> add_bias(const std::vector<double>& vec,
                                const std::vector<double>& bias) const {
        std::vector<double> result(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            result[i] = vec[i] + bias[i];
        }
        return result;
    }

    // Apply activation function to vector
    std::vector<double> apply_sigmoid(const std::vector<double>& vec) const {
        std::vector<double> result(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            result[i] = sigmoid(vec[i]);
        }
        return result;
    }

public:
    NeuralNetwork(int input, int hidden, int output, double lr = 0.1)
        : input_size(input), hidden_size(hidden), output_size(output), learning_rate(lr) {
        initialize_weights();
    }

    // Forward propagation
    std::vector<double> forward(const std::vector<double>& inputs) const {
        // Input to hidden layer
        std::vector<double> hidden = matmul(weights_ih, inputs);
        hidden = add_bias(hidden, bias_h);
        hidden = apply_sigmoid(hidden);

        // Hidden to output layer
        std::vector<double> output = matmul(weights_ho, hidden);
        output = add_bias(output, bias_o);
        output = apply_sigmoid(output);

        return output;
    }

    // Train the network using backpropagation
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<std::vector<double>>& y,
               int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;
            for (size_t i = 0; i < X.size(); ++i) {
                // Forward pass
                std::vector<double> hidden = matmul(weights_ih, X[i]);
                hidden = add_bias(hidden, bias_h);
                std::vector<double> hidden_activations = apply_sigmoid(hidden);

                std::vector<double> output = matmul(weights_ho, hidden_activations);
                output = add_bias(output, bias_o);
                std::vector<double> output_activations = apply_sigmoid(output);

                // Calculate error (mean squared error)
                std::vector<double> errors(output_size);
                for (int j = 0; j < output_size; ++j) {
                    errors[j] = y[i][j] - output_activations[j];
                    total_error += errors[j] * errors[j];
                }

                // Backpropagation
                // Output layer gradients
                std::vector<double> output_gradients(output_size);
                for (int j = 0; j < output_size; ++j) {
                    output_gradients[j] = errors[j] * sigmoid_derivative(output_activations[j]);
                }

                // Update hidden to output weights and biases
                for (int j = 0; j < hidden_size; ++j) {
                    for (int k = 0; k < output_size; ++k) {
                        double delta = learning_rate * output_gradients[k] * hidden_activations[j];
                        weights_ho[j][k] += delta;
                    }
                }
                for (int k = 0; k < output_size; ++k) {
                    bias_o[k] += learning_rate * output_gradients[k];
                }

                // Hidden layer gradients
                std::vector<double> hidden_gradients(hidden_size, 0.0);
                for (int j = 0; j < hidden_size; ++j) {
                    for (int k = 0; k < output_size; ++k) {
                        hidden_gradients[j] += output_gradients[k] * weights_ho[j][k];
                    }
                    hidden_gradients[j] *= sigmoid_derivative(hidden_activations[j]);
                }

                // Update input to hidden weights and biases
                for (int j = 0; j < input_size; ++j) {
                    for (int k = 0; k < hidden_size; ++k) {
                        double delta = learning_rate * hidden_gradients[k] * X[i][j];
                        weights_ih[j][k] += delta;
                    }
                }
                for (int k = 0; k < hidden_size; ++k) {
                    bias_h[k] += learning_rate * hidden_gradients[k];
                }
            }
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << " Error: " << total_error / X.size() << "\n";
            }
        }
    }

    // Predict for new inputs
    std::vector<double> predict(const std::vector<double>& inputs) const {
        return forward(inputs);
    }
};

// Example usage
int main() {
    // Example: XOR problem
    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> y = {{0}, {1}, {1}, {0}};

    // Create neural network: 2 inputs, 4 hidden nodes, 1 output
    NeuralNetwork nn(2, 4, 1, 0.1);

    // Train the network
    nn.train(X, y, 1000);

    // Test predictions
    std::cout << "\nPredictions:\n";
    for (const auto& input : X) {
        auto output = nn.predict(input);
        std::cout << "Input: [" << input[0] << ", " << input[1]
                  << "] Output: " << output[0] << "\n";
    }

    return 0;
}
