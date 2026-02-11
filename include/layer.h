#ifndef LAYER_H
#define LAYER_H

#include "common.h"
#include "matrix.h"


typedef struct Layer {

    // Common data for all layers
    float* x; // input (1 x m) - Pointer to prev layer output - Needed for backprop
    float* b; // bias (1 x n) - Each neuron of current layer has a bias value
    Matrix W; // weights (m x n) - Maps prev layer neurons to curr layer with a weight (fully connnected)
    float* z; // pre-activation output (1 x n) - z = xW + b - (1 x n) = (1 x m) * (m x n) + (1 x n) - Linear Transformation
    float* a; // activated output (1 x n) - a = f(z) (element wise activation)  - Non Linearity (Learn complex patterns)

    // Gradients for Backpropogation
    Matrix dL_dW; // (m x n) - How much current layer's weights effect Loss. (How to change weights to reduce loss)
    float* dL_db; // (1 x n)
    float* dL_dz; // (1 x n)
    float* dL_dx; // (1 x m) - Passed to previous layer - Change in Loss wrt change in input (output of prev layer)

    // Dimensions
    u16 m;  // Input size - Size of previous layer
    u16 n;  // Ouput size - Size of current layer 

    // Layer-specific behavior
    b8 is_output_layer;     // flag determines which act/act_deriv function to use, use da_dz or not
    void (*activation_fn)(const float* z, float* a, u16 size);
    void (*activation_deriv_fn)(struct Layer* layer, const float* upstream_grad);

    // Type-specific data (only used by hidden layers)
    float* da_dz; // derivative cache for ReLU (1 x n)

} Layer;


// Factory functions
Layer* create_hidden_layer(u16 m, u16 n);
Layer* create_output_layer(u16 m, u16 n);

// Common operations (work for all layer types)
void layer_destroy(Layer* layer);
void layer_init_weights_biases(Layer* layer);
void layer_forward(Layer* layer, const float* x);
void layer_backward(Layer* layer, const float* dL_da);  // upstream gradient
void layer_update_weights(Layer* layer, float learning_rate);

// Activation functions (implementation details)
void relu_activate(const float* z, float* a, u16 size);
void softmax_activate(const float* z, float* a, u16 size);

// Backprop functions (implementation details)
void relu_deriv(Layer* layer, const float* dL_da);  // TODO: uses da_dz
void softmax_crossentropy_deriv(Layer* layer, const float* y);  // y  -> true labels


#endif // LAYER_H

/*
For a feedforward neural network specifically:
    You need polymorphism (treating all layers uniformly)
    Shared behavior dominates (90% of code is the same)
    You want simple arrays of layers

So the "inheritance-style" pattern (tagged union + function pointers) is the right tool. Pure composition forces you to either:
    Give up polymorphism (painful for network code)
    Manually re-implement polymorphism with void* and casting (worse than the inheritance pattern)
*/
/*
// Create network
Layer* hidden1 = create_hidden_layer(784, 128);
Layer* hidden2 = create_hidden_layer(128, 64);
Layer* output = create_output_layer(64, 10);

// Forward pass
layer_forward(hidden1, input_data);
layer_forward(hidden2, hidden1->act_output);
layer_forward(output, hidden2->act_output);

// Backward pass (output layer uses true labels, hidden layers use upstream gradient)
layer_backward(output, true_labels);  // Special: takes labels
layer_backward(hidden2, output->dL_dx);
layer_backward(hidden1, hidden2->dL_dx);

// Update weights
float lr = 0.01f;
layer_update_weights(output, lr);
layer_update_weights(hidden2, lr);
layer_update_weights(hidden1, lr);

// Cleanup
layer_destroy(hidden1);
layer_destroy(hidden2);
layer_destroy(output);
*/
