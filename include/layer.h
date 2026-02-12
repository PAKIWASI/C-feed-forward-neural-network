#ifndef LAYER_H
#define LAYER_H

#include "common.h"
#include "matrix.h"


#ifndef LEARNING_RATE
    #define LEARNING_RATE 0.001f
#endif


typedef struct Layer {

    // Common data for all layers

    float* x; // input (1 x m) - Pointer to prev layer output - Needed for backprop
    float* b; // bias (1 x n) - Each neuron of current layer has a bias value
    Matrix W; // weights (m x n) - Maps prev layer neurons to curr layer with a weight (fully connnected)
    float* z; // pre-activation output (1 x n) - z = xW + b - (1 x n) = (1 x m) * (m x n) + (1 x n) - Linear Transformation
    float* a; // activated output (1 x n) - a = f(z) (element wise activation)  - Non Linearity (Learn complex patterns)

    // Gradients for Backpropogation

    Matrix dL_dW; // (m x n) - How much current layer's weights effect Loss. (How to change weights to reduce loss)
    // float* dL_db; // (1 x n) - This comes out as equal to dL_dz
    float* dL_dz; // (1 x n)
    float* dL_dx; // (1 x m) - Passed to previous layer - Change in Loss wrt change in input (output of prev layer)

    // Dimensions

    u16 m;  // Input size - Size of previous layer
    u16 n;  // Ouput size - Size of current layer 

    // Layer-specific behavior

    b8 is_output_layer;     // flag determines which act/act_deriv function to use

} Layer;

/*
>>> 136 + 16 * 4 + 786 * 16 * 4 + 16 * 4 + 16 * 4 + 786 * 16 * 4 + 16 * 4 + 16 * 4 + 786 * 4 + 16 * 4
104272
>>> 104272 / (1024)
101.828125 KB
*/


// Factory functions
Layer* layer_create_hidden(Arena* arena, u16 m, u16 n);
Layer* layer_create_output(Arena* arena, u16 m, u16 n);

// Common operations (work for all layer types)
// void layer_destroy(Layer* layer);   // NOT NEEDED (use Arena)
void layer_init_weights_biases(Layer* layer);
void layer_forward(Layer* layer, const float* x);
void layer_backward(Layer* layer, const float* dL_da);  // upstream gradient
void layer_update_weights(Layer* layer);

// Activation functions (implementation details)
void relu_activate(const float* z, float* a, u16 size);
void softmax_activate(const float* z, float* a, u16 size);

// Backprop functions (implementation details)
void relu_deriv(const float* z, const float* dL_da, float* dL_dz, u16 size);
void softmax_crossentropy_deriv(const float* predicted, 
        const float* true_label, float* dL_dz, u16 size);  // y  -> true labels


#endif // LAYER_H
