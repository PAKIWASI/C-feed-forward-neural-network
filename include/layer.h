#ifndef LAYER_H
#define LAYER_H

#include "common.h"
#include "matrix.h"



typedef struct Layer {

    // Common data for all layers

    float* x; // input (1 x m) - Pointer to prev layer output - Needed for backprop
    float* b; // bias (1 x n) - Each neuron of current layer has a bias value
    Matrixf W; // weights (n x m) - Maps prev layer neurons to curr layer with a weight (fully connnected)
    float* z; // pre-activation output (1 x n) - z = xW + b - (1 x n) = (1 x m) * (m x n) + (1 x n) - Linear Transformation
    float* a; // activated output (1 x n) - a = f(z) (element wise activation)  - Non Linearity (Learn complex patterns)

    // Gradients for Backpropogation

    Matrixf dL_dW; // (n x m) - How much current layer's weights effect Loss. (How to change weights to reduce loss)
    // float* dL_db; // (1 x n) - This comes out as equal to dL_dz
    float* dL_dz; // (1 x n)
    float* dL_dx; // (1 x m) - Passed to previous layer - Change in Loss wrt change in input (output of prev layer)

    // Dimensions

    u16 m;  // Input size - Size of previous layer
    u16 n;  // Ouput size - Size of current layer 

    // Layer-specific behavior

    b8 is_output_layer;     // flag determines which act/act_deriv function to use

    Matrixf W_T; // transpose of W matrix (needed for backprop)

} Layer;


// Factory functions
Layer* layer_create_hidden(Arena* arena, u16 m, u16 n);
Layer* layer_create_output(Arena* arena, u16 m, u16 n);

void layer_init_weights_biases(Layer* layer);
void layer_update_WB(Layer* layer, float learning_rate);

void layer_calc_output(Layer* layer, const float* x);
void layer_calc_deriv(Layer* layer, const float* dL_da);


#endif // LAYER_H
