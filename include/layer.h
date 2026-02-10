#ifndef LAYER_H
#define LAYER_H

#include "common.h"
#include "gen_vector.h"
#include "matrix.h"


typedef void (*activation_fn)(const genVec* z, const genVec* a);
typedef void (*relu_deriv)(const genVec* z, const genVec* da_dz);
typedef void(*softmax_deriv) // softmax cross entropy deriv
    (const genVec* predicted, const genVec* true_label, const genVec* dL_dz);


typedef struct {

    genVec* input;      // pointer to input vector (prev layer) // x     (1 x m)
    genVec  bias;       // bias of each neuron  // b     (1 x n)
    Matrix  weight;     // weights // W     (m x n)
    genVec  preact;     // z     (1 x m)x(m x n)->(1 x n) z = xW + b
    genVec  act_output; // a     (1 x n) a = f(z)

    // Gradients

    // Gradients to update Weights and biases
    Matrix dL_dW; //  = dz/dW x dL/dz = (x)T x dL/dz  (m x 1) x (1 x n) -> (m x n)
    genVec dL_db; // = dL/dz x dz/db = dL/dz x 1  ->(1 x n)
    // Downstream Gradient
    genVec dL_dx; // (1 x m)  goes to the prev layer

    genVec dL_dz;
    // genVec da_dz;        // in hidden, not in output

    u16 input_size;  // m
    u16 output_size; // n

    activation_fn act_fn; // diff but same signature

    union {
        relu_deriv    d_relu;    // hidden layer
        softmax_deriv d_softmax; // output layer
    } act_deriv;

} Layer;


// Arena?
Layer* ffnn_layer_create(u16 input, u16 output);
void   ffnn_layer_destroy(Layer* layer);

void ffnn_init_weights_biases(Layer* layer); // diff for output
void ffnn_update_weights_biases(Layer* layer);

void ffnn_calc_output(Layer* layer, const genVec* input);

void ffnn_calc_deriv(Layer* layer, const genVec* upstream_grad); // diff for output

// output has get prediction function   - can be done from outside tho


#endif // LAYER_H
