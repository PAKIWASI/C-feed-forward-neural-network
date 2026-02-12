#ifndef FFNN_H
#define FFNN_H

#include "gen_vector.h"
#include "layer.h"
#include "mnist_data_processor.h"



typedef struct {
    genVec layers; // of type layer
    float  learning_rate;
    u16    input_size;
    u16    output_size;
    u8     num_layers;
    mnist_dataset set;
} Neural_Network;





void ffnn_forward(Layer* layer, const float* x);
void ffnn_backward(Layer* layer, const float* dL_da);  // upstream gradient

#endif // FFNN_H
