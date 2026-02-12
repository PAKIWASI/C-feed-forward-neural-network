#ifndef FFNN_H
#define FFNN_H

#include "gen_vector.h"
#include "layer.h"
#include "mnist_data_processor.h"



typedef struct {
    genVec layers;          // of type layer
    float  learning_rate;
    u16    input_size;      // 786 for mnist
    u16    output_size;     // 10 for mnist
    // u8     num_layers;   // info in genVec
    mnist_dataset set;
} ffnn;


ffnn* ffnn_create(Arena* arena, u16 input_size, u16 output_size, u16* layer_sizes, u8 num_layers, float learning_rate, const char* mnist_path);



void ffnn_forward(Layer* layer, const float* x);
void ffnn_backward(Layer* layer, const float* dL_da);  // upstream gradient

#endif // FFNN_H
