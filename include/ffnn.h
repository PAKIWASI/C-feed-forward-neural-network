#ifndef FFNN_H
#define FFNN_H

#include "gen_vector.h"
#include "mnist_data_processor.h"



typedef struct {
    genVec layers;          // of type layer
    float  learning_rate;
    float* curr_img;        // normalized image, currently passed to forward pass
    mnist_dataset set;      // grey scale (0-255) u8 pixel images and u8 (0-9) labels
} ffnn;



ffnn* ffnn_create(Arena* arena, u16* layer_sizes, u8 num_layers,
                  float learning_rate, const char* mnist_path);

void ffnn_forward(ffnn* net);
void ffnn_backward(ffnn* net, u8 label);

void ffnn_train(ffnn* net);

#endif // FFNN_H
