#ifndef FFNN_H
#define FFNN_H

#include "gen_vector.h"
#include "mnist_data_processor.h"



typedef struct {
    genVec layers;          // of type layer
    float  learning_rate;   // step size in gradient decent
    float* curr_img;        // normalized image, currently passed to forward pass
    Arena* main_arena;      // arena to allocate layers, weights, biases
    Arena* dataset_arena;   // arena for the training/testing datasets
    mnist_dataset set;      // grey scale (0-255) u8 pixel images and u8 (0-9) labels (arena allocated)
} ffnn;



ffnn* ffnn_create(u16* layer_sizes, u8 num_layers,
                  float learning_rate, const char* mnist_path);

void ffnn_destroy(ffnn* net);


void ffnn_train(ffnn* net);

void ffnn_train_batch(ffnn* net, u16 batch_size, u16 num_epochs);

void ffnn_test(ffnn* net);

void ffnn_change_dataset(ffnn* net, const char* dataset_path);
b8 ffnn_save_parameters(const ffnn* net, const char* outfile);
b8 ffnn_load_parameters(const ffnn* net, const char* filepath);


#endif // FFNN_H
