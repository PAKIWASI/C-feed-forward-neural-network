#include "ffnn.h"
#include "arena.h"
#include "common.h"
#include "gen_vector.h"
#include "layer.h"
#include "mnist_data_processor.h"


#define GET_LABEL(net, i) (*((net)->set.data + i))
#define GET_IMG(net, i) ((net)->set.data + i + 1)
#define GET_LAYER(net, i) ((Layer*)genVec_get_ptr(&(net)->layers, i))
#define GET_PREDICTION_ARR(net) (GET_LAYER(net, genVec_size(&net->layers) - 1)->a)


void normalize_mnist_img(u8* img, float* normalized_img);
u8 get_prediction(float* prediction_arr);


ffnn* ffnn_create(Arena* arena, u16* layer_sizes, u8 num_layers,
                  float learning_rate, const char* mnist_path)
{
    ffnn* net = ARENA_ALLOC(arena, ffnn);

    net->learning_rate = learning_rate;
    net->curr_img = ARENA_ALLOC_N(arena, float, (u64)MNIST_IMG_SIZE);

    mnist_load_custom_file(&net->set, mnist_path, arena);

    // memory for layers alloced on arena so no copy/move semantics
    genVec_init_stk(num_layers, sizeof(Layer*), NULL, NULL, NULL, &net->layers);

    Layer* l; 
    for (u8 i = 0; i < num_layers - 1; i++) {
        if (i == num_layers - 2) {
            l = layer_create_output(arena, layer_sizes[i], layer_sizes[i + 1]);
        } else {
            l = layer_create_hidden(arena, layer_sizes[i], layer_sizes[i + 1]);
        }

        layer_init_weights_biases(l);
        genVec_push(&net->layers, castptr(l));
    }

    return net;
}

void ffnn_forward(ffnn* net)
{
    u64 size = genVec_size(&net->layers);
    for (u64 i = 0; i < size; i++) {
        if (i == 0) {
            // input to fist hidden layer is img
            layer_calc_output(GET_LAYER(net, i), net->curr_img);
        } else {
            // output of prev layer becomes input to next
            layer_calc_output(GET_LAYER(net, i), GET_LAYER(net, i - 1)->a);
        }
    }
}

void ffnn_backward(ffnn* net, u8 label)
{
    // create the true label array (true label 1, else 0)
    float true_label[10] = {0};
    true_label[label] = 1;

    u64 size = genVec_size(&net->layers);
    for (u64 i = size - 1; i >= 0; i--) {
        if (i == size - 1) {
            layer_calc_deriv(GET_LAYER(net, i), (float*)&true_label);
        } else {
            layer_calc_deriv(GET_LAYER(net, i), GET_LAYER(net, i + 1)->dL_dx);
        }
    }
}


void ffnn_train(ffnn* net)
{

    u16 correct = 0;
    for (u16 i = 0; i < MNIST_TRAIN_SIZE; i++) {

        u8 label = GET_LABEL(net, i);

        // normalize each training example
        normalize_mnist_img(GET_IMG(net, i), net->curr_img);

        // do forward pass
        ffnn_forward(net);

        // now 'a' of output layer has the prediction
        u8 prediction = get_prediction(GET_PREDICTION_ARR(net));
        if (prediction == label) {
            correct++;
        }

        // we do backprop, using true and predicted arrays
        // Loss calculated internally in output layer
        ffnn_backward(net, label);

        // update_parameters(net, learning_rate);
    }
}




void normalize_mnist_img(u8* img, float* normalized_img)
{
    for (u16 i = 0; i < MNIST_IMG_SIZE; i++) {
        normalized_img[i] = (float)img[i] / 255.0f;
    }
}

u8 get_prediction(float* prediction_arr)
{
    float max = -9999;
    u8 prediction = 10;

    for (u8 i = 0; i < 10; i++) {
        if (prediction_arr[i] > max) {
            max = prediction_arr[i];
            prediction = i;
        }
    }

    return prediction;
}

