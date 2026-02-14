#include "ffnn.h"
#include "arena.h"
#include "common.h"
#include "gen_vector.h"
#include "layer.h"
#include "matrix.h"
#include "random.h"


#define GET_LABEL(net, i) (*((net)->set.data + i))
#define GET_IMG(net, i) ((net)->set.data + i + 1)
#define GET_LAYER(net, i) ((Layer*)genVec_get_ptr(&(net)->layers, i))
#define GET_PREDICTION_ARR(net) (GET_LAYER(net, genVec_size(&net->layers) - 1)->a)


void ffnn_forward(ffnn* net);
void ffnn_backward(ffnn* net, u8 label);
void ffnn_update_parameters(ffnn* net);

void normalize_mnist_img(u8* img, float* normalized_img);
u8 get_prediction(float* prediction_arr);
void shuffle_indices(u16* indices, u16 size);


ffnn* ffnn_create(u16* layer_sizes, u8 num_layers,
                  float learning_rate, const char* mnist_path)
{
    LOG("creating ffnn");
    LOG("learing_rate set to %f", learning_rate);
    LOG("no of layers: %u", num_layers);
    LOG("loading file: %s", mnist_path);

    ffnn* net = malloc(sizeof(ffnn));

    net->main_arena = arena_create(nMB(1));
    LOG("main arena allocted with %lu", arena_remaining(net->main_arena));
    net->dataset_arena = arena_create(MNIST_TRAIN_SIZE * (MNIST_IMG_SIZE + MNIST_LABEL_SIZE));
    LOG("dataset arena allocted with %lu", arena_remaining(net->dataset_arena));

    net->learning_rate = learning_rate;

    net->curr_img = ARENA_ALLOC_N(net->main_arena, float, MNIST_IMG_SIZE);

    mnist_load_custom_file(&net->set, mnist_path, net->dataset_arena);

    // memory for layers alloced on arena so no copy/move semantics
    genVec_init_stk(num_layers, sizeof(Layer*), NULL, NULL, NULL, &net->layers);

    Layer* l; 
    for (u8 i = 0; i < num_layers - 1; i++) {
        LOG("layer %u input size: %u, output size %u", i, layer_sizes[i], layer_sizes[i + 1]);

        if (i == num_layers - 2) {
            l = layer_create_output(net->main_arena, layer_sizes[i], layer_sizes[i + 1]);
        } else {
            l = layer_create_hidden(net->main_arena, layer_sizes[i], layer_sizes[i + 1]);
        }

        layer_init_weights_biases(l);
        genVec_push(&net->layers, castptr(l));
    }

    return net;
}


void ffnn_destroy(ffnn* net)
{
    genVec_destroy_stk(&net->layers);
    arena_release(net->main_arena);
    arena_release(net->dataset_arena);
    free(net);
}



void ffnn_train(ffnn* net)
{
    u16 correct = 0;

    for (u16 i = 0; i < net->set.num_imgs; i++) { 

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

        // now we update all W & b using gradients
        ffnn_update_parameters(net);
        
        // Progress indicator
        if ((i + 1) % 5000 == 0) {
            LOG("Processed %u/%u samples\n", i + 1, net->set.num_imgs);
        }
    }
    
    // Print accuracy
    float accuracy = (float)correct / (float)net->set.num_imgs * 100.0f;
    LOG("Training Accuracy: %.2f%% (%u/%u)\n", 
           accuracy, correct, net->set.num_imgs);
}


void ffnn_train_batch(ffnn* net, u16 batch_size, u16 num_epochs)
{

}


b8 ffnn_save_parameters(const ffnn* net, const char* outfile)
{
    FILE* f = fopen(outfile, "wb");
    if (!f) {
        LOG("couldn't open parameters file to write");
        return false;
    }

    // num of layers
    u64 size = genVec_size(&net->layers);

    fwrite(&size, sizeof(u64), 1, f);

    for (u64 i = 0; i < size; i++) {
        // write each layer size
        // input size
        u16 m = GET_LAYER(net, i)->m;
        fwrite(&m, sizeof(u16), 1, f);
        // output size
        u16 n = GET_LAYER(net, i)->n;
        fwrite(&n, sizeof(u16), 1, f);

        // write weights
        fwrite(GET_LAYER(net, i)->W.data, sizeof(float), (u64)n * m, f);
        // write biases
        fwrite(GET_LAYER(net, i)->b, sizeof(float), n, f);
    }

    fclose(f);
    return true;
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
    
    for (u64 i = 0; i < size; i++) {
        u64 idx = size - 1 - i;  // Calculate actual index
        
        if (idx == size - 1) {
            layer_calc_deriv(GET_LAYER(net, idx), (float*)&true_label);
        } else {
            layer_calc_deriv(GET_LAYER(net, idx), GET_LAYER(net, idx + 1)->dL_dx);
        }
    }
}


void ffnn_update_parameters(ffnn* net)
{
    u64 size = genVec_size(&net->layers);
    for (u64 i = 0; i < size; i++) {
        layer_update_WB(GET_LAYER(net, i), net->learning_rate);
    }
}


void normalize_mnist_img(u8* img, float* normalized_img)
{
    for (u64 i = 0; i < MNIST_IMG_SIZE; i++) {
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
// Fisher-Yates shuffle algorithm
void shuffle_indices(u16* indices, u16 size)
{
    for (u16 i = size - 1; i > 0; i--) {
        // Generate random index from 0 to i (inclusive)
        u16 j = (u16)pcg32_rand_bounded(i + 1);
        
        // Swap indices[i] and indices[j]
        indices[i] ^= indices[j];
        indices[j] ^= indices[i];
        indices[i] ^= indices[j];
    }
}



