#include "ffnn.h"
#include "arena.h"
#include "common.h"
#include "layer.h"
#include "random.h"



// Calculate the byte offset for each sample (label + image = 785 bytes)
#define SAMPLE_OFFSET(i) ((i) * (MNIST_IMG_SIZE + MNIST_LABEL_SIZE))
// Get label at the start of each sample
#define GET_LABEL(net, i) (*((net)->set.data + SAMPLE_OFFSET(i)))
// Get image data (starts 1 byte after label)
#define GET_IMG(net, i) ((net)->set.data + SAMPLE_OFFSET(i) + MNIST_LABEL_SIZE)

#define GET_LAYER(net, i)       (*(Layer**)genVec_get_ptr(&(net)->layers, i))
#define GET_PREDICTION_ARR(net) (GET_LAYER(net, genVec_size(&net->layers) - 1)->a)



void ffnn_forward(ffnn* net, float* norm_img);
void ffnn_backward(ffnn* net, u8 label);
void ffnn_update_parameters(ffnn* net);

void normalize_mnist_img(u8* img, float* normalized_img);
u8   get_prediction(float* prediction_arr);
void shuffle_indices(u16* indices, u16 size);
u64 calc_layer_size(u16 m, u16 n);



// accumilator gradients for batch training
typedef struct {
    Matrixf dL_dW; // (n x m) - How much current layer's weights effect Loss. (How to change weights to reduce loss)
    float*  dL_dz; // (1 x n) - dL_dz = dL_db
} batch_gradients;


void batch_gradients_init(Arena* arena, batch_gradients* bg, Layer* l);
void batch_gradients_clear(batch_gradients* bg);



ffnn* ffnn_create(u16* layer_sizes, u8 num_layers, float learning_rate, const char* mnist_path)
{
    CHECK_FATAL(!layer_sizes, "layer sizes are null");
    CHECK_FATAL(!mnist_path, "mnist_path is null");
    CHECK_FATAL(num_layers == 0, "can't have 0 layers");
    CHECK_FATAL(learning_rate == 0, "can't have 0 learning_rate");
    LOG("creating ffnn");
    LOG("learing_rate set to %f", learning_rate);
    LOG("no of layers: %u", num_layers);

    ffnn* net = malloc(sizeof(ffnn));

    // allocate dataset_arena (for mnist)       // 60k * (784 + 1)
    net->dataset_arena = arena_create(nKB(1) + (MNIST_TRAIN_SIZE * (MNIST_IMG_SIZE + MNIST_LABEL_SIZE)));
    LOG("dataset arena allocted with %lu", arena_remaining(net->dataset_arena));

    net->learning_rate = learning_rate;

    mnist_load_custom_file(&net->set, mnist_path, net->dataset_arena);

    // memory for layers alloced on arena so no copy/move semantics
    genVec_init_stk(num_layers, sizeof(Layer*), NULL, NULL, NULL, &net->layers);

    // calculate memory needed for main arena (for layer data)
    u64 main_arena_size = nKB(1);   // for safety
    // accumilate size according to layer input,output sizes
    for (u8 i = 0; i < num_layers - 1; i++) {
        u16 m = layer_sizes[i];
        u16 n = layer_sizes[i + 1];
        main_arena_size += calc_layer_size(m, n);
    }
    net->main_arena = arena_create(main_arena_size);
    LOG("main arena allocted with %lu", arena_remaining(net->main_arena));  // 851584


    Layer* l;
    for (u8 i = 0; i < num_layers - 1; i++) {
        u16 m = layer_sizes[i];
        u16 n = layer_sizes[i + 1];
        LOG("layer %u input size: %u, output size %u", i, m, n);

        if (i == num_layers - 2) {
            l = layer_create_output(net->main_arena, m, n);
        } else {
            l = layer_create_hidden(net->main_arena, m, n);
        }

        layer_init_weights_biases(l);
        genVec_push(&net->layers, (u8*)&l);
    }


    LOG("created ffnn successfully\nStructure:");

    for (u64 i = 0; i < num_layers - 1; i++) {
        printf("\t%hu", GET_LAYER(net, i)->m);
        if (i == num_layers - 2) {
            printf("\t%hu", GET_LAYER(net, i)->n);
        }
    }
    putchar('\n');
    return net;
}


ffnn* ffnn_create_trained(const char* saved_path)
{
    CHECK_FATAL(!saved_path, "saved_path is null");
    LOG("creating ffnn on pre trained parameters");

    // open the parameters file
    FILE* f = fopen(saved_path, "rb");
    CHECK_WARN_RET(!f, NULL, "couldn't open parameters file %s", saved_path);

    LOG("opened parameters file %s", saved_path);

    ffnn* net = malloc(sizeof(ffnn));

    net->dataset_arena = NULL;

    // read num of layer from file
    u64 num_layers;
    fread(&num_layers, sizeof(u64), 1, f);
    LOG("number of layers: %lu", num_layers);

    // create layer vec
    genVec_init_stk(num_layers, sizeof(Layer*), NULL, NULL, NULL, &net->layers);

    // calcuate memory needed for all layers and their data
    u64 main_arena_size = nKB(1);

    // read each layer's input/output sizes
    for (u64 i = 0; i < num_layers; i++) {
        u16 m, n;
        fread(&m, sizeof(u16), 1, f);
        fread(&n, sizeof(u16), 1, f);
        main_arena_size += calc_layer_size(m, n);
        fseek(f, (long)(((u64)n * m + n) * sizeof(float)), SEEK_CUR);   // skip over weights and biases
    }
    // reset back 
    fseek(f, sizeof(u64), SEEK_SET);  // skip num_layers header

    // allocate
    net->main_arena = arena_create(main_arena_size);
    LOG("main arena allocted with %lu", arena_remaining(net->main_arena));


    Layer* l;
    for (u64 i = 0; i < num_layers; i++) {
        u16 m, n;
        // read input size of each layer
        fread(&m, sizeof(u16), 1, f);
        // read output size of each layer
        fread(&n, sizeof(u16), 1, f);

        LOG("layer %lu input size: %u, output size %u", i, m, n);

        if (i == num_layers - 1) {
            l = layer_create_output(net->main_arena, m, n);
        } else {
            l = layer_create_hidden(net->main_arena, m, n);
        }

        // read the weights and biases
        // write weights
        fread(l->W.data, sizeof(float), (u64)n * m, f);
        // write biases
        fread(l->b, sizeof(float), n, f);

        genVec_push(&net->layers, (u8*)&l);
    }


    LOG("created ffnn successfully\nStructure:");

    for (u64 i = 0; i < num_layers; i++) {
        printf("\t%hu", GET_LAYER(net, i)->m);
        if (i == num_layers - 1) {
            printf("\t%hu", GET_LAYER(net, i)->n);
        }
    }
    putchar('\n');

    fclose(f);
    return net;
}


void ffnn_destroy(ffnn* net)
{
    CHECK_FATAL(!net, "net is null");
    genVec_destroy_stk(&net->layers);
    arena_release(net->main_arena);
    if (net->dataset_arena) {
        arena_release(net->dataset_arena);
    }
    free(net);
}

// void ffnn_change_dataset(ffnn* net, const char* dataset_path)
// {
//     arena_clear(net->dataset_arena);
//
//     mnist_load_custom_file(&net->set, dataset_path, net->dataset_arena);
// }


void ffnn_set_dataset(ffnn* net, const char* dataset_path)
{
    if (net->dataset_arena != NULL) {
        arena_release(net->dataset_arena);
    }

    FILE* f = fopen(dataset_path, "rb"); 
    u16 num_imgs;
    u8 img_w, img_h;
    fread(&num_imgs, sizeof(u16), 1, f);
    fread(&img_w, sizeof(u8), 1, f);
    fread(&img_h, sizeof(u8), 1, f);
    net->dataset_arena = arena_create(nKB(1) + ((u64)num_imgs * ((img_w*img_h) + 1)));
    LOG("dataset arena allocted with %lu", arena_remaining(net->dataset_arena));
    fclose(f);

    mnist_load_custom_file(&net->set, dataset_path, net->dataset_arena);
}



void ffnn_train(ffnn* net)
{
    LOG("begin training ffnn with %u samples", net->set.num_imgs);

    u16 correct = 0;

    // allocate for one normalized (type float) mnist image as buffer for all
    arena_scratch sc       = arena_scratch_begin(net->main_arena);
    float*        norm_img = ARENA_ALLOC_N(net->main_arena, float, MNIST_IMG_SIZE);

    for (u16 i = 0; i < net->set.num_imgs; i++) {

        u8 label = GET_LABEL(net, i);

        // normalize each training example
        normalize_mnist_img(GET_IMG(net, i), norm_img);

        // do forward pass
        ffnn_forward(net, norm_img);

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
            LOG("\n\tProcessed %u/%u samples\n", i + 1, net->set.num_imgs);
        }

        // if ((i + 1) % LEARN_DECAY_AFTER == 0) {
        //     net->learning_rate *= LEARN_DECAY_RATE;
        //     printf("learning rate %f\n", net->learning_rate);
        // }
    }

    // Print accuracy
    float accuracy = (float)correct / (float)net->set.num_imgs * 100.0f;
    LOG("Training Accuracy: %.2f%% (%u/%u)\n", accuracy, correct, net->set.num_imgs);

    arena_scratch_end(&sc);
}


void ffnn_test(ffnn* net)
{
    LOG("Starting testing on %u samples", net->set.num_imgs);

    u16 correct = 0;

    arena_scratch sc       = arena_scratch_begin(net->main_arena);
    float*        norm_img = ARENA_ALLOC_N(net->main_arena, float, MNIST_IMG_SIZE);


    for (u16 i = 0; i < net->set.num_imgs; i++) {
        u8 label = GET_LABEL(net, i);
        normalize_mnist_img(GET_IMG(net, i), norm_img);

        // Forward pass only (no backprop)
        ffnn_forward(net, norm_img);

        u8 prediction = get_prediction(GET_PREDICTION_ARR(net));
        if (prediction == label) {
            correct++;
        }

        // Progress indicator
        if ((i + 1) % 2000 == 0) {
            printf("  Tested %u/%u samples\n", i + 1, net->set.num_imgs);
        }
    }

    float accuracy = (float)correct / (float)net->set.num_imgs * 100.0f;
    printf("Test Accuracy: %.2f%% (%u/%u)\n", accuracy, correct, net->set.num_imgs);

    arena_scratch_end(&sc);
}


/*
For each epoch:
    Shuffle training data
    
    For each batch:
        Clear gradient accumulators (set to zero)
        
        For each example in batch:
            Forward pass
            Backward pass (compute gradients)
            Add gradients to accumulators
        
        Average accumulated gradients by batch size
        Update weights using averaged gradients
*/
void ffnn_train_batch(ffnn* net, u16 batch_size, u16 num_epochs)
{
    CHECK_FATAL(batch_size == 0, "batch size can't be 0");
    CHECK_FATAL(batch_size > net->set.num_imgs, "batch size larger than dataset");
    CHECK_FATAL(num_epochs == 0, "num_epochs can't be 0");

    const u16 num_batches = net->set.num_imgs / batch_size;
    const u16 remainder   = net->set.num_imgs % batch_size;

    LOG("Started batch training with multiple epochs:");
    LOG("  Total samples: %u", net->set.num_imgs);
    LOG("  Batch size: %u", batch_size);
    LOG("  Batches per epoch: %u", num_batches);
    LOG("  Remainder samples: %u", remainder);
    LOG("  Number of epochs: %u", num_epochs);

    // Allocate gradient accumulators
    arena_scratch sc = arena_scratch_begin(net->main_arena);

    genVec gradients;
    genVec_init_stk(net->layers.size, sizeof(batch_gradients), NULL, NULL, NULL, &gradients);

    for (u64 i = 0; i < net->layers.size; i++) {
        batch_gradients bg;
        batch_gradients_init(net->main_arena, &bg, GET_LAYER(net, i));
        genVec_push(&gradients, (u8*)&bg);
    }

    genVec indices;
    genVec_init_stk(net->set.num_imgs, sizeof(u16), NULL, NULL, NULL, &indices);
    for (u16 i = 0; i < net->set.num_imgs; i++) {
        u16 idx = i;
        genVec_push(&indices, (u8*)&idx);
    }

    float* norm_img      = ARENA_ALLOC_N(net->main_arena, float, MNIST_IMG_SIZE);
    float  best_accuracy = 0.0f;

    // EPOCH LOOP
    for (u16 epoch = 0; epoch < num_epochs; epoch++) {
        LOG("\n     Epoch %u/%u", epoch + 1, num_epochs);

        shuffle_indices((u16*)genVec_front(&indices), (u16)genVec_size(&indices));

        u16 epoch_correct = 0;

        // BATCH LOOP - Process all complete batches
        for (u16 batch = 0; batch < num_batches; batch++) {
            u16 current_batch_size = batch_size;
            u16 batch_start_idx    = batch * batch_size;

            // If this is the last batch and there's a remainder, include it
            if (batch == num_batches - 1 && remainder > 0) {
                current_batch_size += remainder;
            }

            // Clear gradient accumulators
            for (u64 layer_idx = 0; layer_idx < net->layers.size; layer_idx++) {
                batch_gradients* bg = (batch_gradients*)genVec_get_ptr(&gradients, layer_idx);
                batch_gradients_clear(bg);
            }

            // Process samples in current batch
            for (u16 sample_in_batch = 0; sample_in_batch < current_batch_size; sample_in_batch++) {
                u16 dataset_idx = *(u16*)genVec_get_ptr(&indices, batch_start_idx + sample_in_batch);

                u8 label = GET_LABEL(net, dataset_idx);
                normalize_mnist_img(GET_IMG(net, dataset_idx), norm_img);

                ffnn_forward(net, norm_img);

                u8 prediction = get_prediction(GET_PREDICTION_ARR(net));
                if (prediction == label) {
                    epoch_correct++;
                }

                ffnn_backward(net, label);

                // Accumulate gradients
                for (u64 layer_idx = 0; layer_idx < net->layers.size; layer_idx++) {
                    Layer*           layer = GET_LAYER(net, layer_idx);
                    batch_gradients* bg    = (batch_gradients*)genVec_get_ptr(&gradients, layer_idx);

                    u64 total_weights = (u64)layer->n * layer->m;
                    for (u64 w = 0; w < total_weights; w++) {
                        bg->dL_dW.data[w] += layer->dL_dW.data[w];
                    }

                    for (u16 n = 0; n < layer->n; n++) {
                        bg->dL_dz[n] += layer->dL_dz[n];
                    }
                }
            }

            // Average and update weights using current_batch_size
            for (u64 layer_idx = 0; layer_idx < net->layers.size; layer_idx++) {
                Layer*           layer = GET_LAYER(net, layer_idx);
                batch_gradients* bg    = (batch_gradients*)genVec_get_ptr(&gradients, layer_idx);

                float inv_batch_size = 1.0f / (float)current_batch_size;

                u64 total_weights = (u64)layer->n * layer->m;
                for (u64 w = 0; w < total_weights; w++) {
                    layer->W.data[w] -= net->learning_rate * bg->dL_dW.data[w] * inv_batch_size;
                }

                for (u16 n = 0; n < layer->n; n++) {
                    layer->b[n] -= net->learning_rate * bg->dL_dz[n] * inv_batch_size;
                }
            }

            // Progress within epoch
            if ((batch + 1) % 100 == 0) {
                u32 samples_processed = (batch * batch_size) +
                                        (batch == num_batches - 1 && remainder > 0 ? current_batch_size : batch_size);
                float current_acc = (float)epoch_correct / (float)samples_processed * 100.0f;
                printf("    Batch %u/%u (%.2f%%)\n", batch + 1, num_batches, current_acc);
            }
        }

        // Epoch summary
        float epoch_accuracy = (float)epoch_correct / (float)net->set.num_imgs * 100.0f;
        LOG("Epoch %u Complete: Accuracy = %.2f%% (%u/%u)", epoch + 1, epoch_accuracy, epoch_correct,
            net->set.num_imgs);

        if (epoch_accuracy > best_accuracy) {
            best_accuracy = epoch_accuracy;
            LOG("  New best accuracy! ");
        }
    }

    LOG("\n     Training Complete");
    LOG("Best Accuracy: %.2f%%", best_accuracy);

    genVec_destroy_stk(&gradients);
    genVec_destroy_stk(&indices);
    arena_scratch_end(&sc);
}


b8 ffnn_save_parameters(const ffnn* net, const char* outfile)
{
    FILE* f = fopen(outfile, "wb");
    CHECK_WARN_RET(!f, false, "could'nt open file %s", outfile);

    LOG("saving parameters to %s", outfile);

    // num of layers
    u64 size = genVec_size(&net->layers);
    LOG("size : %lu", size);

    fwrite(&size, sizeof(u64), 1, f);

    for (u64 i = 0; i < size; i++) {
        // write each layer size
        // input size
        u16 m = GET_LAYER(net, i)->m;
        fwrite(&m, sizeof(u16), 1, f);
        LOG("layer %lu input: %u", i, m);
        // output size
        u16 n = GET_LAYER(net, i)->n;
        fwrite(&n, sizeof(u16), 1, f);
        LOG("layer %lu input: %u", i, n);

        // write weights
        fwrite(GET_LAYER(net, i)->W.data, sizeof(float), (u64)n * m, f);
        // write biases
        fwrite(GET_LAYER(net, i)->b, sizeof(float), n, f);
    }

    fclose(f);
    return true;
}

void ffnn_forward(ffnn* net, float* norm_img)
{
    u64 size = genVec_size(&net->layers);
    for (u64 i = 0; i < size; i++) {
        if (i == 0) {
            // input to fist hidden layer is img
            layer_calc_output(GET_LAYER(net, i), norm_img);
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
    true_label[label]    = 1;

    u64 size = genVec_size(&net->layers);

    for (u64 i = 0; i < size; i++) {
        u64 idx = size - 1 - i; // Calculate actual index

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


void normalize_mnist_img(u8* restrict img, float* restrict normalized_img)
{
    for (u64 i = 0; i < MNIST_IMG_SIZE; i++) {
        normalized_img[i] = (float)img[i] / 255.0f;
    }
}

u8 get_prediction(float* prediction_arr)
{
    float max        = -9999;
    u8    prediction = 10;

    for (u8 i = 0; i < 10; i++) {
        if (prediction_arr[i] > max) {
            max        = prediction_arr[i];
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

// calculate's a layer's total memory usage (in bytes)
u64 calc_layer_size(u16 m, u16 n) 
{
    u64 size = 0; // in bytes

    size += sizeof(Layer); // for layer struct

    size += sizeof(float) * m * 2;  // x, dL_dx
    size += sizeof(float) * n * 5; // b, z, a, dL_dz, accumilator
    size += (sizeof(Matrixf) + (sizeof(float) * m * n)) * 4; // W, dL_dW, T, accumilator
    size += sizeof(u16) * 4; // m, n, b8 

    return size;
}

void batch_gradients_init(Arena* arena, batch_gradients* bg, Layer* l)
{
    memcpy(&bg->dL_dW, &l->dL_dW, sizeof(l->dL_dW));
    bg->dL_dW.data = ARENA_ALLOC_ZERO_N(arena, float, (u64)l->n * l->m);
    bg->dL_dz      = ARENA_ALLOC_ZERO_N(arena, float, l->n);
}

void batch_gradients_clear(batch_gradients* bg)
{
    memset(bg->dL_dW.data, 0, sizeof(float) * bg->dL_dW.m * bg->dL_dW.n);

    memset(bg->dL_dz, 0, sizeof(float) * bg->dL_dW.n);
}
