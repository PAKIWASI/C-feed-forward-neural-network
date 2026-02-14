#ifndef MNIST_TESTS
#define MNIST_TESTS


#include "fast_math.h"
#include "layer.h"
#include "mnist_data_processor.h"

#define LEARNING_RATE 0.01f


int test_idx_load(void)
{
    return ! mnist_prepare_from_idx(
        "/home/wasi/Documents/projects/c/ffnn/data/raw/",
        "/home/wasi/Documents/projects/c/ffnn/data/"
    );
}

int test_mnist_read(void)
{
    Arena* arena = arena_create(nKB(1) + ((MNIST_IMG_SIZE + MNIST_LABEL_SIZE) * MNIST_TRAIN_SIZE));
    LOG("arena created with size %lu", arena_remaining(arena));

    mnist_dataset* set = (mnist_dataset*)arena_alloc(arena, sizeof(*set));

    mnist_load_custom_file(
        set, 
        "/home/wasi/Documents/projects/c/ffnn/data/dataset.bin",
        arena
    );

    mnist_print_img(set->data, 59999);

    arena_release(arena);
    return 0;
}

// checking layer struct init
int test_layer_creation_1(void)
{
    Arena* arena = arena_create(nKB(200));
    LOG("alloced %lu to arena", arena_remaining(arena));

    Layer* hl = layer_create_hidden(arena, 786, 16);
    LOG("arena usage after hl: %lu", arena_used(arena));
    Layer* ol = layer_create_output(arena, 16, 10);
    LOG("arena usage after ol: %lu", arena_used(arena));


    layer_init_weights_biases(hl);
    layer_init_weights_biases(ol);

    // print_hex((u8*)hl->W.data, sizeof(float) * 786 * 16, 786);
    // putchar('\n');
    // print_hex((u8*)hl->z, sizeof(float) * 16, 16);

    printf("\nSample Weights [HE Init]: \n");

    for (u64 i = 100; i < 200; i++) {
        for (u64 j = 0; j < 16; j++) {
            printf("%f ", MATRIX_AT(&hl->W, i, j));
        }
    }


    printf("\n\nSample Weights [Xavior Init]: \n");

    for (u64 i = 0; i < 16; i++) {
        for (u64 j = 0; j < 10; j++) {
            printf("%f ", MATRIX_AT(&ol->W, i, j));
        }
    }

    arena_release(arena);
    return 0;
}

int test_layer_creation_2(void)
{
    Arena* arena = arena_create(nKB(5));
    LOG("alloced %lu to arena", arena_remaining(arena));

    Layer* hl = layer_create_hidden(arena, 8, 16);
    LOG("arena usage after hl: %lu", arena_used(arena));
    Layer* ol = layer_create_output(arena, 16, 10);
    LOG("arena usage after ol: %lu", arena_used(arena));

    layer_init_weights_biases(hl);
    layer_init_weights_biases(ol);


    for (u8 i = 0; i < 100; i++) 
    {
        float input[8] = { 0, 1, 2, 3, 4, 5, 6, 7};

        LOG("forward pass");
        layer_calc_output(hl, (float*)&input);
        layer_calc_output(ol, hl->a);

        float true_output[10] = { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };

        LOG("backward pass");
        layer_calc_deriv(ol, (float*)true_output);
        layer_calc_deriv(hl, ol->dL_dx);
    }



    arena_release(arena);
    return 0;
}

int test_layer_creation_3(void)
{
    Arena* arena = arena_create(nKB(5));
    LOG("alloced %lu to arena", arena_remaining(arena));
    
    Layer* hl = layer_create_hidden(arena, 8, 16);
    LOG("arena usage after hl: %lu", arena_used(arena));
    
    Layer* ol = layer_create_output(arena, 16, 10);
    LOG("arena usage after ol: %lu", arena_used(arena));
    
    layer_init_weights_biases(hl);
    layer_init_weights_biases(ol);
    
    float input[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float true_output[10] = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0};  // Class 1
    
    printf("\n=== Training Loop ===\n");
    for (u8 epoch = 0; epoch < 100; epoch++) 
    {
        // Forward pass
        layer_calc_output(hl, input);
        layer_calc_output(ol, hl->a);
        
        // Calculate loss (cross-entropy): L = -Σ y_i * log(p_i)
        float loss = 0.0f;
        for (u16 i = 0; i < 10; i++) {
            if (true_output[i] > 0.0f) {  // Only sum over true labels (1s in one-hot)
                loss -= true_output[i] * fast_log(ol->a[i] + 1e-7f);  // Add epsilon to prevent log(0)
            }
        }
        
        // Find predicted class (argmax of output)
        u8 predicted_class = 0;
        float max_prob = ol->a[0];
        for (u16 i = 1; i < 10; i++) {
            if (ol->a[i] > max_prob) {
                max_prob = ol->a[i];
                predicted_class = i;
            }
        }
        
        // Find true class
        u8 true_class = 0;
        for (u16 i = 0; i < 10; i++) {
            if (true_output[i] == 1.0f) {
                true_class = i;
                break;
            }
        }
        
        // Log progress every 10 epochs
        if (epoch % 10 == 0 || epoch == 99) {
            printf("Epoch %3d | Loss: %.4f | Predicted: %d (prob=%.4f) | True: %d | %s\n",
                   epoch, loss, predicted_class, max_prob, true_class,
                   predicted_class == true_class ? "✓ CORRECT" : "✗ WRONG");
        }
        
        // Backward pass
        layer_calc_deriv(ol, true_output);
        layer_calc_deriv(hl, ol->dL_dx);
        
        // Update weights
        layer_update_WB(ol, LEARNING_RATE);
        layer_update_WB(hl, LEARNING_RATE);
    }
    
    // Final accuracy check
    printf("\n=== Final Test ===\n");
    layer_calc_output(hl, input);
    layer_calc_output(ol, hl->a);
    
    printf("Final output probabilities:\n");
    for (u16 i = 0; i < 10; i++) {
        printf("  Class %d: %.4f %s\n", i, ol->a[i], 
               true_output[i] == 1.0f ? "<-- TRUE LABEL" : "");
    }
    
    arena_release(arena);
    return 0;
}


#endif // MNIST_TESTS
