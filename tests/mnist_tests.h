#ifndef MNIST_TESTS
#define MNIST_TESTS

#include "arena.h"
#include "common.h"
#include "layer.h"
#include "matrix.h"
#include "mnist_data_processor.h"
#include <stdio.h>



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


#endif // MNIST_TESTS
