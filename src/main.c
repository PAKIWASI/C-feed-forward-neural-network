
#include "idx_file_reader.h"
#include "arena.h"
#include <stdio.h>



#define MNIST_TRAIN_SIZE 60000      // 60k grey scale images
#define MNIST_TEST_SIZE  10000      // 10k grey scale images

#define MNIST_IMG_DIM 28    // 28 x 28


#define MNIST_SIZE_IMG_TRAIN \
    (((u64)MNIST_TRAIN_SIZE * MNIST_IMG_DIM * MNIST_IMG_DIM))



int main(void) 
{
    const char* files[4] = {
        // training data, labels
        "/home/wasi/Documents/projects/c/ffnn/data/train-images-idx3-ubyte",
        "/home/wasi/Documents/projects/c/ffnn/data/train-labels-idx1-ubyte",
        // testing data, labels
        "/home/wasi/Documents/projects/c/ffnn/data/t10k-images-idx3-ubyte",
        "/home/wasi/Documents/projects/c/ffnn/data/train-labels-idx1-ubyte",
    };

    Arena* arena = arena_create(MNIST_SIZE_IMG_TRAIN + 1000);
    printf("alloced size: %lu\n", arena->size);

    idx_file* idx = (idx_file*)arena_alloc(arena, sizeof(idx_file));


    if (!idx_read_file( idx, files[0])) {
        return 1;
    }

    arena_release(arena);
    return 0;
}
