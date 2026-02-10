

#include "arena.h"
#include "common.h"
#include "mnist_data_processor.h"



int main(void)
{
    // return ! mnist_prepare_from_idx(
    //     "/home/wasi/Documents/projects/c/ffnn/data/raw/",
    //     "/home/wasi/Documents/projects/c/ffnn/data/"
    // );

    Arena* arena = arena_create(1000 + ((MNIST_IMG_SIZE + MNIST_LABEL_SIZE) * MNIST_TRAIN_SIZE));
    LOG("arena created with size %lu", arena_remaining(arena));

    mnist_dataset* set = (mnist_dataset*)arena_alloc(arena, sizeof(*set));

    mnist_load_custom_file(
        set, 
        "/home/wasi/Documents/projects/c/ffnn/data/dataset.bin",
        arena
    );

    mnist_print_img(set->data, 59999);

    arena_release(arena);
}

