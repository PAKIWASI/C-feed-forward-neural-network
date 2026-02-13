#include "arena.h"
#include "common.h"
#include "ffnn.h"
#include "mnist_data_processor.h"



int main(void)
{
    Arena* arena = arena_create(nMB(1) + ((u64)MNIST_TRAIN_SIZE * (MNIST_IMG_SIZE + MNIST_LABEL_SIZE)));
    LOG("allocated %lu bytes for arena", arena_remaining(arena));

    ffnn* net = ffnn_create(
                    arena,
                    (u16[4]){786, 64, 64, 10}, 
                    4, 
                    0.01f, 
                    "/home/wasi/Documents/projects/c/ffnn/data/dataset.bin"
                );

    ffnn_train(net);

    ffnn_save_parameters(net, "/home/wasi/Documents/projects/c/ffnn/data/WB.bin");

    ffnn_destroy(net);
    arena_release(arena);
}


