#include "arena.h"
#include "common.h"
#include "ffnn.h"


int main(void)
{
    // ffnn* net = ffnn_create_trained(
    //     "/home/wasi/Documents/projects/c/ffnn/data/WB.bin",
    //     "/home/wasi/Documents/projects/c/ffnn/data/testset.bin"
    // );
    //
    // ffnn_test(net);
    //
    // LOG("main arena usage: %lu", arena_used(net->main_arena));
    // LOG("dataset arena usage: %lu", arena_used(net->dataset_arena));
    // ffnn_destroy(net);
    // return 0;

    // pcg32_rand_seed(1234, 1);

    ffnn* net = ffnn_create(
            (u16[5]){784, 256, 128, 64, 10},
            5,
            0.01f, 
            "/home/wasi/Documents/projects/c/ffnn/data/dataset.bin");

    ffnn_train(net);
    // ffnn_train_batch_epochs(net, 32, 5);

    ffnn_save_parameters(net, "/home/wasi/Documents/projects/c/ffnn/data/256_128_64.bin");

    ffnn_change_dataset(net, "/home/wasi/Documents/projects/c/ffnn/data/testset.bin");

    ffnn_test(net);

    LOG("main arena usage: %lu", arena_used(net->main_arena));
    LOG("dataset arena usage: %lu", arena_used(net->dataset_arena));
    ffnn_destroy(net);
    return 0;
}


