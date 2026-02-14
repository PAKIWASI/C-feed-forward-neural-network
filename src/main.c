#include "arena.h"
#include "common.h"
#include "ffnn.h"
#include "random.h"


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

    pcg32_rand_seed(1234, 1);

    ffnn* net = ffnn_create(
            (u16[4]){784, 32, 32, 10}, // TODO:
            4,
            0.001f, 
            "/home/wasi/Documents/projects/c/ffnn/data/dataset.bin");

    ffnn_train(net);

    ffnn_save_parameters(net, "/home/wasi/Documents/projects/c/ffnn/data/32_32.bin");

    ffnn_change_dataset(net, "/home/wasi/Documents/projects/c/ffnn/data/testset.bin");

    ffnn_test(net);

    LOG("main arena usage: %lu", arena_used(net->main_arena));
    LOG("dataset arena usage: %lu", arena_used(net->dataset_arena));
    ffnn_destroy(net);
    return 0;
}


