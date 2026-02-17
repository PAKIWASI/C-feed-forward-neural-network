#include "arena.h"
#include "common.h"
#include "ffnn.h"
#include "random.h"


int main(void)
{
    // ffnn* net = ffnn_create_trained("data/512.bin");
    //
    // ffnn_set_dataset(net, "data/testset.bin");
    //
    // ffnn_test(net);
    //
    // LOG("main arena usage: %lu", arena_used(net->main_arena));
    // LOG("dataset arena usage: %lu", arena_used(net->dataset_arena));
    // ffnn_destroy(net);
    // return 0;


    pcg32_rand_seed(1234, 1);

    ffnn* net = ffnn_create(
            (u16[3]){784, 128, 10},
            3,
            0.015f, 
            "data/dataset.bin");

    ffnn_train(net);
    // ffnn_train_batch(net, 64, 5);

    ffnn_save_parameters(net, "data/128.bin");

    ffnn_set_dataset(net, "data/testset.bin");

    ffnn_test(net);

    LOG("main arena usage: %lu", arena_used(net->main_arena));
    LOG("dataset arena usage: %lu", arena_used(net->dataset_arena));
    ffnn_destroy(net);
    return 0;
}


