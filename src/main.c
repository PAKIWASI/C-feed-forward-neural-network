#include "ffnn.h"


int main(void)
{
    ffnn* net = ffnn_create(
            (u16[4]){784, 64, 32, 10},
            4,
            0.01f, 
            "/home/wasi/Documents/projects/c/ffnn/data/dataset.bin");

    ffnn_train(net);

    ffnn_save_parameters(net, "/home/wasi/Documents/projects/c/ffnn/data/WB.bin");

    ffnn_change_dataset(net, "/home/wasi/Documents/projects/c/ffnn/data/testset.bin");

    ffnn_test(net);

    ffnn_destroy(net);
}


