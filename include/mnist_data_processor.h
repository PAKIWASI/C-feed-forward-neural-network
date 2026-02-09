#ifndef MNIST_DATA_PROCESSOR
#define MNIST_DATA_PROCESSOR

#include "String.h"
#include "idx_file_reader.h"



#define MNIST_TRAIN_SIZE 60000      // 60k grey scale images
#define MNIST_TEST_SIZE  10000      // 10k grey scale images

#define MNIST_IMG_DIM 28    // 28 x 28

#define MNIST_LABEL_SIZE 1  //  1 byte label

#define MNIST_SIZE_IMG_TRAIN \
    (((u64)MNIST_TRAIN_SIZE * MNIST_IMG_DIM * MNIST_IMG_DIM))

#define MNIST_SIZE_LABEL_TRAIN \
    ((u64)MNIST_TRAIN_SIZE * MNIST_LABEL_SIZE)



// one time loader
b8 mnist_load_from_idx(String* data_dir);
b8 mnist_save_to_file(idx_file* idx, const char* outpath);

b8 mnist_prepare_from_idx(const char* data_dir, const char* out_dir);

void mnist_load_from_file(const char* filepath);

void mnist_print_img(u8* img);



/*
    all data processing done here
    calls idx_file_reader
    all things happening in main rn will go here
    stores in custom format : img, label next to each other in a single file
*/

#endif // MNIST_DATA_PROCESSOR
