#ifndef MNIST_DATA_PROCESSOR
#define MNIST_DATA_PROCESSOR

#include "common.h"



#define MNIST_TRAIN_SIZE 60000 // 60k grey scale images
#define MNIST_TEST_SIZE  10000 // 10k grey scale images

#define MNIST_IMG_DIM  28                                   // 28 x 28
#define MNIST_IMG_SIZE ((u64)MNIST_IMG_DIM * MNIST_IMG_DIM) // 28 x 28 = 784 bytes

#define MNIST_LABEL_SIZE 1 //  1 byte label

#define MNIST_SIZE_IMG_TRAIN ((u64)MNIST_TRAIN_SIZE * MNIST_IMG_SIZE)

#define MNIST_SIZE_LABEL_TRAIN ((u64)MNIST_TRAIN_SIZE * MNIST_LABEL_SIZE)



// one time loader (PRIVATE-MOVE TO C FILE)
// b8 mnist_load_from_idx(String* data_dir, Arena* arena);
// b8 mnist_save_custom_file(idx_file* img, idx_file* label, const char* outpath);

// PUBLIC API
b8 mnist_prepare_from_idx(const char* data_dir, const char* out_dir);

// load the file with the custom format : dims|label|img...
void mnist_load_custom_file(const char* filepath);


// void mnist_print_img(u8* img);



#endif // MNIST_DATA_PROCESSOR
