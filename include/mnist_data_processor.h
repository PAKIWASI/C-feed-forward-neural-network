#ifndef MNIST_DATA_PROCESSOR
#define MNIST_DATA_PROCESSOR

#include "common.h"
#include "arena.h"



#define MNIST_TRAIN_SIZE 60000 // 60k grey scale images
#define MNIST_TEST_SIZE  10000 // 10k grey scale images

#define MNIST_IMG_DIM  28                                   // 28 x 28
#define MNIST_IMG_SIZE ((u64)MNIST_IMG_DIM * MNIST_IMG_DIM) // 28 x 28 = 784 bytes

#define MNIST_LABEL_SIZE 1 //  1 byte label

#define MNIST_SIZE_IMG_TRAIN ((u64)MNIST_TRAIN_SIZE * MNIST_IMG_SIZE)

#define MNIST_SIZE_LABEL_TRAIN ((u64)MNIST_TRAIN_SIZE * MNIST_LABEL_SIZE)


typedef struct {
    u8* data;
    u16 num_imgs;
    u8 img_w;
    u8 img_h;
} mnist_dataset;


// TODO: mmap?



// PUBLIC API
b8 mnist_prepare_from_idx(const char* data_dir, const char* out_dir);

// load the file with the custom format : dims|label|img...
b8 mnist_load_custom_file(mnist_dataset* set, const char* filepath, Arena* arena);

// print img of size MNIST_IMG_SIZE, label of size MNIST_LABEL_SIZE, at index
void mnist_print_img(u8* data, u64 index);



#endif // MNIST_DATA_PROCESSOR
