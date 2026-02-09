#include "mnist_data_processor.h"
#include "String.h"
#include "common.h"
#include "idx_file_reader.h"
#include "arena.h"




b8 mnist_load_from_idx(String* data_dir) 
{
    u64 path_str_len = string_len(data_dir);
    const char* files[4] = {
        // training data, labels
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        // testing data, labels
        "t10k-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
    };


    Arena* arena = arena_create(MNIST_SIZE_IMG_TRAIN + 
                                MNIST_SIZE_LABEL_TRAIN + 1000);

    printf("alloced size: %lu\n", arena->size);

    idx_file* set[2] = {    // corresponds to files[] index
        // train data idx
        (idx_file*)arena_alloc(arena, sizeof(idx_file)),
        // train label idx
        (idx_file*)arena_alloc(arena, sizeof(idx_file))
    };


    for (u8 i = 0; i < 2; i++) {    // only run for first 2 (train data + lables)

        string_append_cstr(data_dir, files[0]);
        char* full_path = string_to_cstr(data_dir);
        LOG("Reading file %s", full_path);

        if (!idx_read_file(set[i], full_path)) {
            LOG("Read Error file %s", full_path);
            return false;
        }

        // (DEBUG) Print first image
        print_hex(set[i]->data, (u64)28*28, 28);

        string_remove_range(data_dir, path_str_len, string_len(data_dir));
        free(full_path);
    }

    // now we have both img and label in set
    // now we save in custom format
    

    arena_release(arena);
    return true;
}


