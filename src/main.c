
#include "idx_file_reader.h"
#include "arena.h"
#include <stdio.h>



#define NEEDED_SIZE (((u64)60000*28*28) + 1000)


int main(void) 
{
    Arena* arena = arena_create(NEEDED_SIZE);
    printf("alloced size: %lu", NEEDED_SIZE);

    idx_file* idx = (idx_file*)arena_alloc(arena, sizeof(idx_file));


    if (!read_idx_file(
            idx, 
            "/home/wasi/Documents/projects/c/ffnn/data/train-images-idx3-ubyte"
        )) {
        return 1;
    }

    arena_release(arena);
    return 0;
}
