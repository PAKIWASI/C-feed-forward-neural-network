
#include "idx_file_reader.h"
// #include "arena_single.h"
// #include "genVec_single.h"
#include <stdio.h>
#include <stdlib.h>


b8 idx_read_file(idx_file* idx, const char* filepath)
{
    CHECK_FATAL(!idx, "idx_file is null");
    CHECK_FATAL(!filepath, "filepath is null");

    FILE* f = fopen(filepath, "rb");
    if (!f) { return 0; }
    
    fread(&idx->first_2, sizeof(u16), 1, f);

    CHECK_FATAL(idx->first_2 != 0, "first 2 bytes not 0");
    
    fread(&idx->data_type, sizeof(u8), 1, f);
    fread(&idx->num_dim, sizeof(u8), 1, f);
    printf("num of dims: %u\n", idx->num_dim);
    
    // Read dimension sizes
    idx->dim_sizes = malloc(sizeof(u32) * idx->num_dim);
    fread(idx->dim_sizes, sizeof(u32), idx->num_dim, f);
    
    // Convert from big-endian to host byte order
    for (u32 i = 0; i < idx->num_dim; i++) {
        idx->dim_sizes[i] = __builtin_bswap32(idx->dim_sizes[i]);
    }
    
    // Calculate total number of elements
    u32 total_elements = 1;
    for (u32 i = 0; i < idx->num_dim; i++) {
        total_elements *= idx->dim_sizes[i];
    }
    printf("total u8 pixels: %u\n", total_elements);
    
    // Read all data as one contiguous block
    idx->data = malloc(total_elements);
    fread(idx->data, 1, total_elements, f);
    
    printf("Dimensions: \n");
    for (u32 i = 0; i < idx->num_dim; i++) {
        printf("dim_sizes[%u] = %u\n", i, idx->dim_sizes[i]);
    }
    
    // Print first few pixels of first image
    print_hex(idx->data, 64, 8);
    
    idx_type type = idx->data_type;
    printf("data type: %d\n", type);

    free(idx->data);
    free(idx->dim_sizes);
    fclose(f);

    return true;
}



