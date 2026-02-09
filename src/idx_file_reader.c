#include "idx_file_reader.h"
#include "arena.h"
#include "common.h"


b8 idx_read_file(idx_file* idx, const char* filepath, Arena* arena)
{
    CHECK_FATAL(!idx, "idx_file is null");
    CHECK_FATAL(!filepath, "filepath is null");

    FILE* f = fopen(filepath, "rb");
    if (!f) {
        LOG("couldn't open file %s", filepath);
        return 0;
    }

    // first 2 bytes (MUST BE ZERO)
    fread(&idx->first_2, sizeof(u16), 1, f);
    CHECK_FATAL(idx->first_2 != 0, "first 2 bytes not 0");

    // read next 2 bytes
    fread(&idx->data_type, sizeof(u8), 1, f);
    fread(&idx->num_dim, sizeof(u8), 1, f);
    LOG("num of dims: %u", idx->num_dim);

    // Read dimension sizes (u32 MSB each)
    idx->dim_sizes = (u32*)arena_alloc(arena, sizeof(u32) * idx->num_dim);
    fread(idx->dim_sizes, sizeof(u32), idx->num_dim, f);

    // Convert from big-endian to little-endian order
    for (u32 i = 0; i < idx->num_dim; i++) {
        idx->dim_sizes[i] = __builtin_bswap32(idx->dim_sizes[i]);
    }

    // Calculate total number of elements
    u32 total_elements = 1;
    for (u32 i = 0; i < idx->num_dim; i++) {
        total_elements *= idx->dim_sizes[i];
    }
    LOG("total u8 pixels: %u", total_elements);

    // Read all data as one contiguous block
    idx->data = arena_alloc(arena, total_elements);
    fread(idx->data, 1, total_elements, f);

    for (u32 i = 0; i < idx->num_dim; i++) {
        LOG("dim_sizes[%u] = %u", i, idx->dim_sizes[i]);
    }

    LOG("Data Type: %s", idx_get_datatype(idx));


    fclose(f);
    return true;
}
