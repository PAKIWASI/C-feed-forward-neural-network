#ifndef IDX_FILE_READER_H
#define IDX_FILE_READER_H

#include "common.h"


typedef struct {
    //u32 magic;    // first 2 bytes must be 0 (MSB)
    u16  first_2;
    u8   data_type;  // from magic
    u8   num_dim;   // from magic
    u32* dim_sizes; // sizes for each dimention (MSB)
    u8*  data;       // raw image data or labels (0-9)
                     // per byte pixel values (0-255)
    // genVec dim_sizes;   // type u32
    // genVec data;        // type u8
} idx_file;


// idx_file* idx_file_init(Arena* arena);      // Optional arena
// void idx_file_destroy(idx_file* idx_file);  // if heap alloced


b8 idx_read_file(idx_file* idx, const char* filepath);

b8 idx_save_custom_format(const idx_file* idx, const char* outfile);

const char* idx_get_datatype(const idx_file* idx) {
    switch (idx->data_type) {
        case 0x08: return "u8";
        case 0x09: return "i8";
        case 0x0B: return "i16";
        case 0x0C: return "i32";
        case 0x0D: return "f32";
        case 0x0E: return "f64";
        default: return "void";
    }
}


#endif // IDX_FILE_READER_H
