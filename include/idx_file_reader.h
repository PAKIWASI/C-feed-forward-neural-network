#ifndef IDX_FILE_READER_H
#define IDX_FILE_READER_H

#include "common.h"

typedef enum {
    UNSIGNED_BYTE = 0x08,
    SIGNED_BYTE   = 0x09,
    SHORT         = 0x0B,
    INT           = 0x0C,
    FLOAT         = 0x0D,
    DOUBLE        = 0x0E,
} idx_type;


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


b8 read_idx_file(idx_file* idx, const char* filepath);


#endif // IDX_FILE_READER_H
