#ifndef TENSOR_H
#define TENSOR_H

#include "arena.h"
#include "common.h"
#include "matrix.h"


/*
 * tensor.h - Multi-dimensional tensor API for CNNs
 * Uses flat arrays with smart indexing.
 */


// TENSOR STRUCTURES
// ============================================================================

// 3D Tensor: [channels][height][width]
// feature maps, images with channels
typedef struct {
    float* data;
    u64    channels; // depth
    u64    height;   // rows
    u64    width;    // cols
} Tensor3D;

// 4D Tensor: [out_channels][in_channels][height][width]
// convolutional filters
typedef struct {
    float* data;
    u64    out_channels; // number of filters
    u64    in_channels;  // depth of each filter
    u64    height;       // filter height
    u64    width;        // filter width
} Tensor4D;


// INDEXING MACROS
// ============================================================================

// Access 3D tensor: tensor[channel][row][col]
#define TENSOR3D_AT(tensor, c, r, col) \
    ((tensor)->data[((c) * (tensor)->height * (tensor)->width) + ((r) * (tensor)->width) + (col)])

// Access 4D tensor: tensor[out_ch][in_ch][row][col]
#define TENSOR4D_AT(tensor, oc, ic, r, col)                                               \
    ((tensor)->data[((oc) * (tensor)->in_channels * (tensor)->height * (tensor)->width) + \
                    ((ic) * (tensor)->height * (tensor)->width) + ((r) * (tensor)->width) + (col)])

// Get total number of elements
#define TENSOR3D_TOTAL(tensor) ((u64)(tensor)->channels * (tensor)->height * (tensor)->width)

#define TENSOR4D_TOTAL(tensor) \
    ((u64)(tensor)->out_channels * (tensor)->in_channels * (tensor)->height * (tensor)->width)

// Linear index (useful for iteration)
#define TENSOR3D_IDX(tensor, c, r, col) ((c) * (tensor)->height * (tensor)->width + (r) * (tensor)->width + (col))

#define TENSOR4D_IDX(tensor, oc, ic, r, col)                                                                         \
    ((oc) * (tensor)->in_channels * (tensor)->height * (tensor)->width + (ic) * (tensor)->height * (tensor)->width + \
     (r) * (tensor)->width + (col))


// CREATION AND DESTRUCTION - 3D TENSORS
// ============================================================================

// Create heap tensor with channels × height × width
Tensor3D* tensor3d_create(u64 channels, u64 height, u64 width);

// Create heap tensor with initial values from array
Tensor3D* tensor3d_create_arr(u64 channels, u64 height, u64 width, const float* arr);

// Create tensor with everything on the stack (data must be pre-allocated)
void tensor3d_create_stk(Tensor3D* tensor, u64 channels, u64 height, u64 width, float* data);

// Destroy heap-allocated tensor
void tensor3d_destroy(Tensor3D* tensor);


// CREATION AND DESTRUCTION - 4D TENSORS
// ============================================================================

// Create heap tensor with out_channels × in_channels × height × width
Tensor4D* tensor4d_create(u64 out_channels, u64 in_channels, u64 height, u64 width);

// Create heap tensor with initial values
Tensor4D* tensor4d_create_arr(u64 out_channels, u64 in_channels, u64 height, u64 width, const float* arr);

// Create tensor on stack
void tensor4d_create_stk(Tensor4D* tensor, u64 out_channels, u64 in_channels, u64 height, u64 width, float* data);

// Destroy heap-allocated tensor
void tensor4d_destroy(Tensor4D* tensor);


// SETTERS - 3D TENSORS
// ============================================================================

// Set all values from flat array (row-major order: C, H, W)
void tensor3d_set_val_arr(Tensor3D* tensor, u64 count, const float* arr);

// Set single element at [channel][row][col]
void tensor3d_set_elm(Tensor3D* tensor, float value, u64 c, u64 r, u64 col);

// Fill entire tensor with a value
void tensor3d_fill(Tensor3D* tensor, float value);

// Fill with zeros
void tensor3d_zero(Tensor3D* tensor);


// SETTERS - 4D TENSORS
// ============================================================================

// Set all values from flat array
void tensor4d_set_val_arr(Tensor4D* tensor, u64 count, const float* arr);

// Set single element at [out_ch][in_ch][row][col]
void tensor4d_set_elm(Tensor4D* tensor, float value, u64 oc, u64 ic, u64 r, u64 col);

// Fill entire tensor with a value
void tensor4d_fill(Tensor4D* tensor, float value);

// Fill with zeros
void tensor4d_zero(Tensor4D* tensor);


// BASIC OPERATIONS - 3D TENSORS
// ============================================================================

// Tensor addition: out = a + b
// out may alias a and/or b
void tensor3d_add(Tensor3D* out, const Tensor3D* a, const Tensor3D* b);

// Tensor subtraction: out = a - b
// out may alias a and/or b
void tensor3d_sub(Tensor3D* out, const Tensor3D* a, const Tensor3D* b);

// Scalar multiplication: tensor = tensor * val
void tensor3d_scale(Tensor3D* tensor, float val);

// Element-wise division: tensor = tensor / val
void tensor3d_div(Tensor3D* tensor, float val);

// Tensor copy: dest = src
void tensor3d_copy(Tensor3D* dest, const Tensor3D* src);


// BASIC OPERATIONS - 4D TENSORS
// ============================================================================

// Tensor addition: out = a + b
void tensor4d_add(Tensor4D* out, const Tensor4D* a, const Tensor4D* b);

// Tensor subtraction: out = a - b
void tensor4d_sub(Tensor4D* out, const Tensor4D* a, const Tensor4D* b);

// Scalar multiplication
void tensor4d_scale(Tensor4D* tensor, float val);

// Element-wise division
void tensor4d_div(Tensor4D* tensor, float val);

// Tensor copy
void tensor4d_copy(Tensor4D* dest, const Tensor4D* src);


// CHANNEL OPERATIONS - 3D TENSORS
// ============================================================================

// Get pointer to a specific channel (useful for iteration)
static inline float* tensor3d_channel_ptr(Tensor3D* tensor, u64 channel)
{
    return &tensor->data[channel * tensor->height * tensor->width];
}

// Copy a single channel from one tensor to another
void tensor3d_copy_channel(Tensor3D* dest, u64 dest_ch, const Tensor3D* src, u64 src_ch);

// Sum across channels: out[h][w] = sum over all channels
// out must be height × width array
void tensor3d_sum_channels(float* out, const Tensor3D* tensor);

// Mean across channels
void tensor3d_mean_channels(float* out, const Tensor3D* tensor);


// FILTER OPERATIONS - 4D TENSORS
// ============================================================================

// Get pointer to a specific filter (all input channels for one output)
static inline float* tensor4d_filter_ptr(Tensor4D* tensor, u64 out_channel)
{
    return &tensor->data[out_channel * tensor->in_channels * tensor->height * tensor->width];
}

// Copy a single filter
void tensor4d_copy_filter(Tensor4D* dest, u64 dest_filter, const Tensor4D* src, u64 src_filter);


// SHAPE OPERATIONS
// ============================================================================

// Reshape 3D tensor (total size must match)
void tensor3d_reshape(Tensor3D* tensor, u64 new_channels, u64 new_height, u64 new_width);

// Flatten 3D tensor to 1D array (returns pointer to data)
// Useful for connecting conv layers to fully connected layers
static inline float* tensor3d_flatten(Tensor3D* tensor)
{
    return tensor->data;
}

// Get flattened size
static inline u64 tensor3d_flatten_size(const Tensor3D* tensor)
{
    return TENSOR3D_TOTAL(tensor);
}


// PADDING OPERATIONS
// ============================================================================

// Add zero padding to 3D tensor
// Creates new tensor with padding, original tensor unchanged
Tensor3D* tensor3d_pad(const Tensor3D* input, u64 pad_h, u64 pad_w, Arena* arena);

// Remove padding (crop) from 3D tensor
Tensor3D* tensor3d_unpad(const Tensor3D* input, u64 pad_h, u64 pad_w, Arena* arena);


// UTILITIES
// ============================================================================

// Print 3D tensor (prints each channel as a 2D matrix)
void tensor3d_print(const Tensor3D* tensor);

// Print 4D tensor (prints each filter)
void tensor4d_print(const Tensor4D* tensor);

// Print single channel of 3D tensor
void tensor3d_print_channel(const Tensor3D* tensor, u64 channel);

// Print single filter of 4D tensor
void tensor4d_print_filter(const Tensor4D* tensor, u64 filter);

// Check if two tensors have same dimensions
b8 tensor3d_same_shape(const Tensor3D* a, const Tensor3D* b);
b8 tensor4d_same_shape(const Tensor4D* a, const Tensor4D* b);


// ARENA-BASED ALLOCATION - 3D TENSORS
// ============================================================================

/*
Create 3D tensor allocated from arena
No need to call tensor3d_destroy - freed when arena is cleared/released

Usage:
    Tensor3D* features = tensor3d_arena_alloc(arena, 32, 28, 28);
*/
static inline Tensor3D* tensor3d_arena_alloc(Arena* arena, u64 channels, u64 height, u64 width)
{
    CHECK_FATAL(channels == 0 || height == 0 || width == 0, "tensor dimensions must be > 0");

    Tensor3D* tensor = ARENA_ALLOC(arena, Tensor3D);
    CHECK_FATAL(!tensor, "tensor3d arena allocation failed");

    tensor->channels = channels;
    tensor->height   = height;
    tensor->width    = width;

    u64 total    = channels * height * width;
    tensor->data = ARENA_ALLOC_N(arena, float, total);
    CHECK_FATAL(!tensor->data, "tensor3d data arena allocation failed");

    return tensor;
}

/*
Create 3D tensor from arena with initial values

Usage:
    float data[32*28*28] = {...};
    Tensor3D* t = tensor3d_arena_arr_alloc(arena, 32, 28, 28, data);
*/
static inline Tensor3D* tensor3d_arena_arr_alloc(Arena* arena, u64 channels, u64 height, u64 width, const float* arr)
{
    CHECK_FATAL(!arr, "input array is null");

    Tensor3D* tensor = tensor3d_arena_alloc(arena, channels, height, width);
    u64       total  = TENSOR3D_TOTAL(tensor);
    memcpy(tensor->data, arr, sizeof(float) * total);

    return tensor;
}

/*
Create 3D tensor from arena, initialized to zero

Usage:
    Tensor3D* t = tensor3d_arena_zero(arena, 32, 28, 28);
*/
static inline Tensor3D* tensor3d_arena_zero(Arena* arena, u64 channels, u64 height, u64 width)
{
    Tensor3D* tensor = tensor3d_arena_alloc(arena, channels, height, width);
    tensor3d_zero(tensor);
    return tensor;
}


// ARENA-BASED ALLOCATION - 4D TENSORS
// ============================================================================

/*
Create 4D tensor allocated from arena

Usage:
    Tensor4D* filters = tensor4d_arena_alloc(arena, 32, 1, 5, 5);
*/
static inline Tensor4D* tensor4d_arena_alloc(Arena* arena, u64 out_channels, u64 in_channels, u64 height, u64 width)
{
    CHECK_FATAL(out_channels == 0 || in_channels == 0 || height == 0 || width == 0, "tensor dimensions must be > 0");

    Tensor4D* tensor = ARENA_ALLOC(arena, Tensor4D);
    CHECK_FATAL(!tensor, "tensor4d arena allocation failed");

    tensor->out_channels = out_channels;
    tensor->in_channels  = in_channels;
    tensor->height       = height;
    tensor->width        = width;

    u64 total    = out_channels * in_channels * height * width;
    tensor->data = ARENA_ALLOC_N(arena, float, total);
    CHECK_FATAL(!tensor->data, "tensor4d data arena allocation failed");

    return tensor;
}

/*
Create 4D tensor from arena with initial values

Usage:
    float weights[32*1*5*5] = {...};
    Tensor4D* t = tensor4d_arena_arr_alloc(arena, 32, 1, 5, 5, weights);
*/
static inline Tensor4D* tensor4d_arena_arr_alloc(Arena* arena, u64 out_channels, u64 in_channels, u64 height, u64 width,
                                                 const float* arr)
{
    CHECK_FATAL(!arr, "input array is null");

    Tensor4D* tensor = tensor4d_arena_alloc(arena, out_channels, in_channels, height, width);
    u64       total  = TENSOR4D_TOTAL(tensor);
    memcpy(tensor->data, arr, sizeof(float) * total);

    return tensor;
}

/*
Create 4D tensor from arena, initialized to zero

Usage:
    Tensor4D* t = tensor4d_arena_zero(arena, 32, 1, 5, 5);
*/
static inline Tensor4D* tensor4d_arena_zero(Arena* arena, u64 out_channels, u64 in_channels, u64 height, u64 width)
{
    Tensor4D* tensor = tensor4d_arena_alloc(arena, out_channels, in_channels, height, width);
    tensor4d_zero(tensor);
    return tensor;
}


// CONVENIENCE MACROS
// ============================================================================

// Create zero-initialized tensors on stack (data on heap)
#define TENSOR3D_ZEROS(arena, c, h, w)      tensor3d_arena_zero(arena, c, h, w)
#define TENSOR4D_ZEROS(arena, oc, ic, h, w) tensor4d_arena_zero(arena, oc, ic, h, w)


// CONVERSION BETWEEN MATRIX AND TENSOR
// ============================================================================

/*
Convert 2D matrix to 3D tensor (1 channel)
Useful for converting grayscale images

Usage:
    Matrix* img = ...;  // 28 × 28
    Tensor3D* t = matrix_to_tensor3d(img, arena);
    // Result: 1 × 28 × 28 tensor
*/
Tensor3D* matrix_to_tensor3d(const Matrixf* mat, Arena* arena);

/*
Convert 3D tensor (single channel) to 2D matrix

Usage:
    Tensor3D* t = ...;  // 1 × 28 × 28
    Matrix* img = tensor3d_to_matrix(t, arena);
    // Result: 28 × 28 matrix
*/
Matrixf* tensor3d_to_matrix(const Tensor3D* tensor, Arena* arena);

/*
Extract a single channel from 3D tensor as a matrix

Usage:
    Tensor3D* features = ...;  // 32 × 14 × 14
    Matrix* channel5 = tensor3d_channel_to_matrix(features, 5, arena);
    // Result: 14 × 14 matrix of channel 5
*/
Matrixf* tensor3d_channel_to_matrix(const Tensor3D* tensor, u64 channel, Arena* arena);


// STATISTICS AND NORMALIZATION
// ============================================================================

// Calculate min/max values in tensor
float tensor3d_min(const Tensor3D* tensor);
float tensor3d_max(const Tensor3D* tensor);

// Calculate mean and standard deviation
float tensor3d_mean(const Tensor3D* tensor);
float tensor3d_std(const Tensor3D* tensor);

// Normalize tensor: (x - mean) / std
void tensor3d_normalize(Tensor3D* tensor);

// Min-max normalization: (x - min) / (max - min)
void tensor3d_normalize_minmax(Tensor3D* tensor);


#endif // TENSOR_H
