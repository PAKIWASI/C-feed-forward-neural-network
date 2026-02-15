#include "tensor.h"
#include "fast_math.h"


// CREATION AND DESTRUCTION - 3D TENSORS
// ============================================================================

Tensor3D* tensor3d_create(u64 channels, u64 height, u64 width)
{
    CHECK_FATAL(channels == 0 || height == 0 || width == 0,
                "tensor dimensions must be > 0");

    Tensor3D* tensor = malloc(sizeof(Tensor3D));
    CHECK_FATAL(!tensor, "tensor3d malloc failed");

    tensor->channels = channels;
    tensor->height = height;
    tensor->width = width;

    u64 total = channels * height * width;
    tensor->data = malloc(sizeof(float) * total);
    CHECK_FATAL(!tensor->data, "tensor3d data malloc failed");

    return tensor;
}

Tensor3D* tensor3d_create_arr(u64 channels, u64 height, u64 width,
                               const float* arr)
{
    CHECK_FATAL(!arr, "input array is null");

    Tensor3D* tensor = tensor3d_create(channels, height, width);
    u64 total = TENSOR3D_TOTAL(tensor);
    memcpy(tensor->data, arr, sizeof(float) * total);

    return tensor;
}

void tensor3d_create_stk(Tensor3D* tensor, u64 channels, u64 height,
                         u64 width, float* data)
{
    CHECK_FATAL(!tensor || !data, "null pointer");
    CHECK_FATAL(channels == 0 || height == 0 || width == 0,
                "tensor dimensions must be > 0");

    tensor->channels = channels;
    tensor->height = height;
    tensor->width = width;
    tensor->data = data;
}

void tensor3d_destroy(Tensor3D* tensor)
{
    if (tensor) {
        if (tensor->data) {
            free(tensor->data);
        }
        free(tensor);
    }
}


// CREATION AND DESTRUCTION - 4D TENSORS
// ============================================================================

Tensor4D* tensor4d_create(u64 out_channels, u64 in_channels,
                          u64 height, u64 width)
{
    CHECK_FATAL(out_channels == 0 || in_channels == 0 ||
                height == 0 || width == 0,
                "tensor dimensions must be > 0");

    Tensor4D* tensor = malloc(sizeof(Tensor4D));
    CHECK_FATAL(!tensor, "tensor4d malloc failed");

    tensor->out_channels = out_channels;
    tensor->in_channels = in_channels;
    tensor->height = height;
    tensor->width = width;

    u64 total = out_channels * in_channels * height * width;
    tensor->data = malloc(sizeof(float) * total);
    CHECK_FATAL(!tensor->data, "tensor4d data malloc failed");

    return tensor;
}

Tensor4D* tensor4d_create_arr(u64 out_channels, u64 in_channels,
                               u64 height, u64 width, const float* arr)
{
    CHECK_FATAL(!arr, "input array is null");

    Tensor4D* tensor = tensor4d_create(out_channels, in_channels, height, width);
    u64 total = TENSOR4D_TOTAL(tensor);
    memcpy(tensor->data, arr, sizeof(float) * total);

    return tensor;
}

void tensor4d_create_stk(Tensor4D* tensor, u64 out_channels, u64 in_channels,
                         u64 height, u64 width, float* data)
{
    CHECK_FATAL(!tensor || !data, "null pointer");
    CHECK_FATAL(out_channels == 0 || in_channels == 0 ||
                height == 0 || width == 0,
                "tensor dimensions must be > 0");

    tensor->out_channels = out_channels;
    tensor->in_channels = in_channels;
    tensor->height = height;
    tensor->width = width;
    tensor->data = data;
}

void tensor4d_destroy(Tensor4D* tensor)
{
    if (tensor) {
        if (tensor->data) {
            free(tensor->data);
        }
        free(tensor);
    }
}


// SETTERS - 3D TENSORS
// ============================================================================

void tensor3d_set_val_arr(Tensor3D* tensor, u64 count, const float* arr)
{
    CHECK_FATAL(!tensor || !arr, "null pointer");
    u64 total = TENSOR3D_TOTAL(tensor);
    CHECK_FATAL(count != total, "array size mismatch");

    memcpy(tensor->data, arr, sizeof(float) * count);
}

void tensor3d_set_elm(Tensor3D* tensor, float value, u64 c, u64 r, u64 col)
{
    CHECK_FATAL(!tensor, "null tensor");
    CHECK_FATAL(c >= tensor->channels || r >= tensor->height || col >= tensor->width,
                "index out of bounds");

    TENSOR3D_AT(tensor, c, r, col) = value;
}

void tensor3d_fill(Tensor3D* tensor, float value)
{
    CHECK_FATAL(!tensor, "null tensor");

    u64 total = TENSOR3D_TOTAL(tensor);
    for (u64 i = 0; i < total; i++) {
        tensor->data[i] = value;
    }
}

void tensor3d_zero(Tensor3D* tensor)
{
    CHECK_FATAL(!tensor, "null tensor");
    u64 total = TENSOR3D_TOTAL(tensor);
    memset(tensor->data, 0, sizeof(float) * total);
}


// SETTERS - 4D TENSORS
// ============================================================================

void tensor4d_set_val_arr(Tensor4D* tensor, u64 count, const float* arr)
{
    CHECK_FATAL(!tensor || !arr, "null pointer");
    u64 total = TENSOR4D_TOTAL(tensor);
    CHECK_FATAL(count != total, "array size mismatch");

    memcpy(tensor->data, arr, sizeof(float) * count);
}

void tensor4d_set_elm(Tensor4D* tensor, float value,
                      u64 oc, u64 ic, u64 r, u64 col)
{
    CHECK_FATAL(!tensor, "null tensor");
    CHECK_FATAL(oc >= tensor->out_channels || ic >= tensor->in_channels ||
                r >= tensor->height || col >= tensor->width,
                "index out of bounds");

    TENSOR4D_AT(tensor, oc, ic, r, col) = value;
}

void tensor4d_fill(Tensor4D* tensor, float value)
{
    CHECK_FATAL(!tensor, "null tensor");

    u64 total = TENSOR4D_TOTAL(tensor);
    for (u64 i = 0; i < total; i++) {
        tensor->data[i] = value;
    }
}

void tensor4d_zero(Tensor4D* tensor)
{
    CHECK_FATAL(!tensor, "null tensor");
    u64 total = TENSOR4D_TOTAL(tensor);
    memset(tensor->data, 0, sizeof(float) * total);
}


// BASIC OPERATIONS - 3D TENSORS
// ============================================================================

void tensor3d_add(Tensor3D* out, const Tensor3D* a, const Tensor3D* b)
{
    CHECK_FATAL(!out || !a || !b, "null tensor");
    CHECK_FATAL(!tensor3d_same_shape(a, b) || !tensor3d_same_shape(a, out),
                "tensor shape mismatch");

    u64 total = TENSOR3D_TOTAL(out);
    for (u64 i = 0; i < total; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

void tensor3d_sub(Tensor3D* out, const Tensor3D* a, const Tensor3D* b)
{
    CHECK_FATAL(!out || !a || !b, "null tensor");
    CHECK_FATAL(!tensor3d_same_shape(a, b) || !tensor3d_same_shape(a, out),
                "tensor shape mismatch");

    u64 total = TENSOR3D_TOTAL(out);
    for (u64 i = 0; i < total; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
}

void tensor3d_scale(Tensor3D* tensor, float val)
{
    CHECK_FATAL(!tensor, "null tensor");

    u64 total = TENSOR3D_TOTAL(tensor);
    for (u64 i = 0; i < total; i++) {
        tensor->data[i] *= val;
    }
}

void tensor3d_div(Tensor3D* tensor, float val)
{
    CHECK_FATAL(!tensor, "null tensor");
    CHECK_FATAL(val == 0.0f, "division by zero");

    u64 total = TENSOR3D_TOTAL(tensor);
    for (u64 i = 0; i < total; i++) {
        tensor->data[i] /= val;
    }
}

void tensor3d_copy(Tensor3D* dest, const Tensor3D* src)
{
    CHECK_FATAL(!dest || !src, "null tensor");
    CHECK_FATAL(!tensor3d_same_shape(dest, src), "tensor shape mismatch");

    u64 total = TENSOR3D_TOTAL(src);
    memcpy(dest->data, src->data, sizeof(float) * total);
}


// BASIC OPERATIONS - 4D TENSORS
// ============================================================================

void tensor4d_add(Tensor4D* out, const Tensor4D* a, const Tensor4D* b)
{
    CHECK_FATAL(!out || !a || !b, "null tensor");
    CHECK_FATAL(!tensor4d_same_shape(a, b) || !tensor4d_same_shape(a, out),
                "tensor shape mismatch");

    u64 total = TENSOR4D_TOTAL(out);
    for (u64 i = 0; i < total; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

void tensor4d_sub(Tensor4D* out, const Tensor4D* a, const Tensor4D* b)
{
    CHECK_FATAL(!out || !a || !b, "null tensor");
    CHECK_FATAL(!tensor4d_same_shape(a, b) || !tensor4d_same_shape(a, out),
                "tensor shape mismatch");

    u64 total = TENSOR4D_TOTAL(out);
    for (u64 i = 0; i < total; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
}

void tensor4d_scale(Tensor4D* tensor, float val)
{
    CHECK_FATAL(!tensor, "null tensor");

    u64 total = TENSOR4D_TOTAL(tensor);
    for (u64 i = 0; i < total; i++) {
        tensor->data[i] *= val;
    }
}

void tensor4d_div(Tensor4D* tensor, float val)
{
    CHECK_FATAL(!tensor, "null tensor");
    CHECK_FATAL(val == 0.0f, "division by zero");

    u64 total = TENSOR4D_TOTAL(tensor);
    for (u64 i = 0; i < total; i++) {
        tensor->data[i] /= val;
    }
}

void tensor4d_copy(Tensor4D* dest, const Tensor4D* src)
{
    CHECK_FATAL(!dest || !src, "null tensor");
    CHECK_FATAL(!tensor4d_same_shape(dest, src), "tensor shape mismatch");

    u64 total = TENSOR4D_TOTAL(src);
    memcpy(dest->data, src->data, sizeof(float) * total);
}


// CHANNEL OPERATIONS - 3D TENSORS
// ============================================================================

void tensor3d_copy_channel(Tensor3D* dest, u64 dest_ch,
                           const Tensor3D* src, u64 src_ch)
{
    CHECK_FATAL(!dest || !src, "null tensor");
    CHECK_FATAL(dest_ch >= dest->channels || src_ch >= src->channels,
                "channel index out of bounds");
    CHECK_FATAL(dest->height != src->height || dest->width != src->width,
                "tensor spatial dimensions mismatch");

    u64 channel_size = dest->height * dest->width;
    float* dest_ptr = tensor3d_channel_ptr(dest, dest_ch);
    float* src_ptr = tensor3d_channel_ptr((Tensor3D*)src, src_ch);

    memcpy(dest_ptr, src_ptr, sizeof(float) * channel_size);
}

void tensor3d_sum_channels(float* out, const Tensor3D* tensor)
{
    CHECK_FATAL(!out || !tensor, "null pointer");

    u64 spatial_size = (u64)tensor->height * tensor->width;

    // Initialize output to zero
    memset(out, 0, sizeof(float) * spatial_size);

    // Sum across all channels
    for (u64 c = 0; c < tensor->channels; c++) {
        float* channel_data = tensor3d_channel_ptr((Tensor3D*)tensor, c);
        for (u64 i = 0; i < spatial_size; i++) {
            out[i] += channel_data[i];
        }
    }
}

void tensor3d_mean_channels(float* out, const Tensor3D* tensor)
{
    tensor3d_sum_channels(out, tensor);

    u64 spatial_size = (u64)tensor->height * tensor->width;
    float scale = 1.0f / (float)tensor->channels;

    for (u64 i = 0; i < spatial_size; i++) {
        out[i] *= scale;
    }
}


// FILTER OPERATIONS - 4D TENSORS
// ============================================================================

void tensor4d_copy_filter(Tensor4D* dest, u64 dest_filter,
                          const Tensor4D* src, u64 src_filter)
{
    CHECK_FATAL(!dest || !src, "null tensor");
    CHECK_FATAL(dest_filter >= dest->out_channels || 
                src_filter >= src->out_channels,
                "filter index out of bounds");
    CHECK_FATAL(dest->in_channels != src->in_channels ||
                dest->height != src->height || dest->width != src->width,
                "filter dimensions mismatch");

    u64 filter_size = dest->in_channels * dest->height * dest->width;
    float* dest_ptr = tensor4d_filter_ptr(dest, dest_filter);
    float* src_ptr = tensor4d_filter_ptr((Tensor4D*)src, src_filter);

    memcpy(dest_ptr, src_ptr, sizeof(float) * filter_size);
}


// SHAPE OPERATIONS
// ============================================================================

void tensor3d_reshape(Tensor3D* tensor, u64 new_channels,
                      u64 new_height, u64 new_width)
{
    CHECK_FATAL(!tensor, "null tensor");

    u64 old_total = TENSOR3D_TOTAL(tensor);
    u64 new_total = new_channels * new_height * new_width;

    CHECK_FATAL(old_total != new_total, "reshape: total size must match");

    tensor->channels = new_channels;
    tensor->height = new_height;
    tensor->width = new_width;
}


// PADDING OPERATIONS
// ============================================================================

Tensor3D* tensor3d_pad(const Tensor3D* input, u64 pad_h, u64 pad_w,
                       Arena* arena)
{
    CHECK_FATAL(!input || !arena, "null pointer");

    u64 new_height = input->height + (2 * pad_h);
    u64 new_width = input->width + (2 * pad_w);

    Tensor3D* padded = tensor3d_arena_zero(arena, input->channels,
                                          new_height, new_width);

    // Copy each channel with padding
    for (u64 c = 0; c < input->channels; c++) {
        for (u64 r = 0; r < input->height; r++) {
            for (u64 col = 0; col < input->width; col++) {
                float val = TENSOR3D_AT((Tensor3D*)input, c, r, col);
                TENSOR3D_AT(padded, c, r + pad_h, col + pad_w) = val;
            }
        }
    }

    return padded;
}

Tensor3D* tensor3d_unpad(const Tensor3D* input, u64 pad_h, u64 pad_w,
                         Arena* arena)
{
    CHECK_FATAL(!input || !arena, "null pointer");
    CHECK_FATAL(input->height < 2 * pad_h || input->width < 2 * pad_w,
                "padding larger than input");

    u64 new_height = input->height - (2 * pad_h);
    u64 new_width = input->width - (2 * pad_w);

    Tensor3D* unpadded = tensor3d_arena_alloc(arena, input->channels,
                                             new_height, new_width);

    // Copy inner region
    for (u64 c = 0; c < input->channels; c++) {
        for (u64 r = 0; r < new_height; r++) {
            for (u64 col = 0; col < new_width; col++) {
                float val = TENSOR3D_AT((Tensor3D*)input, c, r + pad_h, col + pad_w);
                TENSOR3D_AT(unpadded, c, r, col) = val;
            }
        }
    }

    return unpadded;
}


// UTILITIES
// ============================================================================

void tensor3d_print(const Tensor3D* tensor)
{
    CHECK_FATAL(!tensor, "null tensor");

    printf("Tensor3D: %lu × %lu × %lu\n", 
           tensor->channels, tensor->height, tensor->width);

    for (u64 c = 0; c < tensor->channels; c++) {
        printf("\nChannel %lu:\n", c);
        tensor3d_print_channel(tensor, c);
    }
}

void tensor3d_print_channel(const Tensor3D* tensor, u64 channel)
{
    CHECK_FATAL(!tensor, "null tensor");
    CHECK_FATAL(channel >= tensor->channels, "channel out of bounds");

    for (u64 r = 0; r < tensor->height; r++) {
        for (u64 c = 0; c < tensor->width; c++) {
            printf("%8.4f ", TENSOR3D_AT((Tensor3D*)tensor, channel, r, c));
        }
        printf("\n");
    }
}

void tensor4d_print(const Tensor4D* tensor)
{
    CHECK_FATAL(!tensor, "null tensor");

    printf("Tensor4D: %lu × %lu × %lu × %lu\n",
           tensor->out_channels, tensor->in_channels,
           tensor->height, tensor->width);

    for (u64 oc = 0; oc < tensor->out_channels; oc++) {
        printf("\nFilter %lu:\n", oc);
        tensor4d_print_filter(tensor, oc);
    }
}

void tensor4d_print_filter(const Tensor4D* tensor, u64 filter)
{
    CHECK_FATAL(!tensor, "null tensor");
    CHECK_FATAL(filter >= tensor->out_channels, "filter out of bounds");

    for (u64 ic = 0; ic < tensor->in_channels; ic++) {
        printf("  Input channel %lu:\n", ic);
        for (u64 r = 0; r < tensor->height; r++) {
            printf("    ");
            for (u64 c = 0; c < tensor->width; c++) {
                printf("%8.4f ", TENSOR4D_AT((Tensor4D*)tensor, filter, ic, r, c));
            }
            printf("\n");
        }
    }
}

b8 tensor3d_same_shape(const Tensor3D* a, const Tensor3D* b)
{
    return (a->channels == b->channels &&
            a->height == b->height &&
            a->width == b->width);
}

b8 tensor4d_same_shape(const Tensor4D* a, const Tensor4D* b)
{
    return (a->out_channels == b->out_channels &&
            a->in_channels == b->in_channels &&
            a->height == b->height &&
            a->width == b->width);
}


// CONVERSION BETWEEN MATRIX AND TENSOR
// ============================================================================

Tensor3D* matrix_to_tensor3d(const Matrixf* mat, Arena* arena)
{
    CHECK_FATAL(!mat || !arena, "null pointer");

    Tensor3D* tensor = tensor3d_arena_alloc(arena, 1, mat->m, mat->n);
    u64 total = (u64)mat->m * mat->n;
    memcpy(tensor->data, mat->data, sizeof(float) * total);

    return tensor;
}

Matrixf* tensor3d_to_matrix(const Tensor3D* tensor, Arena* arena)
{
    CHECK_FATAL(!tensor || !arena, "null pointer");
    CHECK_FATAL(tensor->channels != 1, "tensor must have exactly 1 channel");

    Matrixf* mat = matrix_arena_alloc(arena, tensor->height, tensor->width);
    u64 total = (u64)tensor->height * tensor->width;
    memcpy(mat->data, tensor->data, sizeof(float) * total);

    return mat;
}

Matrixf* tensor3d_channel_to_matrix(const Tensor3D* tensor, u64 channel,
                                   Arena* arena)
{
    CHECK_FATAL(!tensor || !arena, "null pointer");
    CHECK_FATAL(channel >= tensor->channels, "channel out of bounds");

    Matrixf* mat = matrix_arena_alloc(arena, tensor->height, tensor->width);
    float* channel_data = tensor3d_channel_ptr((Tensor3D*)tensor, channel);
    u64 total = (u64)tensor->height * tensor->width;
    memcpy(mat->data, channel_data, sizeof(float) * total);

    return mat;
}


// STATISTICS AND NORMALIZATION
// ============================================================================

float tensor3d_min(const Tensor3D* tensor)
{
    CHECK_FATAL(!tensor, "null tensor");

    u64 total = TENSOR3D_TOTAL(tensor);
    float min_val = tensor->data[0];

    for (u64 i = 1; i < total; i++) {
        if (tensor->data[i] < min_val) {
            min_val = tensor->data[i];
        }
    }

    return min_val;
}

float tensor3d_max(const Tensor3D* tensor)
{
    CHECK_FATAL(!tensor, "null tensor");

    u64 total = TENSOR3D_TOTAL(tensor);
    float max_val = tensor->data[0];

    for (u64 i = 1; i < total; i++) {
        if (tensor->data[i] > max_val) {
            max_val = tensor->data[i];
        }
    }

    return max_val;
}

float tensor3d_mean(const Tensor3D* tensor)
{
    CHECK_FATAL(!tensor, "null tensor");

    u64 total = TENSOR3D_TOTAL(tensor);
    float sum = 0.0f;

    for (u64 i = 0; i < total; i++) {
        sum += tensor->data[i];
    }

    return sum / (float)total;
}

float tensor3d_std(const Tensor3D* tensor)
{
    CHECK_FATAL(!tensor, "null tensor");

    float mean = tensor3d_mean(tensor);
    u64 total = TENSOR3D_TOTAL(tensor);
    float variance = 0.0f;

    for (u64 i = 0; i < total; i++) {
        float diff = tensor->data[i] - mean;
        variance += diff * diff;
    }

    variance /= (float)total;
    return fast_sqrt(variance);
}

void tensor3d_normalize(Tensor3D* tensor)
{
    CHECK_FATAL(!tensor, "null tensor");

    float mean = tensor3d_mean(tensor);
    float std = tensor3d_std(tensor);

    CHECK_FATAL(std == 0.0f, "standard deviation is zero");

    u64 total = TENSOR3D_TOTAL(tensor);
    for (u64 i = 0; i < total; i++) {
        tensor->data[i] = (tensor->data[i] - mean) / std;
    }
}

void tensor3d_normalize_minmax(Tensor3D* tensor)
{
    CHECK_FATAL(!tensor, "null tensor");

    float min_val = tensor3d_min(tensor);
    float max_val = tensor3d_max(tensor);

    float range = max_val - min_val;
    CHECK_FATAL(range == 0.0f, "range is zero");

    u64 total = TENSOR3D_TOTAL(tensor);
    for (u64 i = 0; i < total; i++) {
        tensor->data[i] = (tensor->data[i] - min_val) / range;
    }
}


