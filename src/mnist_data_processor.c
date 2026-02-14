#include "mnist_data_processor.h"
#include "arena.h"
#include "idx_file_reader.h"
#include "String.h"




b8 mnist_load_from_idx(String* data_dir, idx_file** set, Arena* arena);
b8 mnist_save_custom_file(idx_file* img, idx_file* label, String* outdir);



// TODO: this function needes internal code changes to switch from train to test
// need to fix that
b8 mnist_prepare_from_idx(const char* data_dir, const char* out_dir)
{
    // Arena* arena = arena_create(MNIST_SIZE_IMG_TRAIN + MNIST_SIZE_LABEL_TRAIN + 1000);
    Arena* arena = arena_create(MNIST_SIZE_IMG_TEST + MNIST_SIZE_LABEL_TEST + 1000);
    LOG("arena alloced size: %lu\n", arena->size);

    // Read IDX input files (img + label), convert to correct byte order

    // allocate idx_file's for the train set
    // corresponds to files[] index
    idx_file* set[2] = {
        // train data idx
        (idx_file*)arena_alloc(arena, sizeof(idx_file)),
        // train label idx
        (idx_file*)arena_alloc(arena, sizeof(idx_file))
    };


    String* data_path = string_from_cstr(data_dir);
    // load data
    if (!mnist_load_from_idx(data_path, set, arena)) {
        LOG("could'nt load data from idx file");
        return false;
    }

    String* out_path = string_from_cstr(out_dir);
    // the set now contains data
    // save to custom format
    if (!mnist_save_custom_file(set[0], set[1], out_path)) {
        LOG("couldn't save data to out_dir");
        return false;
    }

    string_destroy(data_path);
    string_destroy(out_path);
    arena_release(arena);
    return true;
}


b8 mnist_load_custom_file(mnist_dataset* set, const char* filepath, Arena* arena)
{
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        LOG("couldn't open file %s", filepath);
        return false;
    }

    fread(&set->num_imgs, sizeof(u16), 1, f);
    fread(&set->img_w, sizeof(u8), 1, f);
    fread(&set->img_h, sizeof(u8), 1, f);
    
    CHECK_FATAL(set->img_w != set->img_h, "img dims differ");
    CHECK_FATAL(set->img_w != MNIST_IMG_DIM, "mnist img dim mismatch");

    LOG("Read num_imgs: %u", set->num_imgs);
    LOG("Read img_w: %u", set->img_w);
    LOG("Read img_h: %u", set->img_h);

    // allocate mem for num of imgs
    u64 m = arena_get_mark(arena);
    u64 size = set->num_imgs * (MNIST_IMG_SIZE + MNIST_LABEL_SIZE);
    set->data = arena_alloc(arena, size);
    LOG("arena allocated %lu", arena_used(arena) - m);

    // read all at once
    fread(set->data, sizeof(u8), size, f);

    fclose(f);
    return true;
}


void mnist_print_img(u8* data, u64 index)
{
    data += (index * (MNIST_IMG_SIZE + MNIST_LABEL_SIZE));
    printf("\n Label: %u\n\n", *data);

    for (u64 i = 0; i < MNIST_IMG_SIZE; i++) {
        printf("%d ", *(data + i + 1));  // TODO: format ? 

        if (i != 0 && i % MNIST_IMG_DIM == 0) {
            putchar('\n');
        }
    }
    putchar('\n');
}


// private functions

b8 mnist_load_from_idx(String* data_dir, idx_file** set, Arena* arena)
{
    CHECK_FATAL(!data_dir, "data_dir str is null");

    u64  path_str_len = string_len(data_dir);

    const char* files[4]     = {
        // training data, labels
        "train-images-idx3-ubyte", 
        "train-labels-idx1-ubyte",
        // testing data, labels
        "t10k-images-idx3-ubyte", 
        "t10k-labels-idx1-ubyte"
    };

    // only run for first 2 (train data + lables)
    for (u8 i = 2; i < 4; i++) {    // now running for testing

        // appending filename to directory str
        string_append_cstr(data_dir, files[i]);
        // convert to cstr temporarly
        string_append_char(data_dir, '\0');
        const char* full_path = string_data_ptr(data_dir);

        LOG("Reading file %s", full_path);

        if (!idx_read_file(set[i - 2], full_path, arena)) {
            LOG("Read Error file %s", full_path);
            return false;
        }

        // (DEBUG) Print first image
        print_hex(set[i - 2]->data, (u64)28 * 28, 28);

        // this also removes null term
        string_remove_range(data_dir, path_str_len, string_len(data_dir));
    }

    return true;
}

/* Custom Format
num imgs | width img | height img | label1 | img1 |label2 | img2 |......
    |         |           |           |       |
2 bytes     1 byte      1 byte      1 byte  784 bytes

Total for MNIST TRAIN

2 + 1 + 1 + 60000 * (1 + 784) = 47100004 bytes
*/
b8 mnist_save_custom_file(idx_file* img, idx_file* label, String* outdir)
{
    CHECK_FATAL(!img, "img idx_file is null");
    CHECK_FATAL(!label, "label idx_file is null");

    u16 num_imgs  = (u16)img->dim_sizes[0];
    u16 num_label = (u16)label->dim_sizes[0]; // only 1 dim size for label

    u8 img_w = (u8)img->dim_sizes[1];
    u8 img_h = (u8)img->dim_sizes[2];

    CHECK_FATAL(num_imgs != num_label, "no of imgs, labels don't match");
    CHECK_FATAL(img->dim_sizes[1] != img->dim_sizes[2], "mnist img width/heigh mismatch");
    CHECK_FATAL(img->dim_sizes[1] != MNIST_IMG_DIM, "mnist img dim mismatch");

    LOG("num imgs: %u | num labels: %u", num_imgs, num_label);
    LOG("img width: %u | img height: %u", img_w, img_h);

    // appending file name
    string_append_cstr(outdir, "label_img.bin");
    string_append_char(outdir, '\0');
    const char* full_path = string_data_ptr(outdir);

    FILE* f = fopen(full_path, "wb");
    if (!f) {
        LOG("Read Error file %s", full_path);
        return false;
    }
    LOG("opened file %s", full_path); 

    // write dimentions
    // 2 byte total number of images
    fwrite(&num_imgs, sizeof(u16), 1, f);
    // write 2 dims of each image ( 1  byte each)
    fwrite(&img_w, sizeof(u8), 1, f);
    fwrite(&img_h, sizeof(u8), 1, f);

    // write all data
    u64 img_offset   = 0; //MNIST_IMG_SIZE;
    u64 label_offset = 0; //MNIST_LABEL_SIZE;
    for (u64 i = 0; i < num_imgs; i++) {
        fwrite(label->data + label_offset, MNIST_LABEL_SIZE, 1, f);
        fwrite(img->data + img_offset, MNIST_IMG_SIZE, 1, f);

        label_offset += MNIST_LABEL_SIZE;
        img_offset += MNIST_IMG_SIZE;
    }


    fclose(f);
    return true;
}
