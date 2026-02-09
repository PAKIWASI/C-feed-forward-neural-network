

#include "mnist_data_processor.h"



int main(void)
{
    String data_dir; 
    string_create_stk(&data_dir, "/home/wasi/Documents/projects/c/ffnn/data/");
    
    b8 res = mnist_load_from_idx(&data_dir);

    string_destroy_stk(&data_dir);

    return res;
}

