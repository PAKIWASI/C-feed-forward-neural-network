#ifndef FFNN_H
#define FFNN_H

#include "gen_vector.h"
#include "mnist_data_processor.h"




typedef struct {
    genVec        layers;        // of type layer
    float         learning_rate; // step size in gradient decent
    Arena*        main_arena;    // arena to allocate layers, weights, biases
    Arena*        dataset_arena; // arena for the training/testing datasets
    mnist_dataset set;           // grey scale (0-255) u8 pixel images and u8 (0-9) labels (arena allocated)
} ffnn;



ffnn* ffnn_create(u16* layer_sizes, u8 num_layers, float learning_rate, const char* mnist_path);
ffnn* ffnn_create_trained(const char* saved_path, const char* testing_set);
void ffnn_destroy(ffnn* net);



/* Scorcastic Gradient Decent
Dataset: [img0, img1, img2, ..., img59999]
         
Epoch 1:
  img0 → Forward → Backward → Update Weights (W₀ → W₁)
  img1 → Forward → Backward → Update Weights (W₁ → W₂)
  img2 → Forward → Backward → Update Weights (W₂ → W₃)
  ...
  img59999 → Forward → Backward → Update Weights (W₅₉₉₉₉ → W₆₀₀₀₀)

Total Updates: 60,000
*/
void ffnn_train(ffnn* net);


/* Mini Batch Gradient Decent (batch size = 32)
Dataset: [img0, img1, img2, ..., img59999]
Shuffle: [img42, img1005, img73, ..., img19]

Epoch 1:
  Batch 0: [img42, img1005, ..., img891]  (32 images)
    ┌─────────────────────────────────────┐
    │ img42  → Forward → Backward         │
    │           ↓          ↓              │
    │       activations  gradients        │
    │                      ↓              │
    │                  accumulate         │
    │                                     │
    │ img1005 → Forward → Backward        │
    │           ↓          ↓              │
    │       activations  gradients        │
    │                      ↓              │
    │                  accumulate         │
    │                                     │
    │ ... (30 more images)                │
    │                                     │
    │ Average Gradients (÷32)             │
    │ Update Weights (W₀ → W₁)            │
    └─────────────────────────────────────┘

  Batch 1: [img2314, img88, ..., img4521]
    └─→ Same process → (W₁ → W₂)

  ... (1873 more batches)

  Batch 1874: [last 32 images]
    └─→ Same process → (W₁₈₇₄ → W₁₈₇₅)

Total Updates: 1,875
*/                              // size of each batch       // number of passes to do (on complete set)
void ffnn_train_batch_epochs(ffnn* net, u16 batch_size, u16 num_epochs);


void ffnn_test(ffnn* net);

void ffnn_change_dataset(ffnn* net, const char* dataset_path);
b8   ffnn_save_parameters(const ffnn* net, const char* outfile);


#define LEARN_DECAY_RATE 0.9995f
#define LEARN_DECAY_AFTER 1000
/*

   Finding the Min of the cost function

   High Learning Rate (early training):
   - Large steps → Fast progress down the slope
   - But might overshoot the bottom and bounce around

   Low Learning Rate (late training):
   - Small steps → Slow but precise
   - Can settle into the exact Bottom

*/

#endif // FFNN_H
