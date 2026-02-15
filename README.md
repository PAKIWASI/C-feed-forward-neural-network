# Feed-Forward Neural Network (FFNN) for MNIST

A high-performance, from-scratch implementation of a feed forward neural network in C for handwritten digit recognition using the MNIST dataset.
No Libraries (Not even Math)

## Overview

This project implements a complete neural network training and inference system with:
**WCtoolkit** My C toolkit for data structures, arenas, rng and memory management
**Optimized forward and backward propagation**
**Multiple training algorithms** (SGD and Mini-Batch GD)
**Real-time GUI predictor** for interactive digit recognition
**Clean, modular architecture** with no external ML libraries

### Key Features

**Pure C Implementation** - No TensorFlow, PyTorch, or ML frameworks  
**Memory Efficient** - Custom arena allocators for fast allocation/deallocation  
**Multiple Activation Functions** - ReLU (hidden) and Softmax (output)  
**Batch Training** - Mini-batch gradient descent with shuffling  
**Weight Initialization** - He initialization (ReLU) and Xavier (Softmax)  
**Model Persistence** - Save/load trained weights  
**Interactive GUI** - Real-time drawing and prediction with raylib  
**MNIST Support** - Custom format for fast loading  

---

## Project Structure

```
ffnn/
├── src/
│   ├── ffnn.c                    # Neural network implementation
│   ├── layer.c                   # Layer operations (forward/backward)
│   ├── idx_file_reader.c         # IDX format parser (MNIST)
│   ├── mnist_data_processor.c    # Data preprocessing and loading
│   └── main.c                    # CLI training program
│
├── include/
│   ├── ffnn.h                    # Network API
│   ├── layer.h                   # Layer structures
│   ├── mnist_data_processor.h    # Data processing API
│   └── common.h                  # Common types and macros
│
├── external/
│   ├── C-Data-Structures-Lib/    # Vector, arena allocator
│   └── raylib/                   # GUI predictor files
│       ├── mnist_live_predictor.h
│       ├── mnist_live_predictor_impl.c
│       └── live_predictor_main.c
│
├── data/
│   ├── dataset.bin               # Training set (custom format)
│   ├── testset.bin              # Test set (custom format)
│   └── trained_model.bin        # Saved weights and biases
│
├── CMakeLists.txt
└── README.md
```

---

## Quick Start

```

### Prepare MNIST Dataset

Download the MNIST dataset and convert to custom format:

```c
// Convert IDX files to custom binary format
mnist_prepare_from_idx(
    "/path/to/mnist/",     // Directory with IDX files
    "/path/to/output/"     // Output directory for .bin files
);
```

**MNIST IDX files needed:**
- `train-images-idx3-ubyte` (60,000 training images)
- `train-labels-idx1-ubyte` (60,000 training labels)
- `t10k-images-idx3-ubyte` (10,000 test images)
- `t10k-labels-idx1-ubyte` (10,000 test labels)

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

This creates two executables:
- `main` - CLI training/testing program
- `live_predictor` - Interactive GUI

### Train a Model

```c
// In main.c
pcg32_rand_seed(1234, 1);  // Seed RNG for reproducibility

ffnn* net = ffnn_create(
    (u16[4]){784, 128, 64, 10},  // Architecture: 784→128→64→10
    4,                            // Number of layers (includes connections)
    0.01f,                        // Learning rate
    "data/dataset.bin"            // Training data path
);

// Train with mini-batch gradient descent
ffnn_train_batch_epochs(net, 32, 5);  // 32 batch size, 5 epochs

// Save trained parameters
ffnn_save_parameters(net, "data/trained_model.bin");

// Test on test set
ffnn_change_dataset(net, "data/testset.bin");
ffnn_test(net);

ffnn_destroy(net);
```

**Example Output:**
```
=== Epoch 1/5 ===
    Batch 100/1875 (89.34%)
    Batch 200/1875 (91.22%)
    ...
Epoch 1 Complete: Accuracy = 93.45% (56070/60000)

=== Epoch 5/5 ===
Epoch 5 Complete: Accuracy = 97.82% (58692/60000)
  New best accuracy!

=== Training Complete ===
Best Accuracy: 97.82%

Testing Accuracy: 96.24% (9624/10000)
```

### Interactive Prediction (GUI)

```bash
./live_predictor data/trained_model.bin
```

**Controls:**
- **Left Mouse** - Draw digit
- **C** - Clear canvas
- **S** - Save drawing as .raw file
- **+/-** - Adjust brush size
- **ESC** - Exit

---

## Network Architecture

### Layer Structure

Each layer contains:

```c
typedef struct Layer {
    // Forward pass
    float* x;      // Input (1 × m)
    Matrix W;      // Weights (n × m)
    float* b;      // Biases (1 × n)
    float* z;      // Pre-activation (1 × n): z = xW + b
    float* a;      // Activation (1 × n): a = f(z)
    
    // Backward pass (gradients)
    Matrix dL_dW;  // Weight gradients (n × m)
    float* dL_dz;  // Pre-activation gradients (1 × n)
    float* dL_dx;  // Input gradients (1 × m) - passed to prev layer
    
    // Dimensions
    u16 m;         // Input size
    u16 n;         // Output size
    
    bool is_output_layer;
} Layer;
```

### Activation Functions

**Hidden Layers - ReLU:**
```
f(z) = max(0, z)
f'(z) = 1 if z > 0, else 0
```

**Output Layer - Softmax:**
```
f(z_i) = exp(z_i) / Σ exp(z_j)
```

**Loss Function - Cross-Entropy:**
```
L = -Σ y_i × log(p_i)
where y_i = true label (one-hot), p_i = prediction
```

### Weight Initialization

**He Initialization** (Hidden Layers with ReLU):
```c
σ = sqrt(2 / input_size)
W ~ N(0, σ²)
```
Compensates for ReLU killing ~50% of neurons.

**Xavier Initialization** (Output Layer with Softmax):
```c
limit = sqrt(6 / (input_size + output_size))
W ~ Uniform(-limit, +limit)
```
Maintains variance through both forward and backward passes.

---

## Training Algorithms

### Stochastic Gradient Descent (SGD)

```c
void ffnn_train(ffnn* net);
```

Updates weights after **every single sample**:
```
For each sample in dataset:
    1. Forward pass → get prediction
    2. Backward pass → compute gradients
    3. Update weights: W = W - α × ∇L
```

**60,000 samples = 60,000 weight updates per epoch**

**Pros:** Fast convergence on small batches  
**Cons:** Noisy updates, may oscillate  

### Mini-Batch Gradient Descent

```c
void ffnn_train_batch_epochs(ffnn* net, u16 batch_size, u16 num_epochs);
```

Accumulates gradients over **batch_size samples**, then updates:
```
For each epoch:
    Shuffle dataset
    For each batch of 32 samples:
        1. Forward pass on all 32
        2. Accumulate gradients: ∇L_total += ∇L_sample
        3. Average: ∇L_avg = ∇L_total / 32
        4. Update weights: W = W - α × ∇L_avg
```

**60,000 samples ÷ 32 = 1,875 weight updates per epoch**

**Pros:** Smoother updates, better generalization, GPU-friendly  
**Cons:** Slower per-epoch than SGD  

---

## API Reference

### Network Management

```c
// Create and initialize network
ffnn* ffnn_create(u16* layer_sizes, u8 num_layers, 
                  float learning_rate, const char* mnist_path);

// Load pre-trained network
ffnn* ffnn_create_trained(const char* saved_path, const char* testing_set);

// Clean up
void ffnn_destroy(ffnn* net);
```

### Training

```c
// Train with SGD (one sample at a time)
void ffnn_train(ffnn* net);

// Train with mini-batch GD
void ffnn_train_batch_epochs(ffnn* net, u16 batch_size, u16 num_epochs);
```

### Testing and Inference

```c
// Test accuracy on current dataset
void ffnn_test(ffnn* net);

// Switch datasets (e.g., train → test)
void ffnn_change_dataset(ffnn* net, const char* dataset_path);
```

### Model Persistence

```c
// Save trained weights and biases
bool ffnn_save_parameters(const ffnn* net, const char* outfile);

// Binary format:
// [num_layers: u64]
// For each layer:
//   [input_size: u16][output_size: u16]
//   [weights: float[n×m]][biases: float[n]]
```

### Layer Operations

```c
// Forward propagation
void layer_calc_output(Layer* layer, const float* x);
// Computes: z = xW + b, a = f(z)

// Backward propagation
void layer_calc_deriv(Layer* layer, const float* dL_da);
// Computes: dL/dW, dL/db, dL/dx

// Update parameters
void layer_update_WB(Layer* layer, float learning_rate);
// Updates: W = W - α×dL/dW, b = b - α×dL/db
```

---

## Live Predictor GUI

### Architecture

The GUI is built with **raylib** and consists of 3 files:

1. **`mnist_predictor.h`** - Definitions and declarations
2. **`mnist_predictor.c`** - Canvas, UI, and prediction logic
3. **`ray_main.c`** - Main event loop

### Features

- **700×700 drawing canvas** (28×28 pixels scaled 25×)
- **Gaussian brush** for smooth, natural strokes
- **Real-time predictions** update when mouse is released
- **Probability visualization** for all 10 digits (0-9)
- **Adjustable brush size** (0.5 - 5.0)
- **Save drawings** as raw 28×28 grayscale files


The interface displays:
- **Left side:** Drawing canvas with grid
- **Right side:** Prediction panel with:
  - Large predicted digit
  - Confidence percentage
  - Probability bars for all digits (0-9)
- **Bottom:** Controls and status

### Implementation Details

**Prediction Pipeline:**
```
1. Canvas (28×28 u8 array, values 0-255)
   ↓
2. Normalize: pixel / 255.0 → [0.0, 1.0]
   ↓
3. Forward pass through network
   ↓
4. Softmax output → probabilities for each digit
   ↓
5. Display: Find max probability, update UI
```

**Performance:**
- Forward pass: ~1ms (784→128→64→10 network)
- Predictions update only when mouse is released (prevents lag)
- Canvas rendering: 60 FPS

---

## Data Format

### Custom Binary Format

Efficient format for fast loading (no parsing overhead):

```
Header (4 bytes):
  [num_images: u16]  - Number of images in file
  [width: u8]        - Image width (28)
  [height: u8]       - Image height (28)

Data (785 bytes per sample):
  [label: u8]        - Digit label (0-9)
  [pixels: u8[784]]  - Grayscale pixels (row-major)
```

**Total size:**
- Training: 60,000 × 785 + 4 = **47,100,004 bytes** (~45 MB)
- Testing: 10,000 × 785 + 4 = **7,850,004 bytes** (~7.5 MB)

### Converting IDX to Custom Format

```c
bool mnist_prepare_from_idx(const char* data_dir, const char* out_dir);
```

Reads MNIST IDX files and outputs `label_img.bin`:

**Input files:**
- `train-images-idx3-ubyte` 
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

**Output:**
- `dataset.bin` (training)
- `testset.bin` (testing)

---

## Performance Optimizations

### Memory Management

**Arena Allocators:**
```c
Arena* main_arena = arena_create(nMB(5));        // Layers, weights, biases
Arena* dataset_arena = arena_create(nMB(47));    // MNIST data
```

Benefits:
- **No fragmentation** - Linear allocation
- **Fast deallocation** - Free entire arena at once
- **Cache-friendly** - Contiguous memory layout

### Cache Optimization

**Row-major matrix operations:**
```c
// Weight matrix stored as (n × m) for row-wise access
for (u16 i = 0; i < n; i++) {           // Each output neuron
    z[i] = 0.0f;
    for (u16 j = 0; j < m; j++) {       // Sequential memory access
        z[i] += x[j] * W[i][j];
    }
    z[i] += b[i];
}
```

**Matrix transposition for backprop:**
```c
// Pre-compute transpose once, then access row-wise
matrix_T(&W_T, &W);  // Transpose (n×m) → (m×n)

for (u16 i = 0; i < m; i++) {
    dL_dx[i] = 0.0f;
    for (u16 j = 0; j < n; j++) {
        dL_dx[i] += dL_dz[j] * W_T[i][j];  // Sequential access
    }
}
```

### Numerical Stability

**Softmax overflow prevention:**
```c
// Subtract max before exp() to prevent overflow
float max_z = max(z);
for (int i = 0; i < n; i++) {
    a[i] = exp(z[i] - max_z);  // Safe from overflow
    sum += a[i];
}
for (int i = 0; i < n; i++) {
    a[i] /= sum;  // Normalize
}
```

### Fast Math

Custom implementations for better performance:
```c
float fast_exp(float x);    // Approximation of e^x
float fast_sqrt(float x);   // Fast inverse square root
```

---

## Testing and Validation

### Training Metrics

During training, the system tracks:
- **Epoch accuracy** - Percentage of correct predictions
- **Batch progress** - Every 100 batches
- **Best accuracy** - Highest accuracy achieved

### Test Metrics

```c
void ffnn_test(ffnn* net);
```

Outputs:
- **Overall accuracy** - % correct on test set
- **Per-digit accuracy** - Confusion matrix insights
- **Sample-by-sample results** (optional debug mode)

### Example Results

**Typical performance:**
```
Network: 784 → 128 → 64 → 10
Training: 5 epochs, batch size 32
Learning rate: 0.01

Training Accuracy: 97.82%
Test Accuracy: 96.24%
```

**Per-digit breakdown:**
```
Digit 0: 98.4% (964/980)
Digit 1: 98.9% (1123/1135)
Digit 2: 95.3% (983/1032)
Digit 3: 96.1% (970/1010)
Digit 4: 96.4% (947/982)
Digit 5: 95.7% (854/892)
Digit 6: 97.3% (932/958)
Digit 7: 95.8% (985/1028)
Digit 8: 94.8% (923/974)
Digit 9: 95.5% (964/1009)
```

---

## Mathematical Foundation

### Forward Propagation

For layer *l*:
```
z^(l) = x^(l) W^(l) + b^(l)
a^(l) = f(z^(l))
x^(l+1) = a^(l)
```

### Backpropagation

**Output layer (softmax + cross-entropy):**
```
dL/dz = a - y  (simplified derivative)
```

**Hidden layers (ReLU):**
```
dL/dz = dL/da ⊙ f'(z)
where f'(z) = 1 if z > 0, else 0
```

**Gradient computation:**
```
dL/dW = (dL/dz)^T × x       (n×m matrix)
dL/db = dL/dz                (n vector)
dL/dx = dL/dz × W            (m vector, passed downstream)
```

**Weight update:**
```
W = W - α × dL/dW
b = b - α × dL/db
```

### Batch Gradient Averaging

For mini-batch of size *B*:
```
∇L_avg = (1/B) Σ ∇L_i

W = W - α × ∇L_avg
```

---

## Configuration and Tuning

### Hyperparameters

**Learning rate:**
```c
#define DEFAULT_LEARNING_RATE 0.01f

// Optional decay (currently commented out)
#define LEARN_DECAY_RATE 0.9995f
#define LEARN_DECAY_AFTER 1000
```

**Network architecture:**
```c
// Recommended: 2-3 hidden layers
u16 layer_sizes[] = {784, 128, 64, 10};

// Larger for more capacity:
u16 layer_sizes[] = {784, 256, 128, 10};

// Smaller for faster training:
u16 layer_sizes[] = {784, 64, 10};
```

**Batch training:**
```c
// Batch size: 16-64 (powers of 2)
// Epochs: 5-10 for MNIST
ffnn_train_batch_epochs(net, 32, 5);
```

### Tuning Tips

1. **Too low accuracy (<90%):**
   - Increase network size (more neurons/layers)
   - Train for more epochs
   - Check data preprocessing

2. **Overfitting (train >> test accuracy):**
   - Reduce network size
   - Use larger batch sizes
   - Implement dropout (future work)

3. **Training too slow:**
   - Increase learning rate (0.01 → 0.05)
   - Use larger batch sizes (32 → 64)
   - Reduce network size

4. **Training unstable:**
   - Decrease learning rate (0.01 → 0.001)
   - Use smaller batch sizes
   - Check weight initialization

---

## Dependencies

### Core Libraries

- **Standard C Library** - stdio, stdlib, string
- **WCtoolkit** - Custom implementations:
  - `Arena` - Memory allocator
  - `genVec` - Generic vector (dynamic array)
  - `Matrix` - 2D array wrapper
  - `String` - Dynamic string

### GUI (Optional)

- **raylib** - Graphics and input handling
  - Window management
  - Drawing primitives
  - Mouse/keyboard input

---

## Limitations and Future Work

### Current Limitations

- Can't really make 100% working MNIST-like images in the gui
- No dropout regularization
- No batch normalization
- No GPU acceleration
- Fixed to MNIST format only
- No convolutional layers

---

## Contributing

This is a learning/educational project demonstrating neural networks from scratch in C. Feel free to:

- Report bugs or issues
- Suggest optimizations
- Propose new features
- Submit pull requests

---

## References

### Datasets (included in repo)

- **MNIST Database** - Yann LeCun et al.
  - http://yann.lecun.com/exdb/mnist/

### Libraries Used

- **raylib** - https://www.raylib.com/

---

## License

MIT

---
