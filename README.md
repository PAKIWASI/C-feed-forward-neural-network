# Feed-Forward Neural Network in C

A from-scratch feedforward neural network in C for handwritten digit recognition (MNIST) and Fashion-MNIST. No ML frameworks, no math libraries — just C.

---


<video src="https://github.com/user-attachments/assets/77026732-7585-4e2d-8103-722a738ea422" controls></video>
---

## Features

- **Pure C** — no TensorFlow, PyTorch, NumPy, or math.h
- **Custom memory management** — arena allocators for fast, zero-fragmentation allocation
- **Two training modes** — Stochastic Gradient Descent and Mini-Batch GD
- **Correct weight initialization** — He init (ReLU layers), Xavier (Softmax output)
- **Model persistence** — save/load trained weights in a compact binary format
- **Fashion-MNIST support** — same pipeline, different dataset
- **Interactive GUI** — draw digits in real-time and watch predictions update (via raylib)
- **SIMD-friendly loops** — `restrict` pointers and row-major layout for auto-vectorization

---

## Project Structure

```
ffnn/
├── src/
│   ├── ffnn.c                  # Network: create, train, test, save/load
│   ├── layer.c                 # Forward pass, backprop, weight updates
│   ├── idx_file_reader.c       # Parses MNIST IDX binary format
│   ├── mnist_data_processor.c  # Converts IDX → custom .bin format
│   └── main.c                  # Entry point (edit to configure training)
│
├── include/
│   ├── ffnn.h                  # Public API
│   ├── layer.h                 # Layer struct and operations
│   ├── idx_file_reader.h
│   └── mnist_data_processor.h
│
├── external/
│   ├── C-Data-Structures-Lib/  # Arena, genVec, Matrix, String (WCtoolkit)
│   └── raylib/
│       ├── src/
│       │   ├── mnist_predictor.c   # Canvas, UI, prediction logic
│       │   └── ray_main.c          # GUI event loop
│       └── include/
│           └── mnist_predictor.h
│
├── tests/
│   ├── mnist_tests.h           # MNIST and Fashion-MNIST test helpers
│   └── xor_test.h              # XOR sanity check for the network
│
├── data/
│   ├── raw/                    # Original MNIST IDX files (you provide)
│   │   ├── train-images-idx3-ubyte
│   │   ├── train-labels-idx1-ubyte
│   │   ├── t10k-images-idx3-ubyte
│   │   └── t10k-labels-idx1-ubyte
│   ├── dataset.bin             # Converted MNIST training set (~47MB)
│   ├── testset.bin             # Converted MNIST test set (~7.8MB)
│   ├── fashion_mnist/
│   │   ├── raw/                # Fashion-MNIST IDX files
│   │   ├── fashion_train.bin
│   │   └── fashion_test.bin
│   ├── 128.bin                 # Saved weights: 784→128→10
│   └── 256.bin                 # Saved weights: 784→256→10
│
├── CMakeLists.txt
└── README.md
```

---

## Build

Requires **clang** and **CMake 3.20+**. raylib must be pre-built at `build/raylib/`.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

This produces two executables:
- `main` — CLI: train, test, convert datasets
- `gui` — Interactive raylib predictor

For a debug build with sanitizers (ASan, UBSan, LSan):
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

---

## Quick Start

### 1. Prepare the Dataset

Download the MNIST IDX files and place them in `data/raw/`. Then convert to the custom binary format by calling `mnist_prepare_from_idx` in `main.c`:

```c
// Convert training set
mnist_prepare_from_idx("data/raw/", "data/", true);

// Convert test set  
mnist_prepare_from_idx("data/raw/", "data/", false);
```

> **Note:** The directory path must have a trailing slash — `"data/raw/"` not `"data/raw"`.

The same function works for Fashion-MNIST — just point it at the Fashion-MNIST IDX files:

```c
mnist_prepare_from_idx("data/fashion_mnist/raw/", "data/fashion_mnist/", true);
mnist_prepare_from_idx("data/fashion_mnist/raw/", "data/fashion_mnist/", false);
```

### 2. Train

Edit `main.c` and run `./main`:

```c
pcg32_rand_seed(1234, 1);   // seed for reproducibility

ffnn* net = ffnn_create(
    (u16[3]){784, 256, 10}, // architecture: input → hidden → output
    3,                       // number of layers
    0.015f,                  // learning rate
    "data/dataset.bin"       // training data
);

ffnn_train(net);             // SGD over the full 60k dataset

ffnn_save_parameters(net, "data/256.bin");

ffnn_set_dataset(net, "data/testset.bin");
ffnn_test(net);

ffnn_destroy(net);
```

### 3. Run the GUI

```bash
./gui
```

The GUI loads `data/256.bin` by default (configurable in `ray_main.c`). Draw a digit on the canvas and the network predicts in real time when you release the mouse.

**Controls:**

| Key / Input | Action |
|---|---|
| Left Mouse | Draw |
| C | Clear canvas |
| S | Save canvas as `.raw` file |
| `+` / `-` | Increase / decrease brush size |
| ESC | Exit |

---

## API Reference

### Network

```c
// Create a new network and load training data
ffnn* ffnn_create(u16* layer_sizes, u8 num_layers,
                  float learning_rate, const char* mnist_path);

// Load a previously saved network (weights only, no dataset)
ffnn* ffnn_create_trained(const char* saved_path);

// Swap the loaded dataset (e.g. switch train → test)
void ffnn_set_dataset(ffnn* net, const char* dataset_path);

void ffnn_destroy(ffnn* net);
```

### Training

```c
// SGD: one weight update per sample — 60,000 updates per pass
void ffnn_train(ffnn* net);

// Mini-batch GD: accumulate gradients over batch_size samples, then update
// batch_size: 16–64 recommended. num_epochs: passes over the full dataset.
void ffnn_train_batch(ffnn* net, u16 batch_size, u16 num_epochs);
```

### Evaluation and Persistence

```c
// Print accuracy on the currently loaded dataset
void ffnn_test(ffnn* net);

// Save weights and biases to a compact binary file
b8 ffnn_save_parameters(const ffnn* net, const char* outfile);
```

### Data Preparation

```c
// Convert MNIST/Fashion-MNIST IDX files to the custom .bin format
// use_train=true → 60k training set, use_train=false → 10k test set
// data_dir must have a trailing slash
b8 mnist_prepare_from_idx(const char* data_dir, const char* out_dir, b8 use_train);

// Load a .bin file into an mnist_dataset struct
b8 mnist_load_custom_file(mnist_dataset* set, const char* filepath, Arena* arena);
```

---

## Network Architecture

### Layer

Each layer stores everything needed for both forward and backward passes:

```c
typedef struct Layer {
    float*  x;      // input pointer (1×m) — points to prev layer's output
    float*  b;      // biases (1×n)
    Matrixf W;      // weights (n×m) — row-major for cache efficiency
    float*  z;      // pre-activation: z = xW + b
    float*  a;      // activation: a = f(z)

    Matrixf dL_dW;  // weight gradients (n×m)
    float*  dL_dz;  // pre-activation gradients (1×n)
    float*  dL_dx;  // input gradients (1×m) — passed to previous layer

    u16 m;          // input size
    u16 n;          // output size
    b8  is_output_layer;
    Matrixf W_T;    // cached transpose for backprop
} Layer;
```

### Activations

**Hidden layers — ReLU:**
```
f(z)  = max(0, z)
f'(z) = 1 if z ≥ 0, else 0
```

**Output layer — Softmax** (numerically stable, subtracts max before exp):
```
f(z_i) = exp(z_i - max_z) / Σ exp(z_j - max_z)
```

**Loss — Cross-Entropy:**
```
L = -Σ y_i × log(p_i)
```

The softmax + cross-entropy derivative simplifies beautifully:
```
dL/dz_i = p_i - y_i
```

### Weight Initialization

| Layer Type | Method | Formula |
|---|---|---|
| Hidden (ReLU) | He | `σ = sqrt(2 / input_size)`, `W ~ N(0, σ²)` |
| Output (Softmax) | Xavier | `limit = sqrt(6 / (in + out))`, `W ~ Uniform(-limit, +limit)` |

---

## Custom Binary Format

All datasets are stored in a simple format for fast loading — no parsing, just one `fread`:

```
Header (4 bytes):
  [num_images : u16]
  [width      : u8 ]   always 28
  [height     : u8 ]   always 28

Per sample (785 bytes):
  [label : u8      ]   0–9
  [pixels: u8 × 784]   row-major, 0–255
```

| Dataset | Size |
|---|---|
| MNIST train | 47,100,004 bytes (~45 MB) |
| MNIST test  | 7,850,004 bytes (~7.5 MB) |

### Saved Model Format

```
[num_layers : u64]
For each layer:
  [input_size  : u16]
  [output_size : u16]
  [weights     : f32 × (n×m)]
  [biases      : f32 × n    ]
```

---

## Performance

### Memory

Two arenas are used — one for the network, one for the dataset:

```c
Arena* main_arena;     // layers, weights, biases — typically 1–5 MB
Arena* dataset_arena;  // image data — ~47 MB for MNIST train
```

Arena allocation is a pointer bump — O(1) with no fragmentation. Cleanup frees the entire arena in one call.

### Compute

The forward and backward pass inner loops use `restrict` pointers and row-major layout to enable auto-vectorization (SIMD). With `-march=native -O3`, the compiler emits vectorized code for the matrix-vector multiply loops.

The W_T (transpose) matrix is pre-allocated and computed once per backward pass rather than reallocated each time.

---

## Results

| Architecture | Training | Test Accuracy |
|---|---|---|
| 784→128→10 | SGD, lr=0.015 | ~95.5% |
| 784→256→10 | SGD, lr=0.015 | **96.7%** |

Fashion-MNIST is a harder problem (10 clothing categories vs. handwritten digits) and achieves lower accuracy with the same architecture.

---

## Limitations

- No dropout or batch normalization
- No GPU acceleration
- No convolutional layers (limits Fashion-MNIST ceiling)
- Mini-batch training currently achieves lower accuracy than SGD — likely a learning rate tuning issue (batch training requires a higher lr than SGD)
- The live predictor draws at 25× scale; real handwriting differs from MNIST's centered, anti-aliased digits, which can affect prediction quality

---

## Dependencies

**Core:** Standard C library only (`stdio`, `stdlib`, `string`)

**WCtoolkit** (`external/C-Data-Structures-Lib`):
- `Arena` — linear memory allocator
- `genVec` — generic dynamic array
- `Matrixf` — 2D float array wrapper
- `String` — dynamic string with append/remove

**GUI only:** [raylib](https://www.raylib.com/) for window, drawing, and input

---

## License

MIT — see `LICENSE`.
