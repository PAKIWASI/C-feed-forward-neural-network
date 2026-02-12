#include "layer.h"
#include "arena.h"
#include "matrix.h"
#include "random.h"
#include "fast_math.h"


Layer* create_common_layer(Arena* arena, u16 m, u16 n);
void hidden_layer_init_WB(Layer* layer);
void output_layer_init_WB(Layer* layer);

// cached matrix (transpose of W), we allocate memory for it once
Matrix mat_T = {0};


Layer* layer_create_hidden(Arena* arena, u16 m, u16 n)
{
    LOG("creating hidden layer: %u input, %u output", m, n);

    Layer* hl = create_common_layer(arena, m, n);

    hl->is_output_layer = false;
    // hl->activation_fn = relu_activate;
    // hl->act_deriv_fn = relu_deriv;

    return hl;
}


Layer* layer_create_output(Arena* arena, u16 m, u16 n)
{
    LOG("creating output layer: %u input, %u output", m, n);

    Layer* ol = create_common_layer(arena, m, n);

    ol->is_output_layer = true;
    // ol->activation_fn = softmax_activate;
    // ol->act_deriv_fn = softmax_crossentropy_deriv;

    return ol;
}



// this assumes user already inited rng
// we have different init methods for hidden, output
void layer_init_weights_biases(Layer* layer)
{       
    if (!layer->is_output_layer) {
        LOG("Using HE Init for Hidden Layer");
        hidden_layer_init_WB(layer);
    } else {
        LOG("Using Xavior Init for Output Layer");
        output_layer_init_WB(layer);
    }
}


/*
    Forward Propogation
    
    z = x * W + b;

    if input is m and curr layer size is n, then weight matrix is n x m
    so x * W becomes (1 x m) * (n x m), but we will not do a standard xply
    x gets xplied to every row of W (n rows), each row has a bias (b (1xn))

    This is more cache friendly
*/
void layer_forward(Layer* layer, const float* x)
{
    for (u16 i = 0; i < layer->n; i++) {
        layer->z[i] = 0.0f;
        for (u16 j = 0; j < layer->m; j++) {
            layer->z[i] += x[j] * MATRIX_AT(&layer->W, i, j);
        }
        layer->z[i] += layer->b[i];
    }
}

/*
    Backward Propogation
    
    We get an upstream gradient dL_da

    dL_dz = dL_da * da_dz
    da_dz = f'(z)
    dL_dz = dL_da * f'(z)

    need : dL_dW, dL_db, dL_dx
    we do the derivation of dz_dW, dz_db, dz_dx

    dL_dW = dL_dz * dz_dW = dL_dz * x   (1 x n)T * (1 x m) = (n x m)
    dL_db = dL_dz * dz_db = dL_dz * 1   (1 x n)
    dL_dx = dL_dz * dz_dx = dL_dz * W   (1 x n) * (n x m) = (1 x m) - Send downstream
*/
void layer_backward(Layer* layer, const float* dL_da)
{
        // get dL_dz
    if (!layer->is_output_layer) {
        relu_deriv(layer->z, dL_da, layer->dL_dz, layer->n);
    }
    else {
        // for output layer, dL_da passed is actually the true label array
        softmax_crossentropy_deriv(layer->z, dL_da, layer->dL_dz, layer->n);
    }

    // dL_dW
    for (u16 i = 0; i < layer->n; i++) {

        for (u16 j = 0; j < layer->m; j++) {
            // matrix, arrays accessed row wise - good for cache
            MATRIX_AT(&layer->dL_dW, i, j) = layer->dL_dz[i] * layer->x[j];
        }
    }

    //dL_db = dL_dz

    // dL_dx
    // transpose the matrix for cache friendly access
    matrix_T(&mat_T, &layer->W);    // mat is m x n

    for (u16 i = 0; i < layer->m; i++) {
        for (u16 j = 0; j < layer->n; j++) {
            // matrix, arrays accessed row wise - good for cache
            layer->dL_dx[i] += layer->dL_dz[j] * MATRIX_AT(&mat_T, i, j);
        }
    }
}


void layer_update_weights(Layer* layer)
{
    // Update W using dL_dW
    u64 total = (u64)layer->m * layer->n;

    for (u64 i = 0; i < total; i++) {
        layer->W.data[i] += -1 * LEARNING_RATE * layer->dL_dW.data[i];
    }

    // Update biases using dL_db = dL_dz

    for (u16 i = 0; i < layer->n; i++) {
        layer->b[i] += -1 * LEARNING_RATE * layer->dL_dz[i];
    }
}


// Use Leaky ReLu?
void relu_activate(const float* z, float* a, u16 size)
{
    for (u16 i = 0; i < size; i++) {
        a[i] = z[i] >= 0.0f ? z[i] : 0.0f;
    }
}

void relu_deriv(const float* z, const float* dL_da, float* dL_dz, u16 size)
{
    // for (u16 i = 0; i < size; i++) {
    //     da_dz[i] = z[i] >= 0 ? 1.0f : 0.0f;
    // }

    for (u16 i = 0; i < size; i++) {
        // dL_dz[i] = dL_da[i] * ((z[i] >= 0) ? 1.0f : 0.0f);
        dL_dz[i] = (z[i] >= 0) ? dL_da[i] : 0;  // jacobian
    }
}

void softmax_activate(const float* z, float* a, u16 size)
{
    // Numerical stability: subtract max to prevent overflow
    // Prevents overflow when z values are large
    float max_z = z[0];
    for (u16 i = 1; i < size; i++) {
        if (z[i] > max_z) { max_z = z[i]; }
    }
    
    // Compute exp(z - max) and sum
    float sum = 0;
    for (u16 i = 0; i < size; i++) {
        a[i] = fast_exp(z[i] - max_z);  // Store exp values
        sum += a[i];
    }
    
    // Normalize
    for (u16 i = 0; i < size; i++) {
        a[i] /= sum;
    }
}

/*
Cross-Entropy Loss:
L = -Σ y_i × log(p_i)
where:
- y_i = true label (one-hot encoded)
- p_i = predicted probability (softmax output)

When computing dL/dz , the derivative simplifies remarkably:

dL/dz_i = p_i - y_i
*/
void softmax_crossentropy_deriv(const float* predicted, 
                const float* true_label, float* dL_dz, u16 size)
{
    for (u16 i = 0; i < size; i++) {
        dL_dz[i] = predicted[i] - true_label[i];
    }
}


// Private

Layer* create_common_layer(Arena* arena, u16 m, u16 n)
{
    CHECK_FATAL(!arena, "arena is null");
    CHECK_FATAL(m == 0, "input size can't be zero");
    CHECK_FATAL(n == 0, "output size can't be zero");

    u64 mark = arena_get_mark(arena);

    Layer* l = ARENA_ALLOC(arena, Layer);
     
    l->x = NULL;   // will be set layer

    // init to all zeros
    l->b = ARENA_ALLOC_ZERO_N(arena, float, n);

    l->W.m = n; 
    l->W.n = m; 
    l->W.data = ARENA_ALLOC_N(arena, float, (u64)n * m);

    l->z = ARENA_ALLOC_N(arena, float, n);
    l->a = ARENA_ALLOC_N(arena, float, n);

    l->dL_dW.m = n;
    l->dL_dW.n = m;
    l->dL_dW.data = ARENA_ALLOC_N(arena, float, (u64)n * m);

    // l->dL_db = (float*)ARENA_ALLOC_N(arena, float, n);
    l->dL_dz = ARENA_ALLOC_N(arena, float, n);
    l->dL_dx = ARENA_ALLOC_N(arena, float, m);

    l->m = m;
    l->n = n;

    LOG("allocated %lu for common layer", arena_used(arena) - mark);

    if (mat_T.data == NULL) {
        mat_T.m = m;
        mat_T.n = n;
        mat_T.data = ARENA_ALLOC_N(arena, float, (u64)n * m);
    }

    return l;
}


/* HE init for hidden layer (relu)
ReLU kills ~50% of neurons (all negative activations → 0)

Without compensation:
Initial variance → Forward pass → ReLU → Variance cut in half → fucked

With He (factor of 2):
σ = sqrt(2/n) compensates for the lost variance
Maintains stable gradient flow through the network
*/
void hidden_layer_init_WB(Layer* layer)
{
    // sqrt(2/input_size)
    float stddev = fast_sqrt(2.0f / (float)layer->m);

    u64 total = (u64)layer->m * layer->n;
    for (u64 i = 0; i < total; i++) {
        layer->W.data[i] = pcg32_rand_gaussian_custom(0.0f, stddev);
    }

    // biases set to 0 in constructor
}


/* Xavior Init for output layer (softmax)
no gradient killing like ReLU

Without proper scaling:
Too small → Gradients vanish → Learning stops
Too large → Gradients explode → Training unstable

Xavier balances input AND output dimensions:
σ = sqrt(6/(n_in + n_out)) keeps variance stable
Information flows equally well forward AND backward

Forward pass:  Var(output) ≈ Var(input)  → No amplification
Backward pass: Var(gradient) ≈ constant   → No vanishing/exploding
Maintains equilibrium through the entire network
*/
void output_layer_init_WB(Layer* layer)
{
    // range = sqrt(6 / (input_size + output_size))
    float limit = fast_sqrt(6.0f / (float)(layer->m + layer->n));
    
    u64 total = (u64)layer->m * layer->n;
    for (u64 i = 0; i < total; i++) {
        // Uniform in [-limit, +limit]
        layer->W.data[i] = pcg32_rand_float_range(-limit, limit);
    }
    // biases set to 0 in constructor
}


