#include "layer.h"
#include "arena.h"
#include "common.h"
#include "matrix.h"
#include "random.h"


Layer* create_common_layer(Arena* arena, u16 m, u16 n);
void hidden_layer_init_WB(Layer* layer);
void output_layer_init_WB(Layer* layer);



Layer* layer_create_hidden(Arena* arena, u16 m, u16 n)
{
    LOG("creating hidden layer: %u input, %u output", m, n);

    Layer* hl = create_common_layer(arena, m, n);

    hl->is_output_layer = false;
    hl->activation_fn = relu_activate;
    hl->act_deriv.hl_deriv = relu_deriv;

    hl->da_dz = (float*)ARENA_ALLOC_N(arena, float, n);

    return hl;
}


Layer* layer_create_output(Arena* arena, u16 m, u16 n)
{
    LOG("creating output layer: %u input, %u output", m, n);

    Layer* ol = create_common_layer(arena, m, n);

    ol->is_output_layer = true;
    ol->activation_fn = softmax_activate;
    ol->act_deriv.ol_deriv = softmax_crossentropy_deriv;

    ol->da_dz = NULL;   // not needed for output layer 

    return ol;
}



// TODO: this assumes user already inited rng
//
// we have different init methods for hidden, output
void layer_init_weights_biases(Layer* layer)
{       
    CHECK_FATAL(!layer, "layer is null");

    if (!layer->is_output_layer) {
        LOG("Using HE Init for Hidden Layer");
        hidden_layer_init_WB(layer);
    } else {
        LOG("Using Xavior Init for Output Layer");
        output_layer_init_WB(layer);
    }
}


// Use Leaky ReLu?
void relu_activate(const float* z, float* a, u16 size)
{
    for (u16 i = 0; i < size; i++) {
        a[i] = z[i] >= 0.0f ? z[i] : 0.0f;
    }
}

void relu_deriv(const float* z, float* da_dz, u16 size)
{
    // hidden layer so we have da_dz stored in layer
    
    for (u16 i = 0; i < size; i++) {
        da_dz[i] = z[i] >= 0 ? 1.0f : 0.0f;
    }
}

void softmax_activate(const float* z, float* a, u16 size)
{

}

// p = predicted vector , y = true label vector (hot coded)
// dL_dz = (Starting Loss deriv) (got by shortcut)
// The shortcut simplies getting dL_dz directly by:
// dL_dz = predicted - true
void softmax_crossentropy_deriv(const float* predicted, 
                const float* true_label, float* dL_dz, u16 size)
{

}


// Private

Layer* create_common_layer(Arena* arena, u16 m, u16 n)
{
    CHECK_FATAL(!arena, "arena is null");
    CHECK_FATAL(m == 0, "input size can't be zero");
    CHECK_FATAL(n == 0, "output size can't be zero");

    u64 mark = arena_get_mark(arena);

    Layer* l = (Layer*)arena_alloc(arena, sizeof(Layer));
     
    l->x = NULL;   // will be set layer

    // init to all zeros
    l->b = (float*)ARENA_ALLOC_ZERO_N(arena, float, n);

    l->W.m = m; 
    l->W.n = n; 
    l->W.data = (float*)ARENA_ALLOC_N(arena, float, (u64)n * m);

    l->z = (float*)ARENA_ALLOC_N(arena, float, n);
    l->a = (float*)ARENA_ALLOC_N(arena, float, n);

    l->dL_dW.m = m;
    l->dL_dW.n = n;
    l->dL_dW.data = (float*)ARENA_ALLOC_N(arena, float, (u64)n * m);

    l->dL_db = (float*)ARENA_ALLOC_N(arena, float, n);
    l->dL_dz = (float*)ARENA_ALLOC_N(arena, float, n);
    l->dL_dx = (float*)ARENA_ALLOC_N(arena, float, m);

    l->m = m;
    l->n = n;

    LOG("allocated %lu for common layer", arena_used(arena) - mark);

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


