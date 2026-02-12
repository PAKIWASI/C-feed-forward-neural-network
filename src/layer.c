#include "layer.h"
#include "arena.h"
#include "random.h"
#include "fast_math.h"




Layer* create_common_layer(Arena* arena, u16 m, u16 n);
void hidden_layer_init_WB(Layer* layer);
void output_layer_init_WB(Layer* layer);



Layer* layer_create_hidden(Arena* arena, u16 m, u16 n)
{
    LOG("creating hidden layer: %u input, %u output", m, n);

    Layer* hl = create_common_layer(arena, m, n);

    hl->is_output_layer = false;

    return hl;
}


Layer* layer_create_output(Arena* arena, u16 m, u16 n)
{
    LOG("creating output layer: %u input, %u output", m, n);

    Layer* ol = create_common_layer(arena, m, n);

    ol->is_output_layer = true;

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




void layer_update_weights(Layer* layer, float learning_rate)
{
    // Update W using dL_dW
    u64 total = (u64)layer->m * layer->n;

    for (u64 i = 0; i < total; i++) {
        layer->W.data[i] += -1 * learning_rate * layer->dL_dW.data[i];
    }

    // Update biases using dL_db = dL_dz

    for (u16 i = 0; i < layer->n; i++) {
        layer->b[i] += -1 * learning_rate * layer->dL_dz[i];
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

    l->dL_dz = ARENA_ALLOC_N(arena, float, n);
    l->dL_dx = ARENA_ALLOC_N(arena, float, m);

    l->m = m;
    l->n = n;

    l->W_T.m = m;
    l->W_T.n = n;
    l->W_T.data = ARENA_ALLOC_N(arena, float, (u64)m * n);

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


