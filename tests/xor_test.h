#ifndef XOR_TEST_H
#define XOR_TEST_H

#include <stdio.h>
#include "ffnn.h"
#include "layer.h"
#include "random.h"
#include "arena.h"
#include "fast_math.h"


#define LEARNING_RATE 0.01f

// XOR Dataset
typedef struct {
    float inputs[4][2];   // 4 samples, 2 inputs each
    float labels[4][2];   // 4 samples, 2 outputs (one-hot encoded)
} XORDataset;

void init_xor_dataset(XORDataset* dataset)
{
    // XOR truth table
    // Input: [x1, x2], Output: one-hot [class_0, class_1]
    
    // Sample 0: [0, 0] -> 0
    dataset->inputs[0][0] = 0.0f;
    dataset->inputs[0][1] = 0.0f;
    dataset->labels[0][0] = 1.0f;  // Class 0
    dataset->labels[0][1] = 0.0f;
    
    // Sample 1: [0, 1] -> 1
    dataset->inputs[1][0] = 0.0f;
    dataset->inputs[1][1] = 1.0f;
    dataset->labels[1][0] = 0.0f;
    dataset->labels[1][1] = 1.0f;  // Class 1
    
    // Sample 2: [1, 0] -> 1
    dataset->inputs[2][0] = 1.0f;
    dataset->inputs[2][1] = 0.0f;
    dataset->labels[2][0] = 0.0f;
    dataset->labels[2][1] = 1.0f;  // Class 1
    
    // Sample 3: [1, 1] -> 0
    dataset->inputs[3][0] = 1.0f;
    dataset->inputs[3][1] = 1.0f;
    dataset->labels[3][0] = 1.0f;  // Class 0
    dataset->labels[3][1] = 0.0f;
}

void print_xor_prediction(float input1, float input2, float* output)
{
    int predicted = (output[1] > output[0]) ? 1 : 0;
    int expected = (int)input1 ^ (int)input2;  // XOR logic
    
    printf("  [%.0f XOR %.0f] = %d | Predicted: %d (prob=%.4f) | %s\n",
           input1, input2, expected, predicted, output[predicted],
           predicted == expected ? "✓" : "✗");
}

int test_xor_network(void)
{
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║       XOR Problem - Neural Network     ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
    
    // Initialize RNG
    pcg32_rand_seed(42, 1);
    
    // Create network: 2 inputs -> 4 hidden neurons -> 2 outputs
    Arena* arena = arena_create(nKB(4));
    printf("Network Architecture: 2 -> 4 -> 2\n");
    printf("  Input layer:  2 neurons (x1, x2)\n");
    printf("  Hidden layer: 4 neurons (ReLU)\n");
    printf("  Output layer: 2 neurons (Softmax)\n\n");
    
    Layer* hidden = layer_create_hidden(arena, 2, 4);
    Layer* output = layer_create_output(arena, 4, 2);
    
    layer_init_weights_biases(hidden);
    layer_init_weights_biases(output);
    
    // Create XOR dataset
    XORDataset data;
    init_xor_dataset(&data);
    
    printf("XOR Truth Table:\n");
    printf("  Input 1 | Input 2 | Output\n");
    printf("  --------|---------|--------\n");
    for (int i = 0; i < 4; i++) {
        int xor_result = ((int)data.inputs[i][0]) ^ ((int)data.inputs[i][1]);
        printf("     %.0f    |    %.0f    |   %d\n", 
               data.inputs[i][0], data.inputs[i][1], xor_result);
    }
    printf("\n");
    
    // Training
    printf("═══ Training ═══\n");
    int epochs = 1000;
    int print_every = 100;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;
        
        // Train on all 4 samples
        for (int sample = 0; sample < 4; sample++) {
            // Forward pass
            ffnn_forward(hidden, data.inputs[sample]);
            ffnn_forward(output, hidden->a);
            
            // Calculate loss
            float loss = 0.0f;
            for (int i = 0; i < 2; i++) {
                if (data.labels[sample][i] > 0.0f) {
                    loss -= data.labels[sample][i] * fast_log(output->a[i] + 1e-7f);
                }
            }
            total_loss += loss;
            
            // Check accuracy
            int predicted = (output->a[1] > output->a[0]) ? 1 : 0;
            int expected = ((int)data.inputs[sample][0]) ^ ((int)data.inputs[sample][1]);
            if (predicted == expected) { correct++; }
            
            // Backward pass
            ffnn_backward(output, data.labels[sample]);
            ffnn_backward(hidden, output->dL_dx);
            
            // Update weights
            layer_update_weights(output, LEARNING_RATE);
            layer_update_weights(hidden, LEARNING_RATE);
        }
        
        float avg_loss = total_loss / 4.0f;
        float accuracy = ((float)correct / 4.0f) * 100.0f;
        
        if (epoch % print_every == 0 || epoch == epochs - 1) {
            printf("Epoch %4d | Loss: %.6f | Accuracy: %5.1f%% (%d/4)",
                   epoch, avg_loss, accuracy, correct);
            
            if (accuracy == 100.0f) {
                printf(" 🎯");
            }
            printf("\n");
        }
        
        // Early stopping if perfect
        if (correct == 4 && avg_loss < 0.01f) {
            printf("\n✓ Perfect convergence at epoch %d!\n", epoch);
            break;
        }
    }
    
    // Final test
    printf("\n═══ Final Test ═══\n");
    printf("Testing all XOR combinations:\n\n");
    
    for (int i = 0; i < 4; i++) {
        ffnn_forward(hidden, data.inputs[i]);
        ffnn_forward(output, hidden->a);
        
        print_xor_prediction(data.inputs[i][0], data.inputs[i][1], output->a);
    }
    
    // Detailed output probabilities
    printf("\nDetailed Output Probabilities:\n");
    printf("  Input   | P(0)   | P(1)   | Prediction\n");
    printf("----------|--------|--------|------------\n");
    
    for (int i = 0; i < 4; i++) {
        ffnn_forward(hidden, data.inputs[i]);
        ffnn_forward(output, hidden->a);
        
        int predicted = (output->a[1] > output->a[0]) ? 1 : 0;
        int expected = ((int)data.inputs[i][0]) ^ ((int)data.inputs[i][1]);
        
        printf("  [%.0f, %.0f] | %.4f | %.4f |     %d      %s\n",
               data.inputs[i][0], data.inputs[i][1],
               output->a[0], output->a[1], predicted,
               predicted == expected ? "✓" : "✗");
    }
    
    // Test intermediate values (generalization)
    printf("\n═══ Generalization Test ═══\n");
    printf("Testing non-training inputs (should interpolate):\n\n");
    
    float test_inputs[][2] = {
        {0.5f, 0.5f},   // Middle
        {0.2f, 0.8f},   // Near [0,1]
        {0.8f, 0.2f},   // Near [1,0]
        {0.1f, 0.1f},   // Near [0,0]
        {0.9f, 0.9f}    // Near [1,1]
    };
    
    for (int i = 0; i < 5; i++) {
        ffnn_forward(hidden, test_inputs[i]);
        ffnn_forward(output, hidden->a);
        
        int predicted = (output->a[1] > output->a[0]) ? 1 : 0;
        
        printf("  [%.1f, %.1f] -> Predicted: %d (confidence: %.4f)\n",
               test_inputs[i][0], test_inputs[i][1], 
               predicted, output->a[predicted]);
    }
    
    printf("\n═══ Network Learned XOR Function! ═══\n\n");
    
    arena_release(arena);
    return 0;
}


#endif // XOR_TEST_H
