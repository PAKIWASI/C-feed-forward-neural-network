#include "common.h"
#include "layer.h"
#include "mnist_predictor.h"
#include "fast_math.h"



void init_canvas(Canvas* canvas)
{
    memset(canvas->data, 0, sizeof(canvas->data));
}

void clear_canvas(Canvas* canvas)
{
    init_canvas(canvas);
}

float gaussian(float x, float y, float sigma)
{
    float exponent = -((x*x) + (y*y)) / (2.0f * sigma * sigma);
    return fast_exp(exponent) / (2.0f * PI * sigma * sigma);        // this pi from raylib
}

void draw_on_canvas(Canvas* canvas, int center_x, int center_y, float brush_radius)
{
    // Clamp to canvas bounds
    if (center_x < 0 || center_x >= CANVAS_SIZE || 
        center_y < 0 || center_y >= CANVAS_SIZE) {
        return;
    }

    // Gaussian sigma based on brush radius
    float sigma = brush_radius / 2.0f;
    int radius_int = (int)fast_ceil(brush_radius * 2.5f); // 2.5 sigma coverage

    // Apply gaussian brush
    for (int dy = -radius_int; dy <= radius_int; dy++)
    {
        for (int dx = -radius_int; dx <= radius_int; dx++)
        {
            int x = center_x + dx;
            int y = center_y + dy;

            // Check bounds
            if (x < 0 || x >= CANVAS_SIZE || y < 0 || y >= CANVAS_SIZE) {
                continue;
            }

            // Calculate gaussian weight
            float weight = gaussian((float)dx, (float)dy, sigma);
            
            // Normalize weight (max intensity at center should be 255)
            float max_weight = gaussian(0, 0, sigma);
            float normalized = weight / max_weight;

            // Add to existing value with saturation
            int current = canvas->data[y][x];
            int new_value = current + (int)(normalized * 200.0f); // 200 for softer strokes
            canvas->data[y][x] = (new_value > 255) ? 255 : (u8)new_value;
        }
    }
}


void render_canvas(Canvas* canvas)
{
    for (int y = 0; y < CANVAS_SIZE; y++)
    {
        for (int x = 0; x < CANVAS_SIZE; x++)
        {
            u8 value = canvas->data[y][x];
            Color pixel_color = (Color){value, value, value, 255};
            
            DrawRectangle(
                x * SCALE, 
                y * SCALE, 
                SCALE, 
                SCALE, 
                pixel_color
            );
        }
    }


    // Draw grid 
    for (int i = 0; i <= CANVAS_SIZE; i++)
    {
        DrawLine(i * SCALE, 0, i * SCALE, WINDOW_SIZE, 
                 (Color){40, 40, 40, 255});
        DrawLine(0, i * SCALE, WINDOW_SIZE, i * SCALE, 
                 (Color){40, 40, 40, 255});
    }
}

void save_canvas(Canvas* canvas, const char* filename)
{
    FILE* f = fopen(filename, "wb");
    CHECK_WARN_RET(!f, , "Couldn't open fiel %s", filename);

    // Write raw 28x28 grayscale data
    for (int y = 0; y < CANVAS_SIZE; y++)
    {
        fwrite(canvas->data[y], 1, CANVAS_SIZE, f);
    }

    fclose(f);
}

void draw_ui(float brush_radius, int save_count)
{
    int ui_y = WINDOW_SIZE + 5;
    
    // Background                   // TODO: + 30 here too
    DrawRectangle(0, WINDOW_SIZE, TOTAL_WINDOW_WIDTH + 30, UI_HEIGHT, (Color){30, 30, 30, 255});
    
    // Title and instructions 
    DrawText("MNIST Live Predictor", 10, ui_y, 18, WHITE);
    DrawText("Left Mouse: Draw | C: Clear | S: Save | +/-: Brush", 
             10, ui_y + 25, 13, GRAY);
    
    // Brush info
    char brush_text[64];
    snprintf(brush_text, sizeof(brush_text), "Brush: %.2f", brush_radius);
    DrawText(brush_text, 10, ui_y + 45, 14, GREEN);

    // Save count
    char save_text[64];
    snprintf(save_text, sizeof(save_text), "Saved: %d", save_count);
    DrawText(save_text, 120, ui_y + 45, 14, YELLOW);

    // Preview of brush - moved to right side
    int preview_x = WINDOW_SIZE - 100;
    int preview_y = ui_y + 35;
    DrawText("Brush:", preview_x - 50, preview_y - 5, 12, LIGHTGRAY);
    DrawCircle(preview_x, preview_y, brush_radius * 4, WHITE);  // Reduced from *5 to *4
}

void draw_prediction_panel(LivePredictor* pred)
{
    int panel_x = WINDOW_SIZE;
    int panel_y = 0;
    
    // Background                               // TODO: why tf is padding here on right? +30 fixes it
    DrawRectangle(panel_x, panel_y, PREDICTION_PANEL_WIDTH + 30, WINDOW_HEIGHT, 
                   (Color){20, 20, 20, 255});

    // Title
    DrawText("PREDICTION", panel_x + 30, 20, 28, WHITE);
    
    // Main prediction display
    if (pred->predicted_digit >= 0) {
        char digit_str[4];
        snprintf(digit_str, sizeof(digit_str), "%d", pred->predicted_digit);
        
        // Large digit - centered
        int digit_x = panel_x + (PREDICTION_PANEL_WIDTH / 2) - 40;
        DrawText(digit_str, digit_x, 80, 120, GREEN);
        
        // Confidence below digit
        char conf_str[32];
        snprintf(conf_str, sizeof(conf_str), "%.1f%%", pred->confidence * 100.0f);
        int conf_x = panel_x + (PREDICTION_PANEL_WIDTH / 2) - 40;
        DrawText(conf_str, conf_x, 210, 24, LIGHTGRAY);
    } else {
        DrawText("Draw a", panel_x + 105, 120, 24, GRAY);
        DrawText("digit!", panel_x + 110, 150, 24, GRAY);
    }

    // Separator line
    DrawLine(panel_x, 250, panel_x + PREDICTION_PANEL_WIDTH, 250, DARKGRAY);

    // Probability bars
    DrawText("Probabilities:", panel_x + 20, 265, 18, LIGHTGRAY);
    
    for (int i = 0; i < 10; i++) {
        int bar_y = 300 + (i * 38);  // Increased spacing from 28 to 38
        
        // Digit label
        char label[4];
        snprintf(label, sizeof(label), "%d:", i);
        DrawText(label, panel_x + 25, bar_y, 20, WHITE);
        
        // Probability bar
        int bar_width = (int)(pred->predictions[i] * 220);  // Increased from 200 to 220
        Color bar_color = (i == pred->predicted_digit) ? GREEN : DARKGRAY;
        DrawRectangle(panel_x + 55, bar_y + 3, bar_width, 16, bar_color);
        
        // Percentage text
        char prob_str[16];
        snprintf(prob_str, sizeof(prob_str), "%.1f%%", pred->predictions[i] * 100.0f);
        DrawText(prob_str, panel_x + 280, bar_y, 14, LIGHTGRAY);
    }

    // Separator line
    DrawLine(panel_x, 0, panel_x, WINDOW_HEIGHT, DARKGRAY);
}

b8 load_trained_network(ffnn** net_ptr, const char* params_path)
{
    FILE* f = fopen(params_path, "rb");
    CHECK_WARN_RET(!f, false, "countn't open params %s", params_path);

    LOG("loading pre trained wieghts from %s", params_path);

    // Read network structure
    u64 num_layers;
    fread(&num_layers, sizeof(u64), 1, f);
    
    LOG("Network has %lu layers", num_layers);
    
    // Read layer sizes
    u16* layer_sizes = malloc((num_layers + 1) * sizeof(u16));
    fseek(f, 0, SEEK_SET); // Reset to beginning
    fread(&num_layers, sizeof(u64), 1, f);
    
    for (u64 i = 0; i < num_layers; i++) {
        u16 m, n;
        fread(&m, sizeof(u16), 1, f);
        fread(&n, sizeof(u16), 1, f);
        
        if (i == 0) {
            layer_sizes[0] = m; // Input size
        }
        layer_sizes[i + 1] = n; // Output size of this layer
        
        // Skip weights and biases for now
        fseek(f, (long)(sizeof(float) * ((u64)n * m + n)), SEEK_CUR);
    }

    // Print network structure
    printf("Network structure: ");
    for (u64 i = 0; i <= num_layers; i++) {
        printf("%u", layer_sizes[i]);
        if (i < num_layers) { printf(" -> "); }
    }
    putchar('\n');

    // Verify input size
    if (layer_sizes[0] != 784) {
        WARN("Error: Network expects %u inputs, but MNIST images are 784 pixels", 
               layer_sizes[0]);
        fclose(f);
        free(layer_sizes);
        return false;
    }

    // Create network structure
    ffnn* net = malloc(sizeof(ffnn));
    net->main_arena = arena_create(nMB(5));
    net->dataset_arena = arena_create(1000); // Minimal, not used
    net->learning_rate = 0.01f; // Not used for inference

    // Initialize layers vector
    genVec_init_stk(num_layers, sizeof(Layer*), NULL, NULL, NULL, &net->layers);
    
    // Reopen and read weights
    fclose(f);
    f = fopen(params_path, "rb");
    fread(&num_layers, sizeof(u64), 1, f); // Skip num_layers
    
    for (u64 i = 0; i < num_layers; i++) {
        u16 m, n;
        fread(&m, sizeof(u16), 1, f);
        fread(&n, sizeof(u16), 1, f);
        
        Layer* l;
        if (i == num_layers - 1) {
            l = layer_create_output(net->main_arena, m, n);
        } else {
            l = layer_create_hidden(net->main_arena, m, n);
        }
        
        // Read weights and biases
        fread(l->W.data, sizeof(float), (u64)n * m, f);
        fread(l->b, sizeof(float), n, f);
        
        genVec_push(&net->layers, (u8*)&l);
    }
    
    fclose(f);
    free(layer_sizes);

    *net_ptr = net;
    return true;
}

void init_predictor(LivePredictor* pred, const char* params_path)
{
    // Initialize canvas
    init_canvas(&pred->canvas);
    
    // Initialize predictions
    for (int i = 0; i < 10; i++) {
        pred->predictions[i] = 0.0f;
    }
    pred->predicted_digit = -1;
    pred->confidence = 0.0f;
    pred->needs_update = false;

    // Load trained network
    LOG("Loading trained network from %s...", params_path);
    if (!load_trained_network(&pred->network, params_path)) {
        pred->network = NULL;
        return;
    }

    LOG("Network loaded successfully!\n");
}

void update_prediction(LivePredictor* pred)
{
    if (!pred->network) { return; }

    ffnn* net = pred->network;

    // Normalize canvas data (0-255 -> 0.0-1.0)
    float normalized[784];
    for (int y = 0; y < CANVAS_SIZE; y++) {
        for (int x = 0; x < CANVAS_SIZE; x++) {
            int idx = (y * CANVAS_SIZE) + x;
            normalized[idx] = (float)pred->canvas.data[y][x] / 255.0f;
        }
    }

    // Forward pass through network
    u64 num_layers = genVec_size(&net->layers);
    for (u64 i = 0; i < num_layers; i++) {
        Layer* layer = *(Layer**)genVec_get_ptr(&net->layers, i);
        if (i == 0) {
            layer_calc_output(layer, normalized);
        } else {
            Layer* prev = *(Layer**)genVec_get_ptr(&net->layers, i - 1);
            layer_calc_output(layer, prev->a);
        }
    }

    // Get predictions from output layer
    Layer* output_layer = *(Layer**)genVec_get_ptr(&net->layers, num_layers - 1);
    
    // Copy predictions
    float max_prob = -1.0f;
    int max_idx = -1;
    for (int i = 0; i < 10; i++) {
        pred->predictions[i] = output_layer->a[i];
        if (pred->predictions[i] > max_prob) {
            max_prob = pred->predictions[i];
            max_idx = i;
        }
    }

    pred->predicted_digit = max_idx;
    pred->confidence = max_prob;
}

void cleanup_predictor(LivePredictor* pred)
{
    if (pred->network) {
        ffnn_destroy(pred->network);
    }
}


