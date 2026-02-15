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
                SCALE - 1,  // -1 gap for better visibility
                SCALE - 1, 
                pixel_color
            );
        }
    }

    // Draw grid
    for (int i = 0; i <= CANVAS_SIZE; i++)
    {
        // Vertical lines
        DrawLine(i * SCALE, 0, i * SCALE, WINDOW_SIZE, GRID_COLOR);
        // Horizontal lines
        DrawLine(0, i * SCALE, WINDOW_SIZE, i * SCALE, GRID_COLOR);
    }
    
    // Draw border around canvas
    DrawRectangleLines(0, 0, WINDOW_SIZE, WINDOW_SIZE, ACCENT_COLOR);
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

void draw_brush_preview(float brush_radius, int x, int y)
{
    // Draw outer ring
    DrawCircleLines(x, y, brush_radius * 6, ACCENT_COLOR);
    
    // Draw inner filled circle with transparency
    Color preview_color = ACCENT_COLOR;
    preview_color.a = 100;
    DrawCircle(x, y, brush_radius * 5, preview_color);
    
    // Draw center dot
    DrawCircle(x, y, 2, WHITE);
}

void draw_ui(float brush_radius, int save_count)
{
    int ui_y = WINDOW_SIZE;
    int padding = 15;
    
    // Background
    DrawRectangle(0, ui_y, TOTAL_WINDOW_WIDTH, UI_HEIGHT + padding, UI_BG_COLOR);
    
    // Top border line
    DrawLine(0, ui_y, TOTAL_WINDOW_WIDTH, ui_y, ACCENT_COLOR);
    
    // Title with larger font
    DrawText("MNIST LIVE PREDICTOR", padding, ui_y + padding, 28, ACCENT_COLOR);

    // Instructions in two lines with larger font
    DrawText("Left Mouse: Draw   |   C: Clear   |   S: Save", 
             padding, ui_y + padding + 35, 16, LIGHTGRAY);
    DrawText("+ / -: Adjust Brush Size   |   ESC: Exit", 
             padding, ui_y + padding + 55, 16, LIGHTGRAY);
    
    // Brush info panel - moved to right side
    int panel_x = WINDOW_SIZE - 200;
    
    // Brush size indicator
    DrawText("BRUSH SIZE:", panel_x, ui_y + padding, 14, WHITE);
    
    char brush_text[32];
    snprintf(brush_text, sizeof(brush_text), "%.2f", brush_radius);
    
    // Draw brush size value
    int text_width = MeasureText(brush_text, 24);
    DrawText(brush_text, panel_x + 120 - (text_width/2), ui_y + padding + 5, 24, ACCENT_COLOR);
    
    // Brush preview
    draw_brush_preview(brush_radius, panel_x + 80, ui_y + padding + 55);
    
    // Save count with icon-like display
    DrawText("SAVES:", panel_x - 120, ui_y + padding, 14, WHITE);
    
    char save_text[32];
    snprintf(save_text, sizeof(save_text), "%03d", save_count);
    DrawText(save_text, panel_x - 120, ui_y + padding + 20, 28, YELLOW);
}

void draw_prediction_panel(LivePredictor* pred)
{
    int panel_x = WINDOW_SIZE;
    int panel_y = 0;
    int padding = 20;
    
    // Background
    DrawRectangle(panel_x, panel_y, PREDICTION_PANEL_WIDTH, WINDOW_HEIGHT, PANEL_BG_COLOR);
    
    // Left border
    DrawLine(panel_x, 0, panel_x, WINDOW_HEIGHT, ACCENT_COLOR);
    
    // Title
    DrawText("PREDICTION", panel_x + padding, padding, 32, WHITE);
    
    // Main prediction display
    if (pred->predicted_digit >= 0) {
        char digit_str[4];
        snprintf(digit_str, sizeof(digit_str), "%d", pred->predicted_digit);
        
        // Large digit - centered
        int digit_size = 120;
        int digit_x = panel_x + (PREDICTION_PANEL_WIDTH / 2) - (digit_size / 3);
        DrawText(digit_str, digit_x, 80, digit_size, ACCENT_COLOR);
        
        // Confidence below digit with better formatting
        char conf_str[32];
        snprintf(conf_str, sizeof(conf_str), "Confidence: %.1f%%", pred->confidence * 100.0f);
        int conf_x = panel_x + (PREDICTION_PANEL_WIDTH / 2) - (MeasureText(conf_str, 20) / 2);
        DrawText(conf_str, conf_x, 210, 20, LIGHTGRAY);
        
        // Confidence bar
        int bar_width = (int)(pred->confidence * (float)(PREDICTION_PANEL_WIDTH - (2 * padding)));
        DrawRectangle(panel_x + padding, 240, bar_width, 6, ACCENT_COLOR);
        DrawRectangleLines(panel_x + padding, 240, PREDICTION_PANEL_WIDTH - (2 * padding), 6, DARKGRAY);
    } else {
        DrawText("Draw a", panel_x + 100, 120, 28, GRAY);
        DrawText("digit!", panel_x + 105, 160, 32, GRAY);
        
        // Hint
        DrawText("0-9", panel_x + 130, 210, 48, DARKGRAY);
    }

    // Separator line
    int separator_y = 280;
    DrawLine(panel_x + padding, separator_y, panel_x + PREDICTION_PANEL_WIDTH - padding, separator_y, DARKGRAY);

    // Probability bars title
    DrawText("CLASS PROBABILITIES", panel_x + padding, separator_y + 15, 16, WHITE);
    
    // Probability bars with better spacing
    for (int i = 0; i < 10; i++) {
        int bar_y = separator_y + 45 + (i * 32);
        
        // Digit label with background circle
        if (i == pred->predicted_digit) {
            DrawCircle(panel_x + 30, bar_y - 2, 12, ACCENT_COLOR);
        }
        
        char label[4];
        snprintf(label, sizeof(label), "%d", i);
        DrawText(label, panel_x + 25, bar_y - 8, 18, WHITE);
        
        // Probability bar
        int max_bar_width = 180;
        int bar_width = (int)(pred->predictions[i] * (float)max_bar_width);
        Color bar_color = (i == pred->predicted_digit) ? ACCENT_COLOR : (Color){60, 60, 60, 255};
        
        // Draw bar with rounded effect
        DrawRectangle(panel_x + 55, bar_y, bar_width, 18, bar_color);
        DrawRectangleLines(panel_x + 55, bar_y, max_bar_width, 18, DARKGRAY);
        
        // Percentage text
        char prob_str[16];
        snprintf(prob_str, sizeof(prob_str), "%.1f%%", pred->predictions[i] * 100.0f);
        
        // Color the percentage for the predicted digit
        Color text_color = (i == pred->predicted_digit) ? ACCENT_COLOR : LIGHTGRAY;
        DrawText(prob_str, panel_x + 240, bar_y - 2, 14, text_color);
    }
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


