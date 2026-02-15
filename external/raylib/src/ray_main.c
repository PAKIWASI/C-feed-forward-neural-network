#include "mnist_predictor.h"



int main(void)
{
    const char* params_path = "data/256_128_64.bin";

    LOG("using params from %s", params_path);

    // Initialize window
    InitWindow(TOTAL_WINDOW_WIDTH, WINDOW_HEIGHT, "MNIST Live Predictor");
    SetTargetFPS(60);

    // Initialize predictor
    LivePredictor predictor;
    init_predictor(&predictor, params_path);

    // Check if network loaded successfully
    if (!predictor.network) {
        WARN("Failed to load network from: %s\n", params_path);
        WARN("Please check:\n");
        WARN("  1. File exists and path is correct\n");
        WARN("  2. File is a valid trained model (created by ffnn_save_parameters)\n");
        WARN("  3. Model has 784 input neurons (MNIST format)\n\n");
        CloseWindow();
        return 1;
    }

    // Brush settings
    float brush_radius = DEFAULT_BRUSH_RADIUS;
    int save_count = 0;

    LOG("Controls:\n");
    LOG(" - Left Mouse Button  : Draw on canvas\n");
    LOG(" - C                  : Clear canvas\n");
    LOG(" - S                  : Save drawing as .raw file\n");
    LOG(" - + / =              : Increase brush size\n");
    LOG(" - - / _              : Decrease brush size\n");
    LOG(" - ESC                : Exit\n\n");
    LOG("Draw a digit (0-9) and watch the prediction update!\n\n");

    // Main loop
    while (!WindowShouldClose())
    {
        // Input handling - Drawing
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON))
        {
            Vector2 mouse = GetMousePosition();
            if (mouse.y < WINDOW_SIZE && mouse.x < WINDOW_SIZE) // Only draw in canvas area
            {
                int canvas_x = (int)(mouse.x / SCALE);
                int canvas_y = (int)(mouse.y / SCALE);
                
                draw_on_canvas(&predictor.canvas, canvas_x, canvas_y, brush_radius);
                
                // Mark for prediction update
                predictor.needs_update = true;
            }
        }

        // Update prediction when drawing stops
        if (predictor.needs_update && !IsMouseButtonDown(MOUSE_LEFT_BUTTON))
        {
            update_prediction(&predictor);
            predictor.needs_update = false;
        }

        // Clear canvas
        if (IsKeyPressed(KEY_C))
        {
            clear_canvas(&predictor.canvas);
            // Reset predictions
            for (int i = 0; i < 10; i++) {
                predictor.predictions[i] = 0.0f;
            }
            predictor.predicted_digit = -1;
            predictor.confidence = 0.0f;
            LOG("Canvas cleared");
        }

        // Save image
        if (IsKeyPressed(KEY_S))
        {
            char filename[256];
            snprintf(filename, sizeof(filename), "mnist_digit_%03d.raw", save_count);
            save_canvas(&predictor.canvas, filename);
            LOG("Saved: %s ", filename);
            if (predictor.predicted_digit >= 0) {
                LOG("(Predicted: %d, Confidence: %.2f%%)", 
                       predictor.predicted_digit, predictor.confidence * 100.0f);
            } else {
                LOG("(No prediction yet)\n");
            }
            save_count++;
        }

        // Adjust brush size
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD))
        {
            brush_radius += 0.25f;
            if (brush_radius > MAX_BRUSH_RADIUS) {
                brush_radius = MAX_BRUSH_RADIUS;
            }
            printf("Brush size: %.2f\n", brush_radius);
        }

        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT))
        {
            brush_radius -= 0.25f;
            if (brush_radius < MIN_BRUSH_RADIUS) {
                brush_radius = MIN_BRUSH_RADIUS;
            }
            printf("Brush size: %.2f\n", brush_radius);
        }

        BeginDrawing();
        ClearBackground(BLACK);

        // Render canvas (left side)
        render_canvas(&predictor.canvas);

        // Render prediction panel (right side)
        draw_prediction_panel(&predictor);

        // Draw UI controls at bottom
        draw_ui(brush_radius, save_count);

        EndDrawing();
    }

    LOG("\nCleaning up and exiting gui...\n");
    cleanup_predictor(&predictor);
    CloseWindow();
    return 0;
}


