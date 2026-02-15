#ifndef MNIST_LIVE_PREDICTOR_H
#define MNIST_LIVE_PREDICTOR_H

#include <raylib.h>
#include "ffnn.h"


// CANVAS & DRAWING CONFIGURATION

#define CANVAS_SIZE 28
#define SCALE 25              
#define WINDOW_SIZE (CANVAS_SIZE * SCALE)  // 700x700
#define UI_HEIGHT 180  

// Gaussian brush parameters
#define DEFAULT_BRUSH_RADIUS 1.5f
#define MIN_BRUSH_RADIUS 0.5f
#define MAX_BRUSH_RADIUS 3.0f

// Prediction panel
#define PREDICTION_PANEL_WIDTH 350  

#define TOTAL_WINDOW_WIDTH (WINDOW_SIZE + PREDICTION_PANEL_WIDTH)
#define WINDOW_HEIGHT (WINDOW_SIZE + UI_HEIGHT)  // Total height including UI

// UI Colors
#define UI_BG_COLOR ((Color){25, 25, 25, 255})
#define PANEL_BG_COLOR ((Color){20, 20, 20, 255})
#define ACCENT_COLOR ((Color){100, 200, 100, 255})
#define GRID_COLOR ((Color){60, 60, 60, 255})



// CANVAS STRUCTURE

typedef struct {
    u8 data[CANVAS_SIZE][CANVAS_SIZE];
} Canvas;


// LIVE PREDICTOR STATE

typedef struct {
    Canvas canvas;
    ffnn* network;
    float predictions[10];
    int predicted_digit;
    float confidence;
    b8 needs_update;
} LivePredictor;


// CANVAS FUNCTIONS

void init_canvas(Canvas* canvas);
void clear_canvas(Canvas* canvas);
void draw_on_canvas(Canvas* canvas, int center_x, int center_y, float brush_radius);
void render_canvas(Canvas* canvas);
void save_canvas(Canvas* canvas, const char* filename);


// UI FUNCTIONS

void draw_ui(float brush_radius, int save_count);
void draw_prediction_panel(LivePredictor* pred);


// PREDICTOR FUNCTIONS

void init_predictor(LivePredictor* pred, const char* params_path);
void update_prediction(LivePredictor* pred);
void cleanup_predictor(LivePredictor* pred);
b8 load_trained_network(ffnn** net_ptr, const char* params_path);


// UTILITY FUNCTIONS

float gaussian(float x, float y, float sigma);

#endif // MNIST_LIVE_PREDICTOR_H
