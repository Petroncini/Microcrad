#ifndef LAYER_H
#define LAYER_H
#include "neuron.h"
#include "value.h"

typedef struct _layer Layer;
Layer *create_layer(int nin);
Value **call_layer(Value **vals, int val_num, Layer *layer);
void print_layer(Layer *layer);
int get_layer_size(Layer *layer);
#endif
