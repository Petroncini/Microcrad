#include "layer.h"
#include "neuron.h"
#include "value.h"
#include <stdio.h>
#include <stdlib.h>

struct _layer {
  int nin;
  Neuron **neurons;
};

Layer *create_layer(int nin) {
  Layer *layer = malloc(sizeof(Layer));
  layer->nin = nin;
  layer->neurons = malloc(layer->nin * sizeof(Neuron *));
  for (int i = 0; i < nin; i++) {
    layer->neurons[i] = create_neuron();
  }
  return layer;
}

Value **call_layer(Value **vals, int val_num, Layer *layer) {
  Value **outs = malloc(layer->nin * sizeof(Value *));

  for (int i = 0; i < layer->nin; i++) {
    outs[i] = call_neuron(vals, val_num, layer->neurons[i]);
  }

  return outs;
}

void print_layer(Layer *layer) {
  printf("Layer:\n");
  for (int i = 0; i < layer->nin; i++) {
    printf("  ");
    print_neuron(layer->neurons[i]);
    printf("\n");
  }
}

int get_layer_size(Layer *layer) { return layer->nin; }

void layer_update(float lr, Layer *layer) {
  for (int i = 0; i < layer->nin; i++) {
    neuron_update(lr, layer->neurons[i]);
  }
}

void zero_layer(Layer *layer) {
  for (int i = 0; i < layer->nin; i++) {
    zero_neuron(layer->neurons[i]);
  }
}
