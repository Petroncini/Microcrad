#include "layer.h"
#include "neuron.h"
#include "value.h"
#include <stdio.h>
#include <stdlib.h>

struct _layer {
  int nin;
  int nout;
  Neuron **neurons;
};

Layer *create_layer(int nin, int nout) {
  Layer *layer = malloc(sizeof(Layer));
  layer->nin = nin;
  layer->nout = nout;
  layer->neurons = malloc(layer->nout * sizeof(Neuron *));
  for (int i = 0; i < nout; i++) {
    layer->neurons[i] = create_neuron(nin);
  }
  return layer;
}

Value **call_layer(Value **vals, int val_num, Layer *layer) {
  Value **outs = malloc(layer->nout * sizeof(Value *));
  /* printf("Calling layer with %i inputs and %i neurons/outputs for %i
   * values\n", */
  /*        layer->nin, layer->nout, val_num); */
  /* ; */
  for (int i = 0; i < layer->nout; i++) {
    /* printf("Calling Neuron %i\n", i); */
    outs[i] = call_neuron(vals, val_num, layer->neurons[i]);
    /* printf("DONE\n"); */
  }

  return outs;
}

void print_layer(Layer *layer) {
  printf("Layer:\n");
  for (int i = 0; i < layer->nout; i++) {
    printf("  ");
    print_neuron(layer->neurons[i]);
    printf("\n");
  }
}

int get_layer_size(Layer *layer) { return layer->nout; }

void layer_update(float lr, Layer *layer) {
  for (int i = 0; i < layer->nout; i++) {
    neuron_update(lr, layer->neurons[i]);
  }
}

void zero_layer(Layer *layer) {
  for (int i = 0; i < layer->nout; i++) {
    zero_neuron(layer->neurons[i]);
  }
}
