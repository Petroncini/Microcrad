#include "NN.h"
#include "layer.h"
#include "neuron.h"
#include "value.h"
#include <stdlib.h>

struct _nn {
  int size;
  Layer **layers;
};

NN *create_NN(int nin, int *nouts, int size) {
  NN *nn = malloc(sizeof(NN));
  Layer **layers = malloc((size - 1) * sizeof(Layer *));

  for (int i = 0; i < size - 1; i++) {
    layers[i] = create_layer(nouts[i]);
  }

  nn->size = size;
  nn->layers = layers;

  return nn;
}

Value **call_NN(Value **vals, int val_num, NN *nn) {
  Value **X = vals;
  int X_size = val_num;
  for (int i = 0; i < nn->size - 1; i++) {
    X = call_layer(X, X_size, nn->layers[i]);
    X_size = get_layer_size(nn->layers[i]);
  }

  return X;
}
