#include "NN.h"
#include "layer.h"
#include "neuron.h"
#include "value.h"
#include <stdio.h>
#include <stdlib.h>

struct _nn {
  int size;
  Layer **layers;
};

NN *create_NN(int nin, int *nouts, int size) {
  NN *nn = malloc(sizeof(NN));
  Layer **layers = malloc((size - 1) * sizeof(Layer *));
  int n_inputs = nin;
  for (int i = 0; i < size - 1; i++) {
    layers[i] = create_layer(n_inputs, nouts[i]);
    /* printf("Creating layer %i, with %i inputs and %i outputs\n", i, n_inputs,
     */
    /*        nouts[i]); */
    n_inputs = nouts[i];
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

void gradient_descent(float lr, NN *nn) {
  for (int i = 0; i < nn->size - 1; i++) {
    layer_update(lr, nn->layers[i]);
  }
}

void zero_grad(NN *nn) {
  for (int i = 0; i < nn->size - 1; i++) {
    zero_layer(nn->layers[i]);
  }
}
