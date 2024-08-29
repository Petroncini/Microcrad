#include "neuron.h"
#include "value.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

struct _neuron {
  int nin;
  Value **w;
  Value *b;
};

Neuron *create_neuron(int nin) {
  Neuron *neuron = malloc(sizeof(Neuron));
  neuron->nin = nin;
  neuron->w = malloc(nin * sizeof(Value *));
  for (int i = 0; i < nin; i++) {
    float scale = sqrt(2.0 / (nin + 1)); // +1 for the output
    neuron->w[i] = create_value(((float)rand() / RAND_MAX * 2 - 1) * scale);
    /* neuron->w[i] = create_value((rand() / (float)RAND_MAX) * 2.0f - 1.0f); */
  }
  neuron->b = create_value((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
  return neuron;
}

Value *call_neuron(Value **vals, int val_num, Neuron *neuron) {
  Value *out = create_value(0.0);
  for (int i = 0; i < val_num; i++) {
    out = add_value(out, mult_value(vals[i], neuron->w[i]));
  }
  out = add_value(out, neuron->b);
  out = reLu_value(out);
  return out;
}

void print_neuron(Neuron *neuron) {
  /* printf("Neuron(w=%f, b=%f, w grad: %f, b grad: %f)\n", */
  /*        get_value_data(neuron->w), get_value_data(neuron->b), */
  /*        get_value_grad(neuron->w), get_value_grad(neuron->b)); */
}

void neuron_update(float lr, Neuron *neuron) {
  for (int i = 0; i < neuron->nin; i++) {

    neuron->w[i] = sub_value(neuron->w[i],
                             create_value(lr * get_value_grad(neuron->w[i])));
  }
  neuron->b =
      sub_value(neuron->b, create_value(lr * get_value_grad(neuron->b)));
}

void zero_neuron(Neuron *neuron) {
  for (int i = 0; i < neuron->nin; i++) {
    set_value_grad(neuron->w[i], 0.0);
  }
  set_value_grad(neuron->b, 0.0);
}
