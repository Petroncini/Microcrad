#include "neuron.h"
#include "value.h"
#include <stdio.h>
#include <stdlib.h>

struct _neuron {
  Value *w;
  Value *b;
};

Neuron *create_neuron() {
  Neuron *neuron = malloc(sizeof(Neuron));
  neuron->w = create_value((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
  neuron->b = create_value((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
  return neuron;
}

Value *call_neuron(Value **vals, int val_num, Neuron *neuron) {
  Value *out = create_value(0.0);
  for (int i = 0; i < val_num; i++) {
    out = add_value(out, mult_value(vals[i], neuron->w));
  }

  out = add_value(out, neuron->b);
  out = tanh_value(out);
  return out;
}

void print_neuron(Neuron *neuron) {
  printf("Neuron(w=%f, b=%f)", get_value_data(neuron->w),
         get_value_data(neuron->b));
}
