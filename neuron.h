#ifndef NEURON_H
#define NEURON_H
#include "value.h"

typedef struct _neuron Neuron;
Neuron *create_neuron();
Value *call_neuron(Value **vals, int val_num, Neuron *neuron);
void print_neuron(Neuron *neuron);
#endif
