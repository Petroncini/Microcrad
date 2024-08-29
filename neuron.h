#ifndef NEURON_H
#define NEURON_H
#include "value.h"

typedef struct _neuron Neuron;
Neuron *create_neuron(int nin);
Value *call_neuron(Value **vals, int val_num, Neuron *neuron);
void print_neuron(Neuron *neuron);
void neuron_update(float lr, Neuron *neuron);
void zero_neuron(Neuron *neuron);
#endif
