#ifndef NN_H
#define NN_H
#include "layer.h"
#include "neuron.h"
#include "value.h"

typedef struct _nn NN;
NN *create_NN(int nin, int *nouts, int size);
Value **call_NN(Value **vals, int val_num, NN *nn);
void gradient_descent(float lr, NN *nn);
void zero_grad(NN *nn);
#endif
