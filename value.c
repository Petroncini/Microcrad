#include "value.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct _value {
  float data;
  Value *prev[2];
  char op[10];
  float grad;
  char label[50];
  bool visited;
  void (*_backward)(Value *);
};

void default_backward(Value *self) { return; }

Value *create_value(float data) {
  Value *v = (Value *)malloc(sizeof(Value));
  v->data = data;
  v->prev[0] = NULL;
  v->prev[1] = NULL;
  v->grad = 0.0;
  v->visited = false;
  v->_backward = default_backward;
  return v;
}

Value *init_value(float data, Value *prev1, Value *prev2, char op[10],
                  char label[50]) {
  Value *v = (Value *)malloc(sizeof(Value));

  v->data = data;
  strncpy(v->op, op, 10);
  v->grad = 0.0f;
  v->prev[0] = prev1;
  v->prev[1] = prev2;
  v->visited = false;
  strncpy(v->label, label, 50);
  v->_backward = default_backward;

  return v;
}

void add_backward(Value *self) {
  self->prev[0]->grad += 1.0f * self->grad;
  self->prev[1]->grad += 1.0f * self->grad;
}

void mult_backward(Value *self) {
  self->prev[0]->grad += self->prev[1]->data * self->grad;
  self->prev[1]->grad += self->prev[0]->data * self->grad;
}

void tanh_backward(Value *self) {
  self->prev[0]->grad += (1.0 - pow(self->data, 2)) * self->grad;
}

void exp_backward(Value *self) {
  self->prev[0]->grad += self->data * self->grad;
}

void pow_backward(Value *self) {
  float base = self->prev[0]->data;
  float exponent = self->prev[1]->data;
  self->prev[0]->grad += (exponent * pow(base, exponent - 1)) * self->grad;
  self->prev[1]->grad += ((self->data) * (base)) * self->grad;
}

Value *add_value(Value *v1, Value *v2) {
  Value *out = init_value(v1->data + v2->data, v1, v2, "+", "");

  out->_backward = add_backward;
  return out;
}

Value *mult_value(Value *v1, Value *v2) {
  Value *out = init_value(v1->data * v2->data, v1, v2, "*", "");

  out->_backward = mult_backward;
  return out;
}

Value *pow_value(Value *v1, Value *v2) {
  Value *out = init_value(pow(v1->data, v2->data), v1, v2, "^", "");

  out->_backward = pow_backward;
  return out;
}

Value *tanh_value(Value *v) {
  float out_data = (exp(2 * v->data) - 1) / (exp(2 * v->data) + 1);
  Value *out = init_value(out_data, v, NULL, "tanh", "");

  out->_backward = tanh_backward;
  return out;
}

Value *exp_value(Value *v) {
  Value *out = init_value(exp(v->data), v, NULL, "exp", "");

  out->_backward = exp_backward;
  return out;
}

void print_value(Value *v) { printf("Value(data=%f)\n", v->data); }

void build_topo(Value ***topo, Value *v, int *topo_size, int *topo_capacity) {
  if (!(v->visited)) {
    v->visited = true;
    for (int i = 0; i < 2; i++) {
      if (v->prev[i] != NULL) {
        build_topo(topo, v->prev[i], topo_size, topo_capacity);
      }
    }

    if (*topo_size >= *topo_capacity) {
      *topo_capacity += 1000;
      *topo = realloc(topo, (*topo_capacity) * sizeof(Value *));
    }

    (*topo)[(*topo_size)++] = v;
  }
}

void backward(Value *self) {
  int topo_capacity = 1000;
  Value **topo = malloc(topo_capacity * sizeof(Value *));
  int topo_size = 0;
  build_topo(&topo, self, &topo_size, &topo_capacity);

  for (int i = topo_size - 1; i >= 0; i--) {
    if (i == topo_size - 1) {
      topo[i]->grad = 1.0;
    }
    topo[i]->visited = false; // reseting visited status
    topo[i]->_backward(topo[i]);
  }
}

void set_value_grad(Value *v, float grad) { v->grad = grad; }

void set_value_label(Value *v, char label[50]) { strncpy(v->label, label, 50); }

float get_value_grad(Value *v) { return v->grad; }

float get_value_data(Value *v) { return v->data; }
