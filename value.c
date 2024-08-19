#include "value.h"
#include <math.h>
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
  void (*_backward)(Value *);
};

void default_backward(Value *self) { return; }

Value *init_value(float data, Value *prev1, Value *prev2, char op[10],
                  char label[50]) {
  Value *v = (Value *)malloc(sizeof(Value));

  v->data = data;
  strncpy(v->op, op, 10);
  v->grad = 0.0f;
  v->prev[0] = prev1;
  v->prev[1] = prev2;
  strncpy(v->label, label, 50);
  v->_backward = default_backward;

  return v;
}

void add_backward(Value *self) {
  self->prev[0]->grad = 1.0f * self->grad;
  self->prev[1]->grad = 1.0f * self->grad;
}

void mult_backward(Value *self) {
  self->prev[0]->grad = self->prev[1]->data * self->grad;
  self->prev[1]->grad = self->prev[0]->data * self->grad;
}

void tanh_backward(Value *self) {
  self->prev[0]->grad = (1.0 - pow(self->data, 2)) * self->grad;
}

void exp_backward(Value *self) {
  self->prev[0]->grad = self->data * self->grad;
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

void backward(Value *self) { self->_backward(self); }

void set_value_grad(Value *v, float grad) { v->grad = grad; }

float get_value_grad(Value *v) { return v->grad; }

float get_value_data(Value *v) { return v->data; }
