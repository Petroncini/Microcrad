#include "value.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct _value {
  float data;
  Value *prev[2];
  char op;
  float grad;
  char label[50];
  void (*_backward)(Value *);
};

Value *init_value(float data, Value *prev1, Value *prev2, char op,
                  char label[50]) {
  Value *v = (Value *)malloc(sizeof(Value));

  v->data = data;
  v->op = op;
  v->grad = 0.0f;
  v->prev[0] = prev1;
  v->prev[1] = prev2;
  strncpy(v->label, label, 50);
  v->_backward = NULL;

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

Value *add_value(Value *v1, Value *v2) {
  Value *out = init_value(v1->data * v2->data, v1, v2, '+', "");

  out->_backward = add_backward;
  return out;
}

Value *mult_value(Value *v1, Value *v2) {
  Value *out = init_value(v1->data * v2->data, v1, v2, '*', "");

  out->_backward = mult_backward;
  return out;
}

void print_value(Value *v) { printf("Value(data=%f)\n", v->data); }

int main(void) {
  Value *a = init_value(2.0, NULL, NULL, ' ', "a");
  Value *b = init_value(-3.0, NULL, NULL, ' ', "b");
  Value *c = mult_value(a, b);
  c->grad = 1.0f;
  c->_backward(c);
  print_value(c);
  printf("a grad: %f\n", a->grad);
}
