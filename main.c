#include "value.h"
#include <stdio.h>

int main(void) {
  Value *x1 = init_value(2.0, NULL, NULL, "", "");
  Value *w1 = init_value(-3.0, NULL, NULL, "", "");
  Value *x2 = init_value(0.0, NULL, NULL, "", "");
  Value *w2 = init_value(1.0, NULL, NULL, "", "");
  Value *b = init_value(6.8813735870195432, NULL, NULL, "", "");

  Value *x1w1 = mult_value(x1, w1);
  Value *x2w2 = mult_value(x2, w2);
  Value *x1w1x2w2 = add_value(x1w1, x2w2);
  Value *n = add_value(x1w1x2w2, b);
  Value *o = tanh_value(n);
  set_value_grad(o, 1.0);
  backward(o);
  backward(n);
  backward(b);
  backward(x1w1x2w2);
  backward(x2w2);
  backward(x1w1);
  backward(x1);
  backward(w1);
  backward(x2);
  backward(w2);
  printf("x1w1 data: %f\n", get_value_data(x1w1));
  printf("x2w2 data: %f\n", get_value_data(x2w2));
  printf("x1w1x2w2 data: %f\n", get_value_data(x1w1x2w2));
  printf("n data: %f\n", get_value_data(n));
  printf("o data: %f\n", get_value_data(o));
  printf("o grad: %f\n", get_value_grad(o));
  printf("n grad: %f\n", get_value_grad(n));
  printf("x1w1x2w2 grad: %f\n", get_value_grad(x1w1x2w2));
  printf("x1w1 grad: %f\n", get_value_grad(x1w1));
  printf("x2w2 grad: %f\n", get_value_grad(x2w2));
  printf("x1 grad: %f\n", get_value_grad(x1));
  printf("w1 grad: %f\n", get_value_grad(w1));
  printf("x2 grad: %f\n", get_value_grad(x2));
  printf("w2 grad: %f\n", get_value_grad(w2));
}
