#include "value.h"
#include <stdio.h>

int main(void) {
  Value *x1 = create_value(2.0, "");
  Value *w1 = create_value(-3.0, "");
  Value *x2 = create_value(0.0, "");
  Value *w2 = create_value(1.0, "");
  Value *b = create_value(6.8813735870195432, "");

  Value *x1w1 = mult_value(x1, w1);
  Value *x2w2 = mult_value(x2, w2);
  Value *x1w1x2w2 = add_value(x1w1, x2w2);
  Value *n = add_value(x1w1x2w2, b);
  Value *o = tanh_value(n);
  backward(o);
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
