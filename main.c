#include "NN.h"
#include "layer.h"
#include "neuron.h"
#include "value.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
  srand(time(NULL));
  rand();

  int layers[] = {4, 4, 1};
  NN *net = create_NN(3, layers, 4);

  Value *sx[4][3] = {
      {create_value(2.0), create_value(3.0), create_value(-1.0)},
      {create_value(3.0), create_value(-1.0), create_value(0.5)},
      {create_value(0.5), create_value(1.0), create_value(1.0)},
      {create_value(1.0), create_value(1.0), create_value(-1.0)}};

  Value *ys[] = {create_value(1.0), create_value(-1.0), create_value(-1.0),
                 create_value(1.0)};
  Value **ypred[4];
  for (int i = 0; i < 4; i++) {
    ypred[i] = call_NN(sx[i], 3, net);
  }

  for (int i = 0; i < 4; i++) {
    print_value(ypred[i][0]);
  }

  Value *loss = create_value(0.0);

  for (int i = 0; i < 4; i++) {
    loss = add_value(
        loss, (pow_value(sub_value(ypred[i][0], ys[i]), create_value(2.0))));
  }

  print_value(loss);
  backward(loss);
}
