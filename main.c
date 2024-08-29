#include "NN.h"
#include "value.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_EPOCHS 1000
#define EARLY_STOP_THRESHOLD 0.001

void train_step(float lr, NN *net, Value ***ypred, Value **ys, Value ***sx,
                float *prev_loss, float *delta_loss);

int main(void) {
  srand(time(NULL));
  rand();

  int layers[] = {4, 4, 1};
  NN *net = create_NN(2, layers, 4);

  Value ***sx = malloc(4 * sizeof(Value **));
  for (int i = 0; i < 4; i++) {
    sx[i] = malloc(2 * sizeof(Value *));
  }

  sx[0][0] = create_value(0.0);
  sx[0][1] = create_value(0.0);
  sx[1][0] = create_value(0.0);
  sx[1][1] = create_value(1.0);
  sx[2][0] = create_value(1.0);
  sx[2][1] = create_value(0.0);
  sx[3][0] = create_value(1.0);
  sx[3][1] = create_value(1.0);

  Value *ys[] = {create_value(0.0), create_value(1.0), create_value(1.0),
                 create_value(0.0)};

  Value ***ypred = malloc(4 * sizeof(Value **));
  for (int i = 0; i < 4; i++) {
    ypred[i] = malloc(sizeof(Value *));
  }

  float initial_lr = 0.1;
  float lr = 0.1;
  float prev_loss = 0;
  float delta_loss = 0;
  for (int i = 0; i < MAX_EPOCHS; i++) {
    printf("Epoch[%i] | lr: %f |  ", i, lr);
    train_step(lr, net, ypred, ys, sx, &prev_loss, &delta_loss);

    lr = initial_lr / (1 + 0.001 * i);

    /* if (lr > 0) { */
    /*   lr -= 0.000001 * delta_loss; */
    /* } */

    if (prev_loss < EARLY_STOP_THRESHOLD) {
      printf("Early stopping at epoch %d\n", i);
      break;
    }
  }

  printf("Final validation:\n");

  for (int i = 0; i < 4; i++) {
    ypred[i] = call_NN(sx[i], 2, net);
  }

  printf("Predictions:\n");
  for (int i = 0; i < 4; i++) {
    print_value(ypred[i][0]);
  }
}

void train_step(float lr, NN *net, Value ***ypred, Value **ys, Value ***sx,
                float *prev_loss, float *delta_loss) {
  for (int i = 0; i < 4; i++) {
  }

  /* printf("Predictions:\n"); */
  /* for (int i = 0; i < 4; i++) { */
  /*   print_value(ypred[i][0]); */
  /* } */

  Value *loss = create_value(0.0);
  for (int i = 0; i < 4; i++) {
    ypred[i] = call_NN(sx[i], 2, net);
    loss = add_value(
        loss, (pow_value(sub_value(ypred[i][0], ys[i]), create_value(2.0))));

    /* loss = add_value(loss, binary_cross_entropy(ypred[i][0], ys[i])); */
  }

  *delta_loss = get_value_data(loss) - *prev_loss;
  *prev_loss = get_value_data(loss);
  printf("Loss: %f\n", get_value_data(loss));
  // NEED TO ZERO THE GRADS BEFORE DOING THE BACKWARD STEP
  zero_grad(net);
  backward(loss);
  gradient_descent(lr, net);
}
