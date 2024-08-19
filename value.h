#ifndef VALUE_H
#define VALUE_H

typedef struct _value Value;
Value *init_value(float data, Value *prev1, Value *prev2, char op,
                  char label[50]);
void add_backward(Value *self);
void mult_backward(Value *self);
Value *add_value(Value *v1, Value *v2);
Value *mult_value(Value *v1, Value *v2);
void print_value(Value *v);
#endif // DEBUG
