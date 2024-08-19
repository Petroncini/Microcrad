#ifndef VALUE_H
#define VALUE_H

typedef struct _value Value;
Value *create_value(float data, char label[50]);
Value *add_value(Value *v1, Value *v2);
Value *mult_value(Value *v1, Value *v2);
void print_value(Value *v);
Value *exp_value(Value *v);
void backward(Value *self);
void set_value_grad(Value *v, float grad);
float get_value_grad(Value *v);
Value *tanh_value(Value *v);
float get_value_data(Value *v);
#endif
