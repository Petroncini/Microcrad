#ifndef VALUE_H
#define VALUE_H

typedef struct _value Value;
Value *create_value(float data);
Value *add_value(Value *v1, Value *v2);
Value *mult_value(Value *v1, Value *v2);
void print_value(Value *v);
Value *exp_value(Value *v);
void backward(Value *self);
void set_value_grad(Value *v, float grad);
float get_value_grad(Value *v);
Value *tanh_value(Value *v);
float get_value_data(Value *v);
Value *div_value(Value *v1, Value *v2);
Value *sub_value(Value *v1, Value *v2);
Value *pow_value(Value *v1, Value *v2);
Value *reLu_value(Value *v);
Value *sig_value(Value *v);
Value *binary_cross_entropy(Value *pred, Value *target);
#endif
