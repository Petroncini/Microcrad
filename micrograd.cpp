#include <iostream>
#include <memory>
#include <set>
#include <string>

using namespace std;

class Value {
private:
  double data;
  set<shared_ptr<Value>> _prev;

public:
  // Constructor
  Value(double data, initializer_list<shared_ptr<Value>> children = {})
      : data(data), _prev(children) {}

  // Representation
  string toString() const { return "Value(data=" + to_string(data) + ")"; }

  // Addition operator
  shared_ptr<Value> operator+(shared_ptr<Value> other) const {
    return make_shared<Value>(this->data + other->data,
                              initializer_list<shared_ptr<Value>>{
                                  make_shared<Value>(*this), move(other)});
  }

  // Multiplication operator
  shared_ptr<Value> operator*(shared_ptr<Value> other) const {
    return make_shared<Value>(this->data * other->data,
                              initializer_list<shared_ptr<Value>>{
                                  make_shared<Value>(*this), move(other)});
  }

  // Getter for data (optional)
  double getData() const { return data; }
};

// Overload the << operator for easy printing
ostream &operator<<(ostream &os, const Value &v) {
  os << v.toString();
  return os;
}

int main() {
  Value a(5);
  Value b(-3);

  cout << a + b << "\n";
}
