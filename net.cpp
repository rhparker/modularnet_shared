#include "net.h"
#include <iostream>

// constructor
Net::Net(int num_layers) : num_layers(num_layers) {
  // allocate layers
  L = new Layer*[num_layers];
  layer_sizes = new int[num_layers+1];
}

// destructor
Net::~Net() {
    delete[] L;
    delete[] layer_sizes;
}

// print properties
void Net::properties() {
  for (int i = 0; i < num_layers; i++) {
    L[i]->properties();
  }
  std::cout << std::endl;
}

// forward propagation on input
void Net::forward(double** z) {
  for (int i = 0; i < num_layers; i++) {
    L[i]->forward(z[i],z[i+1]);
  }
}

// backward propagation on output
void Net::backward(double** z, double** delta) {
  // work backwards from last layer
  for (int i = num_layers - 1; i > 0; i--) {
    L[i]->backward(z[i], delta[i+1], delta[i]);
  }
}

// run network on input
void Net::run(double* in, double* out) {
    double** z = new double*[num_layers+1];
    // first component is input
    z[0] = in;
    // last component is pointer to output
    z[num_layers] = out;
    // allocate space for remaining layers
    for (int i = 1; i < num_layers; i++) {
      z[i] = new double[ layer_sizes[i] ];
    }
    // forward propagation
    forward(z);
}
