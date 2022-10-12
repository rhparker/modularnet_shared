#include "layer.h"

#include <random>
#include <iostream>
#include <cmath>

// macro for number of weights
#define num_weights inputs*outputs
// macro for getting weight(i,j) from weight array stored as 1D flattened parameter array
#define weight(p,i,j) p[ i*inputs + j ]
// macro for getting bias(i) from bias array stored at end of 1D flattened parameter array
#define bias(p,i) p[ num_weights + i ]

//
// abstract layer class
//

// constructor and destructor
Layer::Layer(int inputs, int outputs) : inputs(inputs), outputs(outputs), pars(0) {};
Layer::~Layer() {}; 
void Layer::print_params() {};
void Layer::properties() {};
void Layer::update_partial_param(double* in, double* delta, double* partial) {};


//
// fully connected linear layer
//

// constructor and destructor
Linear::Linear(int inputs, int outputs, double sigma) : Layer(inputs, outputs) {
  // number of parameters (weights and biases)
  // parameters is a linear vector: weights first (as linear vector), then biases
  pars = num_weights + outputs;
  param = new double[pars];

  // random device initialization
  std::random_device rd; 
  std::mt19937 gen(rd()); 
  // instance of class std::normal_distribution with mean 0, std dev sigma
  std::normal_distribution<double> d(0, sigma); 

  // initialize weights to normal(0, sigma)
  for (int i = 0; i < num_weights; i++) {
    param[i] = d(gen);
  }
  // initialize biases to 0
  for (int i = num_weights; i < pars; i++) {
    param[i] = 0; 
  }
}

// destructor
Linear::~Linear() {
  delete[] param;
}

// print weights and biases
void Linear::print_params() {
  // weights
  std::cout << "Weights: " << std::endl;
  // iterate over rows
  for (int i = 0; i < outputs; i++) {
    // go across each row, i.e. iterate over columns
    for (int j = 0; j < inputs; j++) {
      std::cout << weight(param,i,j) << " ";
    }
    std::cout << std::endl << std::endl;
  }
  // biases
  std::cout << "Biases: " << std::endl;
  for (int i = 0; i < outputs; i++) {
    std::cout << bias(param,i) << " ";
  }
  std::cout << std::endl << std::endl;
}

// print properties
void Linear::properties() {
  std::cout << "Linear layer: ";
  std::cout << inputs << " inputs, ";
  std::cout << outputs << " outputs" << std::endl;
}

// forward propagation
void Linear::forward(double* in, double* out) {
  // iterate over rows
  for (int i = 0; i < outputs; i++) {
    out[i] = bias(param,i);
    // iterate over columns
    for (int j = 0; j < inputs; j++) {
      out[i] += weight(param,i,j) * in[j];
    }
  }
}

// backward propagation
void Linear::backward(double* in, double* out, double* delta) {
  for (int i = 0; i < inputs; i++) {
    delta[i] = 0;
    for (int j = 0; j < outputs; j++ ) {
      delta[i] += out[j] * weight(param,j,i);
    }
  }
}

// updates partials with respect to parameters
void Linear::update_partial_param(double* in, double* delta, double* partial) {
  // bias partials
  for (int j = 0; j < outputs; j++) {
    bias(partial,j) += delta[j];
  }
  // weight partials
  for (int j = 0; j < outputs; j++) {
    for (int k = 0; k < inputs; k++) {
      weight(partial,j,k) += delta[j]*in[k];
    }
  }
}

//
// sigmoid activation layer
//

// sigmoid activation function (single element)
inline double sig(double x) {
  return 1.0/(1.0 + exp(-x));
}

// derivative of sigmoid (single element)
inline double d_sig(double x) {
  return sig(x)*(1.0 - sig(x));
}

// constructor and destructor
Sigmoid::Sigmoid(int inputs) : Layer(inputs, inputs) {};
Sigmoid::~Sigmoid() {};

// print properties
void Sigmoid::properties() {
  std::cout << "Sigmoid activation layer" << std::endl;
}

// forward propagation
void Sigmoid::forward(double* in, double* out) {
  // iterate over inputs
  for (int i = 0; i < inputs; i++) {
    out[i] = sig(in[i]);
  }
}

// backward propagation
void Sigmoid::backward(double* in, double* out, double* delta) {
  for (int i = 0; i < inputs; i++) {
    delta[i] = d_sig(in[i]) * out[i];
  }
}

//
// sigmoid activation layer
//

// rectified linear unit (single element)
inline double rec(double x) {
  return fmax(x, 0.0);
}

// derivative of sigmoid (single element)
inline double d_rec(double x) {
  return (x <= 0) ? 0.0 : 1.0;
}

// constructor and destructor
ReLU::ReLU(int inputs) : Layer(inputs, inputs) {};
ReLU::~ReLU() {};

// print properties
void ReLU::properties() {
  std::cout << "ReLU activation layer" << std::endl;
}

// forward propagation
void ReLU::forward(double* in, double* out) {
  // iterate over inputs
  for (int i = 0; i < inputs; i++) {
    out[i] = rec(in[i]);
  }
}

// backward propagation
void ReLU::backward(double* in, double* out, double* delta) {
  for (int i = 0; i < inputs; i++) {
    delta[i] = d_rec(in[i]) * out[i];
  }
}


//
// softmax activation layer
//

// constructor and destructor
Softmax::Softmax(int inputs) : Layer(inputs, inputs) {};
Softmax::~Softmax() {};

// print properties
void Softmax::properties() {
  std::cout << "Softmax activation layer" << std::endl;
}

// forward propagation
void Softmax::forward(double* in, double* out) {
  double normalizer = 0.0;
  for (int i = 0; i < inputs; i++) {
    out[i] = exp(in[i]);
    normalizer += out[i];
  }
  // divide each entry by normalizer
  for (int i = 0; i < inputs; i++) {
    out[i] /= normalizer;
  }
}

// backward propagation
void Softmax::backward(double* in, double* out, double* delta) {
  for (int i = 0; i < inputs; i++) {
    delta[i] = out[i];
  }
}








