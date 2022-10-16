#include "layer.h"

#include <random>
#include <iostream>
#include <cmath>
#include <stdio.h>

// macro for getting weight(i,j) from weight array stored as 1D flattened parameter array
#define weight(p,i,j) p[ i*inputs + j ]
// macro for getting bias(i) from bias array stored at end of 1D flattened parameter array
#define bias(p,i) p[ num_weights + i ]
// macro for getting mask index
#define mask(p,i) p[ inputs + i ]
// macro for generic index
#define idx(n,i,j) n*i + j

//
// abstract layer class
//

// constructor and destructor
Layer::Layer(int inputs, int outputs) : 
    inputs(inputs), outputs(outputs), pars(0), train(0) {};
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
  num_weights = inputs*outputs;
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
    std::cout << std::endl;
  }
  std::cout << std::endl;
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

//
// dropout layer
//

// constructor and destructor
Dropout::Dropout(int inputs) : Layer(inputs, inputs),
    gen(rd()), d(std::uniform_real_distribution<>(0,1)), 
    drop_prob(0.25) {};

Dropout::~Dropout() {};

void Dropout::set_dropout(double prob) {
  drop_prob = prob;
}

// print properties
void Dropout::properties() {
  std::cout << "Dropout layer: dropout probability " << drop_prob << std::endl;
}

// forward propagation
void Dropout::forward(double* in, double* out) {
  // pass through if not training
  if (train == 0) {
    for (int i = 0; i < inputs; i++) {
      out[i] = in[i];
    }
  }
  // if we are training, the do dropout
  else {
    // generate random mask, store it at end of input vector
    for (int i = 0; i < outputs; i++) {
      mask(in,i) = ( d(gen) > drop_prob ) ? 1.0 : 0;
    }
    for (int i = 0; i < inputs; i++) {
      out[i] = in[i] * mask(in,i) / (1-drop_prob);
    }
  }
}

// backward propagation
void Dropout::backward(double* in, double* out, double* delta) {
  for (int i = 0; i < inputs; i++) {
    delta[i] = out[i] * mask(in,i) / (1-drop_prob);
  }
}


//
// convolution with 2D kernel
//

// constructor
// for now enforce stride 1
Conv::Conv(int m, int n, int km, int kn, int sm, int sn, double sigma) : 
    Layer(m*n, 0), input_m(m), input_n(n), ker_m(km), ker_n(kn), stride_m(1), stride_n(1) {

  // output size is ceil(input_m / sm) x ceil(input_n/sn)
  output_m = (int) ceil ( ((double)input_m) / ((double) stride_m) ); 
  output_n = (int) ceil ( ((double)input_n) / ((double) stride_n) ); 
  outputs = output_m * output_n;

  // number of weights parameters ( 2km+1 x 2kn+1, size of kernel )
  num_weights = (2*ker_m+1) * (2*ker_n+1);
  // total parameters is weights + biases
  pars = num_weights + outputs;
  param = new double[pars];

  // random device initialization
  std::random_device rd; 
  std::mt19937 gen(rd()); 
  // instance of class std::normal_distribution with mean 0, std dev sigma
  std::normal_distribution<double> d(0, sigma); 

  // initialize parameters to normal(0, sigma)
  for (int i = 0; i < pars; i++) {
    param[i] = d(gen);
  }
  // initialize biases to 0
  for (int i = num_weights; i < pars; i++) {
    param[i] = 0; 
  }
}

// destructor
Conv::~Conv() {
  delete[] param;
}

// print properties
void Conv::properties() {
  printf("Convolution layer: input %d (%d, %d), kernel (%d, %d), stride (%d, %d)\n",
    input_m*input_n,input_m,input_n,2*ker_m+1,2*ker_n+1,stride_m,stride_n);
}

void Conv::print_params() {
  std::cout << "Kernel: " << std::endl;
  // iterate over rows
  for (int i = 0; i < (2*ker_m+1); i++) {
    // iterate over columns
    for (int j = 0; j < (2*ker_n+1); j++) {
      std::cout << param[ idx( (2*ker_n+1), i, j) ] << "  ";
    }
    std::cout << std::endl;
  }
  // biases
  std::cout << "Biases: " << std::endl;
  for (int i = 0; i < outputs; i++) {
    std::cout << bias(param,i) << " ";
  }
  std::cout << std::endl << std::endl;
}

// forward propagation
void Conv::forward(double* in, double* out) {
  int row, col;
  // row of output
  for (int i = 0; i < output_m; i++) {
    // column of output
    for (int j = 0; j < output_n; j++) {
      // intialize output to bias
      out[ idx(output_n,i,j) ] = bias( param, idx(output_n,i,j) );
      // row of kernel
      for (int ki = 0; ki < (2*ker_m+1); ki++ ) {
        // column of kernel
        for (int kj = 0; kj < (2*ker_n+1); kj++) {
          // index of row and column we need in input
          row = stride_m*i + ki - ker_m;
          col = stride_n*j + kj - ker_n;
          // if we are in bounds, then multiply by appropriate kernel element
          // to perform convolution; this effectively pads with zeros
          if (row >= 0 && row < input_m && col >= 0 && col < input_n) {
            out[ idx(output_n, i, j) ] += 
              param[ idx( (2*ker_n+1), ki, kj) ] *
              in[ idx(input_n, row, col) ];
          }
        }
      }
    }
  }
}

// backward propagation
// for now enforce stride 1
void Conv::backward(double* in, double* out, double* delta) {
  int row, col;
  // row of input (delta is same size as input)
  for (int i = 0; i < input_m; i++) {
    // column of input (delta is same size as input)
    for (int j = 0; j < input_n; j++) {
      delta[ idx(input_n,i,j) ] = 0;
      // row of kernel
      for (int ki = 0; ki < (2*ker_m+1); ki++ ) {
        // column of kernel
        for (int kj = 0; kj < (2*ker_n+1); kj++) {
          // index of row and column we need from output
          row = i + ki - ker_m;
          col = j + kj - ker_n;
          // if we are in bounds, then multiply by appropriate kernel element
          // for backpropagation we flip the kernel (horiz and vert)
          if (row >= 0 && row < output_m && col >= 0 && col < output_n) {
            delta[ idx(input_n, i, j) ] += 
              param[ idx( (2*ker_n+1), (2*ker_m-ki), (2*ker_n-kj) ) ] *
              out[ idx(output_n, row, col) ];
          }
        }
      }
    }
  }
}

// update partial derivative of loss with respect to parmeters 
void Conv::update_partial_param(double* in, double* delta, double* partial) {
  int row, col;
  // bias partials
  for (int i = 0; i < outputs; i++) {
    bias(partial,i) += delta[i];
  }
  // row of kernel
  for (int ki = 0; ki < (2*ker_m+1); ki++ ) {
    // column of kernel
    for (int kj = 0; kj < (2*ker_n+1); kj++) {
      partial[ idx( (2*ker_n+1), ki, kj) ] = 0;
      // row of output
      for (int i = 0; i < output_m; i++) {
        // column of output
        for (int j = 0; j < output_n; j++) {
          row = stride_m*i + ki - ker_m;
          col = stride_n*j + kj - ker_n;
          if (row >= 0 && row < input_m && col >= 0 && col < input_n) {
            partial[ idx( (2*ker_n+1), ki, kj) ] += 
              delta[ idx(output_n, i, j) ] *
              in[ idx(input_n, row, col) ];
          }
        }
      }
    }
  }
}


//
// Max pool with 2D window
//

// constructor
Maxpool::Maxpool(int m, int n, int wm, int wn, int sm, int sn) : 
    Layer(m*n, 0), input_m(m), input_n(n), window_m(wm), window_n(wn), stride_m(sm), stride_n(sn) {

  // output size is ceil(input_m / sm) x ceil(input_n/sn)
  output_m = (int) ceil ( ((double)input_m) / ((double) stride_m) ); 
  output_n = (int) ceil ( ((double)input_n) / ((double) stride_n) ); 
  outputs = output_m * output_n;
}

// destructor
Maxpool::~Maxpool() {};

// print properties
void Maxpool::properties() {
  printf("Max pool layer: input %d (%d, %d), window (%d, %d), stride (%d, %d)\n",
    input_m*input_n,input_m,input_n,2*window_m+1,2*window_n+1,stride_m,stride_n);
};

// forward propagation
void Maxpool::forward(double* in, double* out) {
  int row, col, center;
  // row of output
  for (int i = 0; i < output_m; i++) {
    // column of output
    for (int j = 0; j < output_n; j++) {
      // initialize current max to center of window
      center = idx(input_n, stride_m*i, stride_n*j);
      out[ idx(output_n,i,j) ] = in[ center ];
      // if training, save argmax
      if (train == 1) {
        in[ inputs + idx(output_n,i,j) ] = center;
      }
      // row of window
      for (int wi = 0; wi < (2*window_m+1); wi++ ) {
        // column of window
        for (int wj = 0; wj < (2*window_n+1); wj++) {
          row = stride_m*i + wi - window_m;
          col = stride_n*j + wj - window_n;
          // only grab elements of input if we are in bounds
          // this effectively pads with zeros
          if (row >= 0 && row < input_m && col >= 0 && col < input_n) {
            // update max if we are greater than current max
            if ( in[ idx(input_n, row, col) ] > out[ idx(output_n,i,j) ] ) {
              out[ idx(output_n,i,j) ] = in[ idx(input_n, row, col) ];
              // if training, update argmax as well
              if (train == 1) {
                in[ inputs + idx(output_n,i,j) ] = idx(input_n, row, col);
              }
            }
          }
        }
      }
    }
  }
}

//  backward propagation
void Maxpool::backward(double* in, double* out, double* delta) {
  int amax;
  // initialize all delta to 0
  for (int i = 0; i < inputs; i++) {
    delta[i] = 0;
  }
  // add outputs to input corresponding to argmax
  for (int i = 0; i < outputs; i++) {
    amax = in[ inputs + i ];
    delta[amax] += out[i];
  }
}









