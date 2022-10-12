// multi-layer classifier

#include "classifier.h"
#include <iostream>
#include <cmath>
#include <algorithm>    
#include <vector>
#include <ctime> 
#include <random>
#include <cstdlib>

#ifdef OMP
  #include <omp.h>
#endif

// timer
#ifdef _OPENMP
  #define get_time() omp_get_wtime()
#else
  #include <chrono>
  #define get_time() std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()
#endif

// random number generator
int myrandom(int i) { 
  return std::rand() % i;
}

// argmax function 
// returns argmax of values, which has length len
unsigned int argmax(int len, double* values) {
  double current_max;
  unsigned int current_arg = 0;
  current_max = values[0];
  for (int i = 0; i < len; i++) {
    if (values[i] > current_max) {
      current_max = values[i];
      current_arg = i;
    }
  }
  return current_arg;
}

// constructor : passes through to Sequential
Classifier::Classifier(int blocks, int* block_sizes, int* block_types, double sigma) 
              : Sequential(blocks, block_sizes, block_types, sigma) {};

// destructor
Classifier::~Classifier() {};

// cross-entropy loss
double Classifier::compute_loss(int cnt, double** data, unsigned int* labels) {
  // start timer
  double start_time = get_time();

  // running total of number correct
  double correct = 0;
  train_loss = 0;

  // iterate over all samples
  // can be parallelized
#pragma omp parallel for default(shared) reduction(+:correct) reduction(-:train_loss)
  for (int i = 0; i < cnt; i++) {
    // allocate output
    double* out = new double[ layer_sizes[num_layers] ];
    // evaluate network at current sample
    run(data[i],out);
    // increment number correct if classification output from network (argmax of probability vector)
    // matches label
    if ( argmax( layer_sizes[num_layers], out ) == labels[i] ) {
      correct += 1;
    }
    // update cross-entropy with sample
    train_loss -= log( out[ labels[i] ] );
  }
  // compute and return training accuracy
  train_accuracy = correct/cnt;

  // return total time
  return get_time() - start_time;
}

// run one epoch of training using mini-batch stochastic gradient descent
// cnt:  number of data samples
// data: array containing data
// labels: labels corresponding to data
// lr: learning rate
// wd: weight decay parameter (unused for now)
// batch_size: size of each mini-batch
double Classifier::train_epoch(int cnt, double** data, unsigned int* labels, 
                                  double lr, double wd, unsigned int batch_size) {

  // start timer
  double start_time = get_time();

  // number of batches
  int num_batches = cnt / batch_size;

  // randomly shuffle training samples
  std::srand ( unsigned ( std::time(0) ) );
  int* order = new int[cnt];
  for (int i = 0; i < cnt; i++) {
    order[i] = i;
  }
  std::random_shuffle(order, order+cnt, myrandom);

  // uniform[0,1]
  std::random_device rd; 
  std::mt19937 gen(rd()); 
  std::uniform_real_distribution<double> d(0.0,1.0);

  // iterate over batches
  for (int b = 0; b < num_batches; b++) {

    // arrays to store parameter partials
    double** d_param = new double*[num_layers];
    for (int j = 0; j < num_layers; j++) {
      if ( L[j]->pars > 0 ) {
        d_param[j] = new double[ L[j]->pars ];
        for (int k = 0; k < L[j]->pars; k++) {
          d_param[j][k] = 0;
        }      
      }
    }

#pragma omp parallel for default(shared)
    // compute contributions to paritals from each training sample in batch
    // this can be parallelized
    for (int i = 0; i < batch_size; i++) {
      // index of training sample in data array
      int index = order[ b*batch_size + i ];

      // allocate pointers for layer data z
      double** z = new double*[num_layers+1];
      // first layer is input layer, so set pointer to current sample
      z[0] = data[index];
      for (int j = 1; j <= num_layers; j++) {
        z[j] = new double[ layer_sizes[j] ];
      }

      // step 1: forward propagation on training sample (fills z)
      forward(z);

      // step 2: backward propagation (fills delta)
      // don't need delta[0], but this way keeps indices constant
      double** delta = new double*[num_layers+1];
      for (int j = 1; j <= num_layers; j++) {
        delta[j] = new double[ layer_sizes[j] ];
      }

      // compute output, which we feed back; put in last component of delta
      unsigned int yj;
      for (int j = 0; j < layer_sizes[num_layers]; j++) {
        yj = (j == labels[index]);
        delta[num_layers][j] = z[num_layers][j] - yj;
      }
      // step 2: backward propagation
      backward(z,delta);

#pragma omp critical 
      // step 3: update parameter partials using results of backpropagation
      for (int j = 0; j < num_layers; j++) {
        if ( L[j]->pars > 0 ) {
          L[j]->update_partial_param(z[j], delta[j+1], d_param[j]);
        }
      }

      // delete z and delta
      for (int j = 1; j <= num_layers; j++) {
        delete[] delta[j];
        delete[] z[j];
      }
      delete[] delta;
      delete[] z;
    }

    // step 4: now that we have finished with our mini-batch, update the weights
    // using partial derivatives for entire mini-batch, one layer at a time
    // using stochastic gradient descent
    for (int j = 0; j < num_layers; j++) {
      if ( L[j]->pars > 0 ) {
        // stochastic gradient descent on minibatch
        for (int k = 0; k < L[j]->pars; k++) {
          L[j]->param[k] -= (lr/batch_size) * d_param[j][k];
        }
      }
    }

    // delete partial derivative 
    for (int j = 0; j < num_layers; j++) {
      if ( L[j]->pars > 0 ) {
        delete[] d_param[j];     
      }
    }
    delete[] d_param;
  }

  delete[] order;
  
  // return total time
  return get_time() - start_time;
}

