// network comprising sequence of layers

#include "sequential.h"
#include <iostream>

// constructor
Sequential::Sequential(std::vector< std::vector <int> > config, double sigma) 
            : Net( config.size() ), valid(1) {
  int inputs, outputs;
  layer_types = new int[num_layers];
  for (int i = 0; i < num_layers; i++) {
    layer_types[i] = config[i][0];

    switch (layer_types[i]) {
      case LINEAR:
        L[i] = new Linear(config[i], sigma);
        break;

      case DROPOUT:
        L[i] = new Dropout(config[i]);
        layer_data_sizes[i] = inputs;
        break;

      case CONV:
        L[i] = new Conv(config[i], sigma);
        break;

      case MAXPOOL:
        L[i] = new Maxpool(config[i]);
        layer_data_sizes[i] = outputs;
        break;

      case CONV3:
        L[i] = new Conv3(config[i], sigma);
        break;

      case SIG:
        L[i] = new Sigmoid(config[i]);
        break;

      case RELU:
        L[i] = new ReLU(config[i]);
        break;

      case SOFTMAX:
        L[i] = new Softmax(config[i]);
        break;
    }

    // take number of inputs and outputs from newly created layer
    inputs  = L[i]->inputs;
    outputs = L[i]->outputs;

    // total number of parameters
    pars += L[i]->pars;

    // if input-output mismatch, mark invalid
    // only matters for layers after the first
    if (i > 0 && layer_sizes[i] != inputs ) {
      valid = 0;
    }
    layer_sizes[i] = inputs;
    layer_sizes[i+1] = outputs;
  }
}

// destructor
Sequential::~Sequential() {
  // delete layers
  for (int i = 1; i < num_layers; i++) {
    delete L[i];
  }
}

