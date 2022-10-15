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
        inputs  = config[i][1];
        outputs = config[i][2];
        L[i] = new Linear( inputs, outputs, sigma);
        break;

      case DROPOUT:
        inputs  = config[i][1];
        outputs = config[i][1];
        L[i] = new Dropout( inputs );
        layer_data_sizes[i] = inputs;
        break;

      case CONV:
        inputs = config[i][1]*config[i][2];
        L[i] = new Conv(config[i][1],config[i][2],config[i][3],config[i][4],1,1,sigma);
        outputs = L[i]->outputs;
        break;

      case MAXPOOL:
        inputs = config[i][1]*config[i][2];
        L[i] = new Maxpool(config[i][1],config[i][2],config[i][3],config[i][4],
            config[i][5],config[i][6]);
        outputs = L[i]->outputs;
        layer_data_sizes[i] = outputs;
        break;

      case SIG:
        inputs  = config[i][1];
        outputs = config[i][1];
        L[i] = new Sigmoid( inputs );
        break;

      case RELU:
        inputs  = config[i][1];
        outputs = config[i][1];
        L[i] = new ReLU( inputs );
        break;

      case SOFTMAX:
        inputs  = config[i][1];
        outputs = config[i][1];
        L[i] = new Softmax( inputs );
        break;
    }

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

