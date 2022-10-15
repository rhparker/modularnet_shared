// network comprising sequence of layers

#include "sequential.h"

// constructor
Sequential::Sequential(int blocks, int* block_sizes, int* block_types, double sigma) 
            : Net(blocks), block_types(block_types) {
  layer_sizes[0] = block_sizes[0];
  layer_data_sizes[0] = 0;
  // add layers in sequence
  for (int i = 0; i < blocks; i++) {
    layer_data_sizes[i+1] = 0;
    layer_sizes[i+1] = block_sizes[i+1];
    switch (block_types[i]) {
      case LINEAR:
        L[i] = new Linear( block_sizes[i], block_sizes[i+1], sigma);
        break;

      case DROPOUT:
        L[i] = new Dropout( block_sizes[i+1] );
        layer_data_sizes[i] = block_sizes[i+1];
        break;

      case SIG:
        L[i] = new Sigmoid( block_sizes[i+1] );
        break;

      case RELU:
        L[i] = new ReLU( block_sizes[i+1] );
        break;

      case SOFTMAX:
        L[i] = new Softmax( block_sizes[i+1] );
        break;
    }
  }
}

// destructor
Sequential::~Sequential() {
  // delete layers
  for (int i = 1; i < num_layers; i++) {
    delete L[i];
  }
}

