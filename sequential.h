// network comprising sequence of layers

// layer types
#define LINEAR 101
#define DROPOUT 102
// activations
#define SIG 201
#define RELU 202
#define SOFTMAX 203

#include "net.h"

class Sequential : public Net {
  public:
    int* block_types;

    // constructor and destructor
    Sequential(int blocks, int* block_sizes, int* block_types, double sigma);
    ~Sequential();
};