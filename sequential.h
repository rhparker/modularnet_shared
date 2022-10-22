// network comprising sequence of layers

// layer types
#define LINEAR 101
#define DROPOUT 102
#define CONV 103
#define MAXPOOL 104

// activations
#define SIG 201
#define RELU 202
#define SOFTMAX 203

#include <vector>
#include "net.h"

class Sequential : public Net {
  public:
    int* layer_types;
    int valid;

    // constructor and destructor
    // Sequential(int blocks, int* block_sizes, int* block_types, double sigma);
    Sequential(std::vector< std::vector <int > >, double sigma);
    ~Sequential();
};