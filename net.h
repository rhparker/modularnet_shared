#include "layer.h"

class Net {
  public:
    int num_layers;
    int* layer_sizes;

    // layers
    Layer** L;

    // constructor and destructor
    Net(int num_layers);
    ~Net(); 

    // print properties
    void properties();

    // forward propagation on input
    void forward(double** z);

    // backward propagation
    void backward(double** z, double** delta);

    // run network on input
    void run(double* in, double* out);
};