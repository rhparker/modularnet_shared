#include "layer.h"

class Net {
  public:
    int num_layers;
    int* layer_sizes;
    int* layer_data_sizes;

    // layers
    Layer** L;

    // constructor and destructor
    Net(int num_layers);
    ~Net(); 

    // print properties
    void properties();

    // forward propagation on input
    void forward(double** z, int train);

    // backward propagation
    void backward(double** z, double** delta);

    // run network on input
    void run(double* in, double* out);
};