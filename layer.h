#include <random>

//
// abstract layer class
//

class Layer {
  public:
    // number of inputs, outputs, and parameters
    int inputs;
    int outputs;
    int pars;

    // are we training or not?
    int train;

    // parameters
    double* param;

    // constructor and destructor
    Layer(int inputs, int outputs);
    virtual ~Layer(); 

    // print parameters and properties
    virtual void print_params();
    virtual void properties();

    // forward and backward propagation
    virtual void forward(double* in, double* out) = 0;
    virtual void backward(double* in, double* out, double* delta) = 0;

    // update partial derivative of loss with respect to parmeters 
    virtual void update_partial_param(double* in, double* delta, double* partial);
};

//
// fully connected linear layer
//

class Linear : public Layer {
  public:
    // number of weights
    int num_weights;

    // constructor and destructor
    Linear(int inputs, int outputs, double sigma);
    ~Linear();

    // print parameters and properties
    void print_params();
    void properties();

    // forward and backward propagation
    virtual void forward(double* in, double* out);
    virtual void backward(double* in, double* out, double* delta);

    // update partial derivative of loss with respect to parmeters 
    virtual void update_partial_param(double* in, double* delta, double* partial);
};

//
// sigmoid activation layer
//

class Sigmoid : public Layer {
  public:
     // constructor and destructor : same number of inputs and outputs
    Sigmoid(int inputs);
    ~Sigmoid();

    // print properties
    void properties();

    // forward and backward propagation
    virtual void forward(double* in, double* out);
    virtual void backward(double* in, double* out, double* delta);
};

//
// ReLU (rectified linear unit) activation layer
//

class ReLU : public Layer {
  public:
     // constructor and destructor : same number of inputs and outputs
    ReLU(int inputs);
    ~ReLU();

    // print properties
    void properties();

    // forward and backward propagation
    virtual void forward(double* in, double* out);
    virtual void backward(double* in, double* out, double* delta);
};

//
// softmax activation layer
//

class Softmax : public Layer {
  public:
     // constructor and destructor : same number of inputs and outputs
    Softmax(int inputs);
    ~Softmax();

    // print properties
    void properties();

    // forward and backward propagation
    virtual void forward(double* in, double* out);
    virtual void backward(double* in, double* out, double* delta);
};


//
// dropout layer
//

class Dropout : public Layer {
  public:
    // dropout probability
    double drop_prob;

    // constructor and destructor : same number of inputs and outputs
    Dropout(int inputs);
    ~Dropout();

    // set dropout probability (default from constructor is 0.25)
    void set_dropout(double prob);

    // print properties
    void properties();

    // forward and backward propagation
    virtual void forward(double* in, double* out);
    virtual void backward(double* in, double* out, double* delta);

  private:
    // uniform[0,1] random number generator
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> d;
};

//
// convolution with 2D kernel
//

class Conv : public Layer {
  public:
    // dimensions in 2D (m x n)
    int input_m, input_n;
    // kernel dimensions
    int ker_m, ker_n;
    // strides
    int stride_m, stride_n;
    // output dimensions (om x om)
    int output_m, output_n;
    // number of weights (kernel components)
    int num_weights;

    // constructor and destructor
    Conv(int m, int n, int km, int kn, int sm, int sn, double sigma);
    ~Conv();

    // print parameters and properties
    void print_params();
    void properties();

    // forward and backward propagation
    virtual void forward(double* in, double* out);
    virtual void backward(double* in, double* out, double* delta);

    // update partial derivative of loss with respect to parmeters 
    virtual void update_partial_param(double* in, double* delta, double* partial);
};


//
// Max pool with 2D window
//

class Maxpool : public Layer {
  public:
    // dimensions in 2D (m x n)
    int input_m, input_n;
    // window dimensions
    int window_m, window_n;
    // strides
    int stride_m, stride_n;
    // output dimensions (om x om)
    int output_m, output_n;

    // constructor and destructor
    Maxpool(int m, int n, int wm, int wn, int sm, int sn);
    ~Maxpool();

    // print properties
    void properties();

    // forward and backward propagation
    virtual void forward(double* in, double* out);
    virtual void backward(double* in, double* out, double* delta);
};