//
// abstract layer class
//

class Layer {
  public:
    // number of inputs, outputs, and parameters
    int inputs;
    int outputs;
    int pars;
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
