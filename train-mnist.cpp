// trains classifier neural network on MNIST data

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include "classifier.h"
#include "loadmnist.h"

int main(int argc, char* argv[]) {

  // properties of MNIST data
  int input_size = 784;
  int output_size = 10;

  // timers
  double batch_time;
  double loss_time;

  // arrays to be allocated for data and labels
  // training
  unsigned int train_cnt;
  double** train_data;
  unsigned int* train_labels;
  // test
  unsigned int test_cnt;
  double** test_data;
  unsigned int* test_labels;

  // load training data
  train_cnt = mnist_load("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", 
        train_data, train_labels);
  if (train_cnt <= 0) {
    printf("An error occured: %d\n", train_cnt);
  } 
  else {
    printf("training image count: %d\n", train_cnt);
  }

  // load test data
  test_cnt = mnist_load("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte", 
        test_data, test_labels);
  if (test_cnt <= 0) {
    printf("An error occured: %d\n", test_cnt);
  } 
  else {
    printf("test image count: %d\n", test_cnt);
  }
  std::cout << std::endl;

  //
  // initialize network
  // 

  // std dev for weight initialization
  double sigma = 0.1;

  // std::vector< std::vector <int > > config = {
  //   {LINEAR,784, 64},
  //   {RELU, 64},
  //   {LINEAR,64, 10},
  //   {SOFTMAX, 10}
  // };

  // std::vector< std::vector <int > > config = {
  //   {LINEAR,784, 64},
  //   {RELU, 64},
  //   {DROPOUT, 64},
  //   {CONV,8,8,1,1},
  //   {MAXPOOL,8,8,1,1,2,2},
  //   {RELU, 16},
  //   {LINEAR,16, 10},
  //   {SOFTMAX, 10}
  // };

  std::vector< std::vector <int > > config = {
    {LINEAR,784, 64},
    {RELU, 64},
    {DROPOUT, 64},
    {CONV3,1,8,8,4,1,1},
    {RELU, 256},
    {LINEAR,256, 10},
    {SOFTMAX, 10}
  };

  Classifier C( config, sigma );

  // print network properties
  C.properties();
  std::cout << std::endl;
  
  //
  // run training epochs
  //

  // initial accuracy and cross entropy (pre-training)
  std::cout 
    << std::left
    << std::setw(8)  << "epoch"
    << std::setw(20) << "cross-entropy loss"
    << std::setw(20) << "train accuracy"
    << std::setw(20) << "test accuracy" 
    << std::setw(20) << "loss time"
    << std::setw(20) << "training time"
    << std::endl;

  loss_time = C.compute_loss(train_cnt, train_data, train_labels);
  std::cout 
    << std::left
    << std::setw(8) << 0
    << std::setw(20) << C.train_loss
    << std::setw(20) << C.train_accuracy;
  loss_time += C.compute_loss(test_cnt, test_data, test_labels);
  std::cout << std::setw(20) << C.train_accuracy
    << std::setw(20) << loss_time
    << std::endl;

  int epochs = 5;
  int batch_size = 10;
  double learning_rate = 0.1;
  double weight_decay = 0;

  // run training epochs
  for (int i = 1; i <= epochs; i++) {
    batch_time = C.train_epoch(train_cnt, train_data, train_labels, 
          learning_rate, weight_decay, batch_size);
    loss_time = C.compute_loss(train_cnt, train_data, train_labels);
    std::cout 
      << std::left
      << std::setw(8) << i
      << std::setw(20) << C.train_loss
      << std::setw(20) << C.train_accuracy;
    loss_time += C.compute_loss(test_cnt, test_data, test_labels);
    std::cout << std::setw(20) << C.train_accuracy
      << std::setw(20) << loss_time
      << std::setw(20) << batch_time
      << std::endl;
  }

  // unallocate training data
  for (int i = 0; i < train_cnt; i++) {
    delete[] train_data[i];
  }
  delete[] train_data;
  delete[] train_labels;

  // unallocate test data
  for (int i = 0; i < test_cnt; i++) {
    delete[] test_data[i];
  }
  delete[] test_data;
  delete[] test_labels;
  

  return 0;
}

