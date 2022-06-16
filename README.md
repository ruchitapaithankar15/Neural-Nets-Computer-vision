# Neural Nets
In this project you will learn about neural networks and use them to classify images. You will experiment with network architectures and training hyperparameters to gain intuition about training neural networks.

# Goals

For this assignment, you will write a program that trains a neural network to classify images from the Cifar-10 dataset. Specifically, the program will:

1. Initialize a network (you will be implementing three different network architectures)
2. Trains the network with a chosen set of hyperparameters and options
3. Evaluates the trained network on the test set
4. Plot the training and test metrics over time

# Getting Started

## Environment Setup

We'll be using Python 3 for this assignment. To test your code on tux, you may need to run:

```
pip3 install --user torch
pip3 install --user torchvision
pip3 install --user tqdm
```

## Running on Tux

Tux seems to have a memory allocation limit (about 32MB) which you may run into in this assignment. One way to avoid hitting this limit is to reduce the batch size.

When you use the Adam optimizer on tux, it may throw this error: `libgomp: Thread creation failed: Resource temporarily unavailable`. To resolve this, use `export OMP_NUM_THREADS=1`.

## Skeleton Code

Skeleton code has been linked below to get you started. Though it is not necessary, you are free to add functions as you see fit to complete the assignment. You are _not_, however, allowed to import additional libraries or submit separate code files. Everything you will need has been included in the skeleton. DO NOT change how the program reads parameters or loads the files. failure to follow these rule will result in a large loss of credit.

You can use command line arguments to test different configurations. For example:
```
python3 hw2.py convnet --aug --lr 0.1
```

# Submission

All of your code should be in a single file called `hw2.py`. Be sure to try to write something for each function. If your program errors on run you will lose many, if not all, points.

In addition to your code, you _must_ include the following items:

* A  `ReadMe.pdf` with a description of your experiments. This assignment will be weighted more heavily on this report. In it, please include sections for each of the following:
  * MLP Network
    * Try training the MLP network without any activations. How do the activations affect the training process, the results, and why?
  * Convolutional network
    * Experiment with the number and size of the convolutional layers. What model gave the best results? Which gave the worst results? Speculate as to why.
    * Experiment with the batch size. What works best and worst? Speculate as to why.
    * Experiment with the optimizer. Try SGD with momentum and Adam. Experiment with different learning rates for each. What works best? How does it affect the training process?
    * How does learning rate scheduling affect the performance of the network?
    * How does the model trained with data augmentation compare to the model trained without?
    * How does changing the loss function affect the result?
  * How did the assignment go? Describe what you struggled with, if anything. IF you didn't complete the assignment or there is a serious bug in your code, indicate it here. If you believe your code is awesome, just say so.
  * Any extra credit you implemented.

Call your python file `hw2.py` and zip it, along with your example images and `ReadMe.pdf`, into an archive called `DREXELID_hw_1.zip` where `DREXELID` is your `abc123` alias. DO NOT include _any_ other files.

# Grading
Because tux has limited hardware, and to encourage you to explore more complex ideas if you would like, this assignment does not need to run on tux to be graded. If you'd like to explore GPU-based training, you can try Google Colab for free (https://colab.research.google.com/).

To avoid runtime errors:
* do no import any libraries (this is a requirement)
* do not rely on environment variables
* do not hard code any paths into your program

The assignment is worth 15 points and will be graded as follows:
* 2 points for correctly implementing a single linear layer network
* 2 points for correctly implementing the multi-layer perceptron (MLP) network
* 2 points for correctly implementing the convolutional network
* 1 point for correctly implementing learning rate scheduling
* 1 point for implementing data augmentations
* 7 points for completing the report

Extra Credit Options:
* 1 point: implement equalized learning rate and compare the results in the report
* 1 point: implement a state-of-the-art network (check https://paperswithcode.com/sota/image-classification-on-cifar-10 for state-of-the-art network architectures)
* 1 point: do something really interesting. You must describe the technique and results in the report.

