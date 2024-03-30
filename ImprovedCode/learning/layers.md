# Layers

## Layers General Info

- different layers perform different operations on it's input
- different layers are best suited for different tasks
- *Dense* layers -> connects each input to each output of the respective layer
  - Dense(5, input_shape=(3,), activation='relu')
    - 5 neurons in the Dense Layer
    - input_shape -> required only by the first layer, \
      the form in which the data is feeded into the network **???** \
      also makes an implicit input layer(instead of calling *Input()* **???**)
    - activation -> the activ_function used for **???**
- *Hidden* layers -> layers between *input* and *output* layers
- *weight* -> strength of the connection between the units/nodes/neurons
- *output of a layer?* = activation_function(wighted sum of inputs)
  - this process repeats again and again until the output layer is reached

## How To Choose Layers

- *input layer* -> is determined by the data you use
  - examples :
    - images of *30x30* pixels you will need *30x30=900* neurons
    - table with *10* columns you will need *10* neurons
- *output layer* -> is determined by the task you want to perform
  - examples :
    - categorize by male/female
      - *1 neuron*, 0 = male and 1 = female(0 and 1 are the output values)
      - *2 neurons*, first one male, second one female(percentages ???)
- *hidden layers*
  - *layers* learn features at different levels of abstraction(complexity)

<!-- N.B. for layers
- unwritten rule : it's better to have more layers(1) than neurons(2)
- it's better you have a deeper network(1) than a wider network(2)
- 1, 2 layers are ussually enough for most problems
- unritten rule : start with 2 layers then build your way up
- more layers = more levels of complexity
-->

### Dropout Layer

- *Dropout* layer -> randomly sets a fraction of input units to 0 at each update during training time
  - *Dropout(0.2)* -> 20% of the neurons will be set to 0
  - *Dropout(0.5)* -> 50% of the neurons will be set to 0
  - *Dropout(0.8)* -> 80% of the neurons will be set to 0
  - basically drops data to prevent overfitting

### More Info on Layers

- NN with no hidden layers is called *perceptron* **???** (linear model)
  - is a linear equation(doing linear regression)
  - resolves problems in a linear fashion
- Parameters = nr_of_weights + nr_of_biases
- How many layers do I need ?
  - plot the data and decide
  - whatever gives the best fit for the data
  - make the loss function traverse through the data smoothly **???**
  - training the data too much will lead to overfitting(learning the *training* data too well, and not generalizing well to new data)

### Abstract ???

### Multiple Linear Regression

yi = b0 + b1 * xi1 + b2 * xi2 + ... + bn * xin
yi - dependent variable(output/*prediction*)
b0 - *bias*(intercept)
b1, b2, ... bn - *weights*(coefficients)
xi1, xi2, ... xin - independent variables(*input neurons*/features)

### Mean Squared Error
<!-- this is the loss ? must read more about them -->
- error between the actual value and the predicted value squared and then averaged
- error = (1/n) * Σ(yi - ŷi)^2
- if the error is the lowest possible value then the model is the best possible model

### Otimizers
<!-- 'SGD'(stochastic gradient descent) is way faster than 'adam' -->

### Validation and Training Loss

[tutorial on loss](https://www.youtube.com/watch?v=p3CcfIjycBA)
<!-- N.B. to see changes in the plotting of these 2, use diff random_split values -->
- if the validation loss is lower than the training loss, it usually means that
  training data is harder to model than the validation data(or model is not good enough)