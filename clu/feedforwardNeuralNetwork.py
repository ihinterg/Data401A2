import numpy as np

class Loss(object):
    
    def __call__(self, predicted, actual):
        """Calculates the loss as a function of the prediction and the actual.
        
        Args:
          predicted (np.ndarray, float): the predicted output labels
          actual (np.ndarray, float): the actual output labels
          
        Returns: (float) 
          The value of the loss for this batch of observations.
        """
        raise NotImplementedError
        
    def derivative(self, predicted, actual):
        """The derivative of the loss with respect to the prediction.
        
        Args:
          predicted (np.ndarray, float): the predicted output labels
          actual (np.ndarray, float): the actual output labels
          
        Returns: (np.ndarray, float) 
          The derivatives of the loss.
        """
        raise NotImplementedError
        
        
class SquaredErrorLoss(Loss):

    def __call__(self, predicted, actual):
        return (predicted - actual) ** 2
    
    def derivative(self, predicted, actual):
        return 2 * (predicted - actual)
    
class ActivationFunction(object):
        
    def __call__(self, a):
        """Applies activation function to the values in a layer.
        
        Args:
          a (np.ndarray, float): the values from the previous layer (after 
            multiplying by the weights.
          
        Returns: (np.ndarray, float) 
          The values h = g(a).
        """
        return a
    
    def derivative(self, h):
        """The derivatives as a function of the outputs at the nodes.
        
        Args:
          h (np.ndarray, float): the outputs h = g(a) at the nodes.
          
        Returns: (np.ndarray, float) 
          The derivatives dh/da.
        """
        return 1
    
    
    
    
    
    
class ReLU(ActivationFunction):
    
    def __init__(self):
        import numpy as np

    def __call__(self, a):
        a = np.array(a)
        a[a < 0] = 0
        return a
    
    def derivative(self, h):
        h = np.array(h)
        h[h < 0] = 0
        h[h > 0] = 1
        return h


class Sigmoid(ActivationFunction):
    
    def __init__(self):
        import numpy as np
    
    def apply(self, x):
        return 1 / (1 + np.exp(-x))
    
    def __call__(self, a):
        return np.array([self.apply(i) for i in a])
        
    def derivative(self, h):
        return np.array([self.apply(i) * (1 - self.apply(i)) for i in h])
    
    

    
class Layer(object):
    """A data structure for a layer in a neural network.
    
    Attributes:
      num_nodes (int): number of nodes in the layer
      activation_function (ActivationFunction)
      values_pre_activation (np.ndarray, float): most recent values
        in layer, before applying activation function
      values_post_activation (np.ndarray, float): most recent values
        in layer, after applying activation function
    """
    
    def __init__(self, num_nodes, activation_function=ActivationFunction()):
        import numpy as np
        self.num_nodes = num_nodes
        self.activation_function = activation_function
        
    def get_layer_values(self, values_pre_activation):
        """Applies activation function to values from previous layer.
        
        Stores the values (both before and after applying activation 
        function)
        
        Args:
          values_pre_activation (np.ndarray, float): 
            A (batch size) x self.num_nodes array of the values
            in layer before applying the activation function
        
        Returns: (np.ndarray, float)
            A (batch size) x self.num_nodes array of the values
            in layer after applying the activation function
        """
        #print(values_pre_activation)
        
        self.values_pre_activation = values_pre_activation
        self.values_post_activation = np.array(self.activation_function(
            values_pre_activation
        ))
        return self.values_post_activation
        
        """
        self.values_post_activation = []
        self.values_pre_activation = values_pre_activation
        
        for i in self.values_pre_activation:
            #print(i)
            self.values_post_activation.append(self.activation_function(i))
        
        self.values_post_activation = np.array(self.values_post_activation)
        
        return self.values_post_activation
        """

        
class FullyConnectedNeuralNetwork(object):
    """A data structure for a fully-connected neural network.
    
    Attributes:
      layers (Layer): A list of Layer objects.
      loss (Loss): The loss function to use in training.
      learning_rate (float): The learning rate to use in backpropagation.
      weights (list, np.ndarray): A list of weight matrices,
        length should be len(self.layers) - 1
      biases (list, float): A list of bias terms,
        length should be equal to len(self.layers)
    """
    

    def __init__(self, layers, loss, learning_rate):
            
        import numpy as np
            
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        
        # initialize weight matrices and biases to NOT zeros
        self.weights = []
        self.biases = []
        for i in range(1, len(self.layers)):
            
            # he-et-al initialization
            import numpy as np
            self.weights.append(
                np.random.randn(self.layers[i - 1].num_nodes, self.layers[i].num_nodes) *\
                np.sqrt(2 / self.layers[i - 1].num_nodes)
            )
            self.biases.append(np.zeros((1, self.layers[i].num_nodes))[0])

        
    def validate(self, inputs, outputs):
        errs = []
        
        for i in range(len(inputs)):
            predicted = self.feedforward(inputs[i])
            errs.append(self.loss.__call__(predicted, outputs[i]))
        
        return errs
        
    def run_batch(self, inputs, outputs):
        errs = []
        
        for i in range(len(inputs)):
            predicted = self.feedforward(inputs[i])
            errs.append(self.loss.__call__(predicted, outputs[i]))
            self.backprop(predicted, outputs[i])
            
        return errs
    
    def feedforward(self, inputs):
        """Predicts the output(s) for a given set of input(s).
        
        Args:
          inputs (np.ndarray, float): A (batch size) x self.layers[0].num_nodes array
          
        Returns: (np.ndarray, float) 
          An array of the predicted output labels, length is the batch size
        """
        # TODO: Implement feedforward prediction.
        # Make sure you use Layer.get_layer_values() at each layer to store the values
        # for later use in backpropagation.
        
        for i in range(0, len(self.layers)):
            
            if i == 0:
                self.layers[i].get_layer_values(inputs)
            else:
                tmp = self.layers[i - 1].values_post_activation.dot(self.weights[i - 1]) + self.biases[i - 1]
                self.layers[i].get_layer_values(tmp)
               
        return self.layers[-1].values_post_activation

    def backprop(self, predicted, actual):
        """Updates self.weights and self.biases based on predicted and actual values.
        
        This will require using the values at each layer that were stored at the
        feedforward step.
        
        Args:
          predicted (np.ndarray, float): An array of the predicted output labels
          actual (np.ndarray, float): An array of the actual output labels
        """
        
        # ... i guess we're doing SGD.

        # TODO: Implement backpropagation.
        for i in range(len(self.layers) - 1, 0, -1):
    
            # dL / dw = dL / da * da / dz * dz / dw
            
            # if we are in the final layer
            if i == len(self.layers) - 1:

                # dL / da * da / dz = delta
                delta = self.loss.derivative(predicted, actual)

                # delta * dz / dw
                grad = np.outer(delta, self.layers[i].values_post_activation)

                self.weights[i - 1] -= self.learning_rate * grad
                
                # bias grads, assuming NO activation (fix if doing softmax)
                self.biases[i - 1] -= self.learning_rate * delta.reshape(self.biases[i-1].shape)
                self.layers[i].delta = delta
                
            else:
                # i represents weights from layer i to i + 1
                
                # dL / da
                etot = self.layers[i + 1].delta.dot(self.weights[i].T)
                
                # da / dz
                delta = self.layers[i].activation_function.derivative(self.layers[i].values_pre_activation)
                delta = etot * delta
                
                # delta * dz / dw
                grad = np.outer(delta, self.layers[i - 1].values_post_activation).T
                
                # update
                self.weights[i - 1] -= self.learning_rate * grad
                self.biases[i - 1] -= self.learning_rate * delta.reshape(self.biases[i-1].shape)
                
                self.layers[i].delta = delta
                
        
    def run_num_epochs(self, epochs, inputs, outputs, val_inputs, val_outputs, which = "single"):
        for i in range(epochs):
            errors = self.run_batch(inputs, outputs)
            if which == "single":
                print("Training MSE for epoch {}: {}".format(i + 1, round(np.sum(errors) / len(errors), 6)))
                valerr = self.validate(val_inputs, val_outputs)
                print("Validation MSE for epoch {}: {}".format(i + 1, round(np.sum(valerr) / len(valerr), 6)))
                print()
            elif which == "multiple":
                errors = np.array(errors)
                print("Training MSE for epoch {}: {}".format(i + 1, round(np.sum(errors) / len(errors), 6)))
                valerr = self.validate(val_inputs, val_outputs)
                print("Validation MSE for epoch {}: {}".format(i + 1, round(np.sum(valerr) / len(valerr), 6)))
                print()
                    
            #if i % (epochs // 10) == 0:
            #    print(np.sum(errors) / len(errors))
        
    def train(self, inputs, labels):
        """Trains neural network based on a batch of training data.
        
        Args:
          inputs (np.ndarray): A (batch size) x self.layers[0].num_nodes array
          labels (np.ndarray): An array of ground-truth output labels, 
            length is the batch size.
        """
        predicted = self.feedforward(inputs)
        self.backprop(predicted, labels)
        
    def predict(self, x):
        return self.feedforward(x)