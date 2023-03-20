import numpy as np

np.random.seed(0)# for testing purposes ensures that all random numbers stay the same

#DATA CREATOR
def create_data(points, classes):
    #points - amount of points on ~spiral
    #classes - amount of ~spirals
    #idk how the creator itself works I stole it (Data_creator.py)
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range (classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r*np.sin(t * 2.5), r*np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y

X, y = create_data(100, 3) 
# X(aka inputs) - lol of coordinates for 300 points on 3 ~spirals 
# y(aka target_vals) - spiral number for each point (correct output)


class Layer_Dense:#Neuron layer
    def __init__(self, n_inputs, n_neurons):
        # n_inputs - amount of data points for each point in this situation 2: the coordinates of the points on the ~spirals
        # n_neurons - amount of neurons
        self.weights = np.random.randn(n_inputs, n_neurons)# assings random weights for each input of each neuron (will be later tweaked by optimizer)
        self.biases = np.zeros((1, n_neurons))# assigns bias=0 to each neuron (will be later tweaked by optimizer)

    def forward(self, inputs):# calculates the output of each neuron and puts it in lol
        self.output = np.dot(inputs, self.weights) + self.biases#multiplies the input and weight and adds the bias for each neuron

'''
activation funcions modify the basic output of the neurons
allowing for another layer of control after weights and biases and are not tweaked by the optimizer
I use rectified linear for hidden layers as it is easier and softmax for output layer as it gives probabilities
'''
class Activation_ReLU: #Rectified Linear activation function
     def forward(self, inputs):   
        self.output = np.maximum(0, inputs)# if x > 0 return x; else return 0


class Activation_Softmax: #Softmax activation function
    def forward(self, inputs):#converts the output values into probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #Exponentiates the input values and subtracts from all values in row the max value of that row
        #The max row value is subtracted to protect from numbers too large as exp increases fast
        probabilities =  exp_values / np.sum(exp_values, axis=1, keepdims=True)
        #Normalises the exponentiated values by dividing each one by the sum of the row
        #norm_value = exp_value / row_sum
        self.output = probabilities

class Loss:#  Common loss class
    def calculate(self, output, y):
        #calculates the mean loss from a list of losses of single outputs 
        sample_losses = self.forward(output, y)
        batch_loss = np.mean(sample_losses)
        return(batch_loss)
    

class Loss_CCE(Loss):
    def forward(self,output_vals, target_vals):
        samples = len(output_vals)# finds amount of output samples
        output_vals_clipped = np.clip(output_vals, 1e-7, 1-1e-7)
        # clips the output values to get rid of 0.0 values
        if len(target_vals.shape) == 1:#determines if target values are given in the form of class values
            correct_confidences = output_vals_clipped[range(samples), target_vals]
            # takes only the confidences for the target true values
        elif len(target_vals.shape) == 2:#determines if target values are given in the form of one-hot vectors
            correct_confidenses = np.sum(output_vals_clipped * target_vals, axis=1)
            # takes only the confidences for the values that should be true 
        neg_log_probs = -np.log(correct_confidences)#finds the -ln() of the confidences
        return neg_log_probs
#----------------------------------------------------------------------------------------------------------

#DATA CREATION
X, y = create_data(100, 3)

#FIRST LAYER DEFINITION
dense1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

#SECOND LAYER DEFINITION
dense2 = Layer_Dense(5, 3)
activation2 = Activation_Softmax()

#FIRST LAYER ACTIVATION
dense1.forward(X)
activation1.forward(dense1.output)

#SECOND LAYER ACTIVATION
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print('NN OUTPUT :', activation2.output)# lol of predictions for each point

#LOSS CALCULATION
loss_function = Loss_CCE()
loss = loss_function.calculate(activation2.output, y)
print('LOSS : ', loss)


"""
ToDo:
- optimization
- backpropagation
- real data loading
"""
