import numpy
import scipy.special

class neuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        self.lr = learningRate
        self.wih = (numpy.random.rand(self.hNodes, self.iNodes)-0.5)
        self.who = (numpy.random.rand(self.oNodes, self.hNodes)-0.5)

        self.activationFunction = lambda x:scipy.special.expit(x)
       
    def train(self,inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin = 2).T
        targets = numpy.array(targets_list, ndmin = 2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activationFunction(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activationFunction(final_inputs)


        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr * numpy.dot((output_errors*final_outputs*(1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors*hidden_outputs*(1.0 - hidden_outputs)), numpy.transpose(inputs))


    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin = 2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activationFunction(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activationFunction(final_inputs)

        return final_outputs

############################

input_nodes = 648 
hidden_nodes = 236
output_nodes = 2

learning_rate = 0.161

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)


training_data_file = open("train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

#train
epochs = 5

for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[:648])/255.0*0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01

        targets[int(all_values[648])] = 0.99
        n.train(inputs, targets)

test_data_file = open("test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#test
scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[648])
    inputs = (numpy.asfarray(all_values[:648])/255.0*0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)

    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)














