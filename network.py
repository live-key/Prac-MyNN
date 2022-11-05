import numpy as np

from layer import Layer
from activation import Activation
from loss import Loss, Loss_CCE

class Network:
    def __init__(self, layer_struct, activation):
        '''
        Inputs:
            array layer_struct =>   array containing the number of 
                                    neurons in each layer incl. in and out

                                    eg. [# of input, hidden, hidden, # of output]
                                        [784, 16, 16, 10]

            string activate =>      array containing the activation method
                                    for each layer

                                    eg. relu or soft
            
            myNN = Network(layer_struct=[784, 16, 16, 10], activation="relu")

            Note:   What's created is a network with len(layer_struct)-2    
                    hidden layers and 1 output layer
        '''

        self.layers = []
        for i in range(len(layer_struct) - 1):
            self.layers.append(Layer(n_ins=layer_struct[i], n_neurons=layer_struct[i+1]))
        
        self.activation = Activation(type=activation)

        ## Console Output ##
        print(f"Layers Created ({len(self.layers)}):")
        for i in range(len(self.layers)):
            print(f"Layer {i+1}:\n\tInputs: {self.layers[i].ins}\n\tNeurons: {self.layers[i].neurons}\n\tActivation: {self.activation.type}\n")

    def propogate(self, input_data, pred_data):

        for i in range(len(self.layers)):
            self.layers[i].forward(ins=input_data)
            self.activation.activate(ins=self.layers[i].res)

            input_data = self.activation.res
        
        print(self.activation.res[:5])
        
        loss_func = Loss_CCE()
        self.loss = loss_func.calc(self.activation.res, pred_data)
            
