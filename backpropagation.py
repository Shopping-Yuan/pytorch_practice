from typing import Any
import numpy as np



# model
class ML_Model:
    def __init__(self, input_dim=3, output_dim=1, feedforward_dim=[2,2,2] ):
        """
        Args:
            input_dim (:obj:`int`):
                number of input features
            output_dim (:obj:`int`):
                number of output predictions
            layers (:obj:`int):
                number of hidden layers in DNN
            feedforward_dim (:obj:`int`):˙
                number of neurons in each hidden layers
        """
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.ff_dim = feedforward_dim
        self.hlayers = len(feedforward_dim)
        
    def __call__(self, x):
        return self.forward(x)
            

    def forward(self, x):
        
        self.backwards = []

        # 1 : first  layer :
        # weight
        self.input_weights = np.random.random((self.ff_dim[0],self.in_dim))
        dim = self.ff_dim[0]
        # output
        output = (self.input_weights * np.array(x)).sum() # x = [1, 2, 3]
        self.backwards.append([output])

        # 2 : feedforward layer
        self.ff_weights = []

        for i in range(1,self.hlayers):
            # weight
            weights = np.random.random((self.ff_dim[i],dim))
            self.ff_weights.append(weights)
            dim = self.ff_dim[i]
            # output
            output = (weights* output).sum()
            self.backwards.append([output])

        self.ff_weights = np.array(self.ff_weights)
        
        # 3 : last layer
        # weight
        self.output_weights = np.random.random((self.out_dim,dim))
        # output
        output = (weights* output).sum()
        self.backwards.append([output])

        self.backwards = np.array(self.backwards) 
        return output,self.backwards
    
    # def loss(self,labels):
    #     l = 0
    #     for i in range(len(self.out_neurons)):
    #         l +=  (self.out_neurons[i]-labels[i])**2
    #     return (l)
    def __loss_diff__(self, labels,backwards):
        loss_diff_output = []
        for i in range(len(labels)):
            loss_diff_output.append([2 * (backwards[-1][i]-labels[i])])
        return loss_diff_output
    def get_diff(self,labels,backwards):
        output = self.__loss_diff__(labels,backwards)
        print(output)
        out_grad = []
        for i in range(len(backwards)):
            out_grad.append(output[i]*backwards[-1][i])
        return(np.array(out_grad))
    # def backward(self):



model = ML_Model()

x = [1, 2, 3]
y = model(x)
label = [5]
print(model.get_diff(label,y[1]))