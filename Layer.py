import time
from matplotlib.pyplot import box
import numpy as np
import cv2 as cv2
import CS
class Conv:
    def __init__(self,pad=0,stri=1):
        same=np.array([[-1,-1,-1],
                            [-1,8,-1],
                            [-1,-1,-1]])
        sharp=np.array([[ 0,-1, 0],
                    [-1, 4,-1],
                    [ 0,-1, 0]])    
        edge=np.array  ([[-1,0,1],
                        [-1,0,1],
                        [-1,0,1]])
        edge1=np.array([[1,1,1],
                    [0,0,0],
                    [-1,-1,-1]])
        diag=np.array([[-1, -1,2],
                        [-1, 2, -1],
                        [2,-1, -1]])
        diag1=np.array([ [ 2,-1,-1],
                            [-1,2,-1],
                            [-1,-1, 2]])
        grad=np.array([ [ 3,-1,-2],
                            [3,-1,-2],
                            [3,-1, -2]])
        grad1=np.array([ [ 3,3,3],
                            [-1,-1,-1],
                            [-2,-2, -2]])
        self.allk=[same,sharp,edge,edge1,diag,diag1,grad,grad1]
        self.pad=pad
        self.stri=stri
    def forward(self,inputs,training):
        if training==True:
            self.output=inputs
        else:    
            pad=self.pad
            stri=self.stri
            allk=self.allk
            im=inputs
            for run in range(len(allk)):
                    kernel=allk[run]
                    ker = np.flipud(np.fliplr(kernel))
                    xKernShape = ker.shape[0]
                    yKernShape = ker.shape[1]
                    xImgShape = im.shape[0]
                    yImgShape = im.shape[1]
                    xOutput = int(((xImgShape - xKernShape + 2 * self.pad) / self.stri) + 1)
                    yOutput = int(((yImgShape - yKernShape + 2 * pad) / stri) + 1)
                    output = np.zeros((xOutput,yOutput))
                    if pad != 0:
                        imPadded = np.zeros((im.shape[0] + pad*2, im.shape[1] + pad*2))
                        imPadded[int(pad):int(-1 * pad), int(pad):int(-1 * pad)] = im
                    else:
                        imPadded = im
                    for y in range(im.shape[1]):
                        if y > im.shape[1] - yKernShape:
                            break
                        if y % stri == 0:
                            for x in range(im.shape[0]):
                                if x > im.shape[0] - xKernShape:
                                    break
                                try:
                                    if x % stri == 0:
                                        output[x, y] = (ker * imPadded[x: x + xKernShape, y: y + yKernShape]).sum()
                                except:
                                    break           
                    output=np.atleast_3d(output)
                    if run==0:
                        box=output
                    if run>0:
                        box=np.append(box,output,axis=2)
            self.output=box
class Pool:
    def __init__(self):
        self.biases = np.zeros((1, 1))
    def forward(self, inputs,training):
        if training==True:
            self.output=inputs
        else:
            data=CS.tools.decode(inputs)
            run=0
            for im in data:
                section=np.ones((2,2)).astype(np.float16)
                arrx=int(im.shape[0])
                arry=int(im.shape[1])
                halfx=int(arrx/2)
                halfy=int(arry/2)
                output=np.zeros((halfx,halfy)).astype(np.float16)
                run=run+1
                for x in range(0,arrx,2):
                        if x>=arrx-1:
                            break
                        if x%1==0:
                            for y in range(0,arry,2):
                                if y>=arry-1:
                                    break
                                x1=x+2
                                y1=y+2
                                section=im[x:x1,y:y1].sum()
                                if section>1:
                                        i=int(x/2)
                                        j=int(y/2)
                                        output[i][j]=section
                output=np.atleast_3d(output)
                if run==1:
                    pooled=output
                if run>1:
                    pooled=np.append(pooled,output,axis=2).astype(np.float16)
            self.output=pooled
class flat:
    def __init__(self):
        self.biases = np.zeros((1, 1))
    def forward(self, inputs,training):
        if training==True:
            self.output=inputs   
        else:
            data=CS.tools.decode(inputs)
            vectors=np.empty((0))
            for im in data:
                vector=np.ravel(im)
                vectors=np.append(vectors,vector)
            self.output=vectors
class Input:
    def forward(self, inputs,training):
        self.output = inputs
class Output:
    def forward(self, inputs,training):
        self.output = np.atleast_2d(inputs)
class Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                             self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                            self.biases
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
    # Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases
    # Set weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
class Layer_Dropout:
    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate
    # Forward pass
    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs
        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
                           size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask
    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask
