import numpy as np
import Layer
import Activation_Function
import Optimizer
import Loss
import CS
import Accuracy
import pickle
import time

class CNN:

    def __init__(self):
        
        self.layers = []
        self.softmax_classifier_output = None
        self.loss = Loss.MSErr()
        self.optimizer = Optimizer.Adam(learning_rate=0.001,decay=1e-4)
        self.accuracy = Accuracy.Categorical()
    

    def forwardCV(self, data,training):
        self.input_layer.forward(data,training)
        for layer in self.layers[:3]:
            layer.forward(layer.prev.output,training)
        return layer.output

    def forwardFC(self, data, training):
        self.input_layer.forward(data,training)
        for layer in self.layers:
            layer.forward(layer.prev.output,training)
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        self.loss.backward(output, y)
        for layer in reversed(self.layers[4:]):
            layer.backward(layer.next.dinputs)
    def add(self, layer):
        self.layers.append(layer)

    def finalize(self):
        self.cv_output_layer=Layer.Output()
        self.input_layer = Layer.Input()
        layer_count = len(self.layers)
        self.trainable_layers = []
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        
        if self.loss is not None:
            self.loss.remember_trainable_layers(
                self.trainable_layers
            )
        if isinstance(self.layers[-1], Activation_Function.SMAF) and \
           isinstance(self.loss, Loss.CCent):
            self.SMAF_output = \
                Loss.SMAF_CCent()

    def train(self, images, labels, *, epochs=1, batch_size=None):
        self.accuracy.init(labels)
        train_steps = 1
        for epoch in range(1, epochs + 1):
            print(f'epoch: {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()
            for sample in range(len(images)):
                X = self.forwardCV(images[sample], training=False)
                if batch_size is not None:
                    train_steps = len(X) // batch_size
                    if train_steps * batch_size < len(X):
                        train_steps += 1
                    for step in range(train_steps):
                        batch_X = X[step * batch_size:(step + 1) * batch_size]
                        batch_y = labels[sample]
                        output = self.forwardFC(batch_X, training=True)
                        self.loss.calculate(output, batch_y, include_regularization=True)
                        predictions = self.output_layer_activation.predictions(output)
                        self.accuracy.calculate(predictions, batch_y)
                        step_data_loss, step_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
                        step_loss = step_data_loss + step_regularization_loss
                        print(f"\r"
                              f"{np.nan_to_num(batch_X / batch_X)}%\n"
                              f"circle confidence: {output[0, 0] * 100:.2f}%\n"
                              f"square confidence: {output[0, 1] * 100:.2f}%\n"
                              f"loss: {step_loss:.3f}", end="")
                        time.sleep(0.3)











        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model

shape_lables = {
 0: 'circle',
 1: 'square'}

data=CS.data()

images,labels=data.randset(40)
model=CNN()
model.add(Layer.Conv())
model.add(Layer.Pool())
model.add(Layer.flat())
model.add(Layer.Output())
model.add(Layer.Dense(576,288))
model.add(Activation_Function.SGAF())
model.add(Layer.Dense(288,144))
model.add(Activation_Function.SGAF())
model.add(Layer.Dense(144,2))
model.add(Activation_Function.SMAF())
model.finalize()
model.train(images,labels,epochs=10,batch_size=576)