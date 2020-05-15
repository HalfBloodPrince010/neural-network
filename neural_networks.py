import numpy as np
import cv2
import pandas as pd


class Neuralnetworks:
    def __init__(self, neurons, data, labels, n_layers, epocs, number_of_images):
        self.neurons = neurons
        self.data = data
        self.label = labels
        self.number_of_layers = n_layers
        self.weights = [[], [], []]
        self.signals = [[], [], []]
        self.epocs = epocs
        self.number_of_inputs = number_of_images
        self.inputs = [[], [], []]
        self.deltas = [[], [], []]
        self.learning_rate = 0.1

    def train(self):
        print("===========================RANDOM WEIGHTS==========================================")
        # Initialize weight matrix and signal vectors
        for i in range(1, 3):
            self.weights[i] = np.random.uniform(-0.1, 0.1, (self.neurons[i - 1] + 1, self.neurons[i]))
            print(self.weights[i].shape)
        print("=============================FEED FORWARD==========================================")
        for i in range(self.epocs):
            random_index = np.random.randint(1, self.number_of_inputs)
            print(random_index)
            self.feed_forward(random_index)
            print("=======================================BACK PROPAGATION================================================")
            self.back_propagation(random_index)
            print("=======================================UPDATE WEIGHTS==================================================")
            self.update_weights()
        print("=======================================FINAL WEIGHTS==================================================")
        print(self.weights[0])
        print(self.weights[1].shape)
        print(self.weights[2].shape)

    def feed_forward(self, randnum):
        initial_input = self.data[randnum]
        self.inputs[0] = initial_input
        print("Shape of Initial Weights", self.inputs[0].shape)
        print("Shape of Weight transposed[CHECK]", self.weights[1].T.shape)
        for layer in range(1, self.number_of_layers):
            self.signals[layer] = np.dot(self.weights[layer].T, self.inputs[layer-1])
            print("SIG", self.signals[layer].shape)
            sigmoid_input = sigmoid(self.signals[layer])
            # Last layer only 1 input, so
            if layer != self.number_of_layers-1:
                self.inputs[layer] = np.array([1] + sigmoid_input)
            else:
                self.inputs[layer] = np.array(sigmoid_input)
            print("input shape :", self.inputs[layer].shape, "input", self.inputs[layer])

    def back_propagation(self, randnum):
        for layer in range(self.number_of_layers - 1, 0, -1):
            if layer == self.number_of_layers-1:
                self.deltas[layer] = np.multiply(np.multiply(self.inputs[layer] - self.label[randnum], 2), np.multiply(self.inputs[layer], np.array([1]) - self.inputs[layer]))
                print("Delta at layer L:", self.deltas[layer])
            else:
                self.deltas[layer] = np.multiply(np.dot(self.deltas[layer + 1], self.weights[layer+1].T), np.multiply(self.inputs[layer], np.array([1]) - self.inputs[layer]))
                print("Delta Layer ", layer, ":", self.deltas[layer])

    def update_weights(self):
        for i in range(1, self.number_of_layers):
            if i == self.number_of_layers-1:
                self.weights[i] = self.weights[i] - np.multiply(self.learning_rate, np.outer(self.deltas[i], self.inputs[i-1]).T)
            else:
                self.weights[i] = self.weights[i] - np.multiply(self.learning_rate, np.outer(self.deltas[i][1:], self.inputs[i-1]).T)
        print("===========================================WEIGHTS=====================================================")
        print(self.weights[0])
        print(self.weights[1].shape)
        print(self.weights[2].shape)


def sigmoid(input_signals):
    sigmoid_signals = []
    for signal in input_signals:
        sigmoid_signals.append(1 / (1 + np.exp(-signal)))
    return sigmoid_signals


if __name__ == "__main__":
    pd_train_list = pd.read_csv('downgesture_train.list.txt', names=['image_list_train'])
    image = cv2.imread("./gestures/A/A_down_1.pgm", -1)
    image_array = []
    label_array = []
    for index, imageL in pd_train_list.iterrows():
        image_file_name = imageL['image_list_train']
        val = None
        if image_file_name.find('down') != -1:
            label_array.append(1)
            val = 1
        else:
            label_array.append(0)
            val = 0
        image = cv2.imread(image_file_name, -1)
        image = [1] + list(image.reshape(len(image)*len(image[0])))
        image_array.append(image)
        print("Image File Name:", imageL['image_list_train'], "\t label:", val, "\t shape:", len(image), "\tImage:", image)

    image_data = np.array(image_array)
    label = np.array(label_array)
    label = np.expand_dims(label, axis=1)
    print("===========================================================================================================")
    print("Image Data\n", image_data.shape[1])
    print("Image Label", label.shape)
    print(len(label))
    NN = Neuralnetworks([960, 100, 1], image_data, label, 3, 1000, len(label))
    NN.train()
