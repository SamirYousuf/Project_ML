# Module for processing the data
import re, sys
import numpy as np
import pandas as pd

# Modules for CNN and LSTM network
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import LSTM

# Read dna_sequencing file from fasta format
def read_data_file(filename):
    return open(filename, "r")

# Pre processing the data to use in the model
def data_preprocess(filename):
    file = read_data_file(filename)
    data_ATG = []
    dna_data = [line for line in file]
    dna_data = "".join(dna_data)
    dna_data = dna_data.split(">")
    dna_data.pop(0)
    for x in range(len(dna_data)):
        if "ATG" in dna_data[x]:
            data_ATG.append(dna_data[x])
    print(len(data_ATG)) # Total number of sequences in the dataset with ATG

    # List of sequences from 5'UTR till start codon ATG
    ATG_data = []
    for x in range(len(data_ATG)):
        y = re.search(r'(\w+\|\w+\d+)([\n\w]+ATG)([\n\w]+)', data_ATG[x])
        if y:
            z = y.group(2)
            z = z.strip('\n')
            z = z.split()
            z = ''.join(z)
            ATG_data.append(z.lower())
    # Output is list of strings like this "aaaacatgaatgttgtgcattttgtgattttggaaatactcaatg"
    # List of sequences with each lenght of sequence 45 base units preceding the ATG
    # And the minimum cutoff is below 15 base units
    dna_seq_45bpMax = []
    for x in range(len(ATG_data)):
        if len(ATG_data[x]) >= 45:
            dna_seq_45bpMax.append(ATG_data[x][-45:])
        elif len(ATG_data[x]) <= 15:
            pass
        else:
            y,z = divmod(len(ATG_data[x]), 3)
            if z == 0:
                dna_seq_45bpMax.append(ATG_data[x])
            else:
                dna_seq_45bpMax.append(ATG_data[x][-(y*3):])
    print(len(dna_seq_45bpMax)) # Total lenght of the sequence which have length (45-15)

    # List of sequences where each sequence is split into 3 base pair
    # This base pairs are also called codon in genetics and it translate to protein
    list_of_list = []
    for x in range(len(dna_seq_150bpMax)):
        y1 = str(dna_seq_150bpMax[x])
        list_of_list.append([y1[i:i+3] for i in range(0, len(y1), 3)])
    list_of_seq = [' '.join(list_of_list[x]) for x in range(len(list_of_list))]

    # To anlyze the dataset, it is divided into two categories
    # Positive and Negative with binary values 1 and 0
    # This positive class are the sequences that don't have any stop codon
    # Negative class are the sequences which have one of the three stop codon
    # Stop codon are TAA, TAG, TGA
    atg_without_codon = [list_of_list[x] for x in range(len(list_of_list)) if "tag" not in list_of_list[x] and "taa" not in list_of_list[x] and "tga" not in list_of_list[x]]
    atg_with_stopcodon = [list_of_list[x] for x in range(len(list_of_list)) if "tag" in list_of_list[x] or "taa" in list_of_list[x] or "tga" in list_of_list[x]]
    class_positive = [1 for x in range(len(list_of_list)) if "tag" not in list_of_list[x] and "taa" not in list_of_list[x] and "tga" not in list_of_list[x]]
    class_negative = [0 for x in range(len(list_of_list)) if "tag" in list_of_list[x] or "taa" in list_of_list[x] or "tga" in list_of_list[x]]

    atg = atg_without_codon + atg_with_stopcodon # Total sequences with and without stop codon
    classes = class_positive + class_negative # Total labels with positive and negative class
    atg1 = [' '.join(atg[x]) for x in range(len(atg))] # modified list to use for encoding

    print(len(atg_without_codon), len(atg_with_stopcodon))
    print(len(class_positive), len(class_negative))
    print(len(atg), len(classes))

    # Dataframe of sequences and the labels to use for train and test split
    data_pd = pd.DataFrame(np.array(atg1))
    data_pd['labels'] = classes
    data_pd.columns = ["sequence", "labels"]
    data_pd = data_pd.sample(frac=1).reset_index(drop=True)
    X = data_pd["sequence"]
    Y = data_pd["labels"]
    split_length = int(len(X)*0.8)
    trainX = X[:split_length]
    testX = X[split_length:]
    trainy = Y[:split_length]
    testy = Y[split_length:]

    vocabulary_size = 64 # Total size of the vocabulary which is all the combination of A,C,T and G
    length = 15 # Maximum length of the sequence
    # One hot encoding for more expressive output for both train and test
    encoded_data = [one_hot(x, vocabulary_size) for x in trainX]
    trainXpad = pad_sequences(encoded_data, maxlen=length, padding="pre")
    encoded_data1 = [one_hot(x, vocabulary_size) for x in testX]
    testXpad = pad_sequences(encoded_data1, maxlen=length, padding="pre")
    return trainXpad, testXpad, trainy, testy

# CNN network takes input 4 arguments
# trainXpad, testXpad, trainy, testy
# Result accuracy and loss for convolution NN as well plot the result
def cnn_network(trainXpad, testXpad, trainy, testy):
    input_layer = Input(shape=(length,))
    embedded_layer = Embedding(vocabulary_size, 100)(input_layer)
    con1_layer = Conv1D(32, 8, activation="relu")(embedded_layer)
    dropout = Dropout(0.5)(con1_layer)
    pool = MaxPooling1D(pool_size=2)(dropout)
    flat = Flatten()(pool)
    dense = Dense(10, activation="relu")(flat)
    output = Dense(1, activation="sigmoid")(dense)

    model = Model(input_layer, output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    history = model.fit(trainXpad, trainy, epochs=50, batch_size=30, validation_data=(testXpad, testy))

    # Evaluation
    loss, accuracy = model.evaluate(trainXpad, trainy, verbose=0)
    print("Train accuracy = %f" %(accuracy*100))
    print("Train loss = %f" %(loss*100))

    loss1, accuracy1 = model.evaluate(testXpad, testy, verbose=0)
    print("Test accuracy = %f" %(accuracy1*100))
    print("Test loss = %f" %(loss1*100))

    # Plot for the accuracy of the model using CNN
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of the Model")
    plt.legend(["Training", "Testing"], loc="lower right")
    plt.show()

    # Plot for the Model-loss using CNN
    fig2, ax_loss = plt.subplots()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model- Loss")
    plt.legend(["Training", "Testing"], loc="upper right")
    plt.show()

# LSTM network takes input 4 arguments
# trainXpad, testXpad, trainy, testy
# Result accuracy and loss for RNN as well plot the result
# There are two LSTM layer in this network for better accuracy
def lstm_network(trainXpad, testXpad, trainy, testy):
    input_layer1 = Input(shape=(length,))
    embedded_layer1 = Embedding(vocabulary_size, 100)(input_layer1)
    lstm1 = LSTM(100, return_sequences=True)(embedded_layer1)
    lstm2 = LSTM(100)(lstm1)
    dropout1 = Dropout(0.5)(lstm2)
    dense1 = Dense(10, activation="relu")(dropout1)
    output1 = Dense(1, activation="sigmoid")(dense1)

    model1 = Model(input_layer1, output1)
    model1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model1.summary()
    history1 = model1.fit(trainXpad, trainy, epochs=50, batch_size=30, validation_data=(testXpad, testy))

    # Evaluation
    loss, accuracy = model.evaluate(trainXpad, trainy, verbose=0)
    print("Train accuracy = %f" %(accuracy*100))
    print("Train loss = %f" %(loss*100))
    
    loss1, accuracy1 = model.evaluate(testXpad, testy, verbose=0)
    print("Test accuracy = %f" %(accuracy1*100))
    print("Test loss = %f" %(loss1*100))

    # Plot for the accuracy of the model using LSTM network
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of the Model')
    plt.legend(['Training', 'Testing'], loc='lower right')
    plt.show()
    
    # Plot for the Model-loss using LSTM network
    fig2, ax_loss = plt.subplots()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.legend(['Training', 'Testing'], loc='upper right')
    plt.show()

if __name__ == "__main__":
    # Input is a dna fasta fromat file with geneID, transciptID and dna sequence
    filename = sys.argv[-1]
    # Preprocessing funciton take one argument i.e. the fasta file
    # Return trainX, testX, trainY and testY
    trainXpad, testXpad, trainy, testy = data_preprocess(filename)
    # CNN network
    cnn_network(trainXpad, testXpad, trainy, testy)
    # LSTM network
    lstm_network(trainXpad, testXpad, trainy, testy)

