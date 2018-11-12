# Project_ML

ML project on non-coding DNA using CNN and LSTM

dna_sequence.py
It takes one argument (i.e. the human genome data file) 
data_processing() function preprocess the fasta format file and split the data in training and testing for both the sequences and its labels 
CNN network function takes 4 argument and result the accuracy and loss of the model
LSTM network function also takes 4 argument and result the accuracy and loss of the model

mart_export.txt
	This is the non coding fasta file downloaded from the ENSEMBL browser for human genome and preprocessed so only the 5’UTR data is used in this project


Sample Preprocessing Data :

>ENSG00000003137|ENST00000001146
AGGCAATTTTTTTCCTCCCTCTCTCCGCTCCCCTCGCAGCCTCCACTCCCTTTCCCTTGG
CCCCTTCCTCCTTCTCTGTTTCGGCTGGAGGTGCCAGGACCCCCGGCCGCAGCCTCCCCT
CCCCCGCCGCTCCGGTCCCCTCCCGTCGGGCCCTCCCCTCCCCCGCCGCGGCCGGCACAG
CCAATCCCCCGAGCGGCCGCCAACATGCTCTTTGAGGGCTTGGATCTGGTGTCGGCGCTG
GCCACCCTCGCCGCGTGCCTGGTGTCCGTGACGCTGCTGCTGGCCGTGTCGCAGCAGCTG
AGGCAATTTTTTTCCTCCCTCTCTCCGCTCCCCTCGCAGCCTCCACTCCCTTTCCCTTGG
CCCCTTCCTCCTTCTCTGTTTCGGCTGGAGGTGCCAGGACCCCCGGCCGCAGCCTCCCCT
CCCCCGCCGCTCCGGTCCCCTCCCGTCGGGCCCTCCCCTCCCCCGCCGCGGCCGGCACAG
CCAATCCCCCGAGCGGCCGCCAACATGCTCTTTGAGGGCTTGGATCTGGTGTCGGCGCTG
GCCACCCTCGCCGCGTGCCTGGTGTCCGTGACGCTGCTGCTGGCCGTGTCGCAGCAGCTG

CNN Convolution Neural Network

Accuracy and loss are calculated for both training and the testing data of the dna 5’UTR sequences using convolution neural network. An epochs of 50 and a batch size of 30 is used in model.fit to see how well the model perform. One good thing about CNN is that it is very fast compare to RNN.  The loss is calculated using the binary cross entropy and the optimiser used in this model as “adam”. Given below is the table with the result of accuracy and loss

RNN (LSTM network)

In this network, I used two LSTM layer to see how the accuracy improve compare to CNN. Here as well there is one input layer followed by embedded layer and then two LSTM layer and dropout and dense layer. This network is really very slow compare to the network above but the results shows that the model have a very significant improvement in accuracy. 

LSTM layer 1 also have a return_sequence=True so that it return a vector of dimension 100. 

In this project the aim was to understand and learn the sequences in the 5’UTR using CNN and LSTM network. The result shows that both the models given a significant performance on the dna dataset. LSTM network shows better result for both accuracy and loss on this dataset (train and test). This model can be used to any dna sequence data not only from human genome but other species as well to analyse and see how well the dataset are related and similar. 
