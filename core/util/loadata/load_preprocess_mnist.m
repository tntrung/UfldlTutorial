function [data_train, labels_train, data_test, labels_test] = load_preprocess_mnist()
%% TODO ensure this is consistent with common loaders
% assumes relative paths to the common directory
% assumes common directory on paty for access to load functions
% adds 1 to the labels to make them 1-indexed

data_train = loadMNISTImages('/home/mangroup/Documents/TRUNG/MyCode/impCNN/data/train-images.idx3-ubyte');
labels_train = loadMNISTLabels(['/home/mangroup/Documents/TRUNG/MyCode/impCNN/data/train-labels.idx1-ubyte']);
labels_train  = labels_train + 1;

data_test = loadMNISTImages('/home/mangroup/Documents/TRUNG/MyCode/impCNN/data/t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels(['/home/mangroup/Documents/TRUNG/MyCode/impCNN/data/t10k-labels.idx1-ubyte']);
labels_test = labels_test + 1;

