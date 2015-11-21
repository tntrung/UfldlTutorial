% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../data;
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));
addpath(genpath('../core'));


%% load mnist data
% binary_digits = true;
% [train,test] = ex1_load_mnist(binary_digits);
% % Add row of 1s to the dataset to act as an intercept term.
% data_train = train.X; 
% data_test = test.X;
% labels_train = train.y+1; % make labels 1-based.
% labels_test = test.y+1; % make labels 1-based.

[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce 100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [256, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';

%% run training
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options, ei, data_train, labels_train);

%% check gradient
% average_error = grad_check(@supervised_dnn_cost, opt_params, 10, ei, data_train, labels_train );
% fprintf('gradient check error: %f\n', average_error);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);
