function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+2, 1);
gradStack = cell(numHidden+1, 1);

%% set activation function
switch ei.activation_fun
    case 'logistic'
        act_fun = @sigmoid;
        act_fun_grad = @sigmoid_act_grad;
    case 'tanh'
        act_fun = @tanh;
        act_fun_grad = @tanh_act_grad;
    case 'relu'
        act_fun = @relu;
        act_fun_grad = @relu_act_grad;
end


%% forward prop
hAct{1} = data;

for i = 1:numHidden+1
    hAct{i+1} = bsxfun(@plus,stack{i}.W * hAct{i},stack{i}.b);
    if i == numHidden+1
        for_cost = hAct{i+1};
    end
    hAct{i+1} = act_fun(hAct{i+1});
end

M = exp(for_cost);
p = bsxfun(@rdivide, M, sum(M,1));

pred_prob = p;

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  return;
end;

%% compute cost

m = size(data, 2);
groundTruth = full(sparse(labels, 1:m, 1));
ceCost = -sum(sum(groundTruth.*log(p)));
delta  = p - groundTruth;

for i = numHidden+1:-1:1
    
    gradStack{i} = struct;
   
    if (i > numHidden)
        delta = delta.*ones(size(hAct{i+1}));
    else
        delta = delta.*act_fun_grad(hAct{i+1}); %% Fixed bug (not sigmoidGrad(hAct{i+1}))
    end
    
    gradStack{i}.W = delta * hAct{i}';
    gradStack{i}.b = sum(delta,2);
    
    delta = stack{i}.W' * delta;

end

%% compute weight penalty cost and gradient for non-bias terms
wCost = 0;

for i = 1:numHidden+1
    wCost = wCost + sum(stack{i}.W(:).^2);
end

cost = ceCost + .5 * ei.lambda * wCost;

for i = numHidden + 1 : -1 : 1
    gradStack{i}.W = gradStack{i}.W + ei.lambda * stack{i}.W;
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);

end