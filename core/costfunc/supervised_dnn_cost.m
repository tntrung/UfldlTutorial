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

%% forward prop
hAct{1} = data;

for i = 1:numHidden+1
    hAct{i+1} = bsxfun(@plus,stack{i}.W * hAct{i},stack{i}.b);
    if i == numHidden+1
        for_cost = hAct{i+1};
    end
    hAct{i+1} = sigmoid(hAct{i+1});
end

pred_prob = hAct{numHidden+2};

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  return;
end;

%% compute cost
M = exp(for_cost);
p = bsxfun(@rdivide, M, sum(M,1));

m = size(data, 2);
groundTruth = full(sparse(labels, 1:m, 1));
ceCost = -sum(sum(groundTruth.*log(p)));
delta  = -(groundTruth - p);

for i = numHidden+1:-1:1
    
    gradStack{i} = struct;
    gradStack{i}.W = delta * hAct{i}';
    gradStack{i}.b = sum(delta,2);
    delta = (stack{i}.W'*delta).*sigmoidGrad(hAct{i});
    
end

%% compute weight penalty cost and gradient for non-bias terms
wCost = 0;

for i = 1:numHidden+1
    wCost = wCost + sum(stack{i}.W(:).^2);
end

cost = ceCost + .5 * ei.lambda * wCost ;

for i = numHidden + 1 : -1 : 1  
    gradStack{i}.W = gradStack{i}.W + ei.lambda * stack{i}.W;
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);

end



