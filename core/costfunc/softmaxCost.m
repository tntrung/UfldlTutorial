function J = softmaxCost( X, y, theta, lambda )

numCases = size(X, 2);

groundTruth = full(sparse(y, 1:numCases, 1));

M = theta'*X;
M = bsxfun(@minus, M, max(M, [], 1)); % to prevent overflow
h = exp(M);
h = bsxfun(@rdivide, h, sum(h));

J = -1/numCases*sum(sum(groundTruth.*log(h)))+lambda/2*sum(sum(theta.^2));

end