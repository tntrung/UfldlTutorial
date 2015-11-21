function g = softmaxGrad( X, y, theta, lambda )

numCases = size(X, 2);

groundTruth = full(sparse(y, 1:numCases, 1));

M = theta'*X;     % (numClasses,N)*(N,M)
M = bsxfun(@minus, M, max(M, [], 1));
h = exp(M);
h = bsxfun(@rdivide, h, sum(h));

g = -1/numCases*X*(groundTruth - h)' + lambda*theta;

end