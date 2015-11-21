function J = logregCost( X, y, theta, lambda )

m=size(X,2);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J = 1/m * ( - y * log( sigmoid(theta'*X)' ) - ( 1 - y ) * log( 1 - sigmoid(theta'*X) )'  ) + lambda/2/m*(sum(theta.^2) - theta(1)*theta(1));