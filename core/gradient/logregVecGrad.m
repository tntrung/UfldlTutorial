function g = logregVecGrad ( X, y, theta, lambda )

m=size(X,2);
g = zeros(size(theta));

g(1) = 1/m*((sigmoid(theta'*X) - y)*X(1,:)');

y_hat = sigmoid(theta'*X); 

g(2:end) = (((y_hat - y) * X(2:end,:)')'  + lambda*theta(2:end))/m;

end