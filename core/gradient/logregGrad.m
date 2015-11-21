function g = logregGrad ( X, y, theta, alpha )

m=size(X,2);
g = zeros(size(theta));

g(1) = 1/m*((sigmoid(theta'*X) - y)*X(1,:)');

for i = 2 : size(theta);
    g(i) = 1/m*((sigmoid(theta'*X) - y)*X(i,:)'+alpha*theta(i));
end

end
