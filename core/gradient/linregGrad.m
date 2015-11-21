function g = linregGrad ( X, y, theta, alpha )

m = length(y);
dim = size(X,1);
g=zeros(size(theta));

for i = 1 : dim
    g(i) = alpha*((theta'*X - y)*X(i,:)')/m;
end

end
