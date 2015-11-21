function g = linregVecGrad( X, y , theta , lambda )

% so y_hat(i) = theta' * X(:,i).  Note that y_hat is a *row-vector*.
m = length(y);

y_hat = theta'*X; 

g = (lambda * (y_hat - y) * X' / m)';

% for i = 1 : dim
%     g(i) = alpha*((theta'*X - y)*X(i,:)')/m;
% end


end