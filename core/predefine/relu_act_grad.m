function x = relu_act_grad ( x )

idx0 = x < 0;
idx1 = x > 0;

x (idx0) = 0;
x (idx1) = 1;

end