function x = relu (x) 

idx = x < 0 ;
x (idx) = 0;