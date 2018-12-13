function [relu_derivative] = relu_dash(x)


if((x)>0)
    relu_derivative=1;

else((x)<=0)
        
    relu_derivative=0;
    end
end