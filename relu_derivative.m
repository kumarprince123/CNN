function relu_derivative= relu_dash(x)
if(x)>0
    relu_derivative=1;
end
    if(x)<0
        relu_derivative=0;
    end
end