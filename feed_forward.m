function [ out1_first,out2_second] = feed_forward(input,W1,W2)

out1= W1*input;
out1_first= sigmoid(out1);
out2= W2*(out1_first);
out2_second= sigmoid(out2);
end