function [d_output,d_hidden_layer,error_at_input]=back_propagation(y,target_vector,W2,hidden_op,W1)
lr=0.005;
error=(target_vector) - y;


slope_out_layer= sigmoid(y).*(1-sigmoid(y));
slope_hidden_layer= sigmoid(hidden_op).*(1-sigmoid(hidden_op));
d_output=error.*slope_out_layer.*lr;
error_at_hidden_layer=((d_output)')*W2;
d_hidden_layer=((error_at_hidden_layer)').*(slope_hidden_layer);
error_at_input= (((d_hidden_layer')*W1))';

end


 
 