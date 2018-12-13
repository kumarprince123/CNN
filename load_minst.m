images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-label.idx1-ubyte');
 
% We are using display_network from the autoencoder code
%display_network(images(:,1:1)); % Show the first 100 images
%disp(labels(1:10));
imshow(reshape(images(:,1),[28,28]));