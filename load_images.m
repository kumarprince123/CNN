function z=image(i)

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
for i=1:1000
z=(reshape(images(:,i),[28,28])); % Show the first 100 images
%disp(labels(1:2));  
end
end