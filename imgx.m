function [z,target1]=imgx(w)

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

z=(reshape(images(:,w),[28,28])); 
target2=(labels(:,:)');
target1=(target2(1,w));

end