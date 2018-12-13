function [yout,ESUM]= test1(W1,W2,w)

%Convolutional Layer
esum=0;
 

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
for q=801:1000
z=reshape(images(:,q),[28,28]);


im1=im2double(z);

[m n]=size(im1);

for k=1:4
for x=1:26
    for y=1:26
        c(x,y,k)=w(1,1,k)*im1(x,y)+w(1,2,k)*im1(x,y+1)+w(1,3,k)*im1(x,y+2)+w(2,1,k)*im1(x+1,y)+w(2,2,k)*im1(x+1,y+1)+w(2,3,k)*im1(x+1,y+2)  ...
        +w(3,1,k)*im1(x+2,y)+w(3,2,k)*im1(x+2,y+1)+w(3,3,k)*im1(x+2,y+2);
            end
end
end




  %relu layer
  for i=1:4
   c1(:,:,i)=max(0,c(:,:,i));
    
end

  
  %POOLING
  t=1;
  s=1;
 for k=1:1:4
 for i=1:2:26
     
  for j=1:2:26
      a=max(c1(i,j,1),c1(i+1,j,k));
      b=max(c1(i,j+1,k),c1(i+1,j+1,k));
      f(t,s,k)=max(a,b);
      s=s+1;
  end
  s=1;
     t=t+1;
 end
 s=1;
 t=1;
  
 end
 %feeding forward
 input= (horzcat((reshape(f(:,:,1)',[],1))',(reshape(f(:,:,2),[],1))',(reshape(f(:,:,3),[],1))',(reshape(f(:,:,4)',[],1))'))';
 [hidden_op,y] = feed_forward(input,W1,W2);
 yout=y;
 
 target2=(labels(:,:)');

 ta=target2(1,q);
 yc = zeros( 1, 10 );
r=ta;
if(r==0)
    target_vector=[1 0 0 0 0 0 0 0 0 0]';
else
for i = 1:9
    rows = i == r;
    yc( i+1,rows) = 1;
end
target_vector=yc(:,1);
end

esum=esum+abs(sum((target_vector-y)));
end
ESUM=esum
end