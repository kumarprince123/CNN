function [true ,matrixx]= test32(W1,W2)

%Convolutional Layer
temp=zeros(10,100);
temp1=zeros(10,100);
 true=0;
 
for e=1:100
images1 = loadMNISTImages('t10k-images.idx3-ubyte');
labels1= loadMNISTLabels('t10k-labels.idx1-ubyte');

z=reshape(images1(:,e),[28,28]);
w(:,:,1) =[6.7150   14.6020   22.5861;   32.5203   40.5817   48.6697;   55.8330   64.0406   72.2091];


w(:,:,2) =[ 78.9240   88.8111   95.7952;  102.7294  112.7908  119.8788;  127.0420  137.2497  144.4181];


w(:,:,3) =[  152.1331  160.0201  168.0042;  176.9385  184.9999  193.0878;  200.2511  207.4587  215.6272];


w(:,:,4) =[  224.3422  233.2292  240.2133;  248.1476  257.2089  263.2969;  272.4602  281.6678  287.8363];


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
 yout1=y;
 YOUT=max(yout1);
 target21=(labels1(:,:)');

 ta1=target21(1,e);
 yc = zeros( 1, 10 );
r=ta1;
if(r==0)
    target_vector=[1 0 0 0 0 0 0 0 0 0]';
else
for i = 1:9
    rows = i == r;
    yc( i+1,rows) = 1;
end
target_vector=yc(:,1);
end

temp(:,e)=target_vector;
temp1(:,e)=yout1;


if(yout1(r+1)==YOUT)
    true=true+1;
end


end
[p,matrixx]=confusion(temp,temp1);
end