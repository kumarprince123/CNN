clc;
clear;
close all;
input=[];


  
W1=randn(64,676);%initialisation of weights
W2=randn(10,64);%initialisation of weights

w=[-1 -1 -1; 1 1 1;0 0 0]%initialisation of weights
w(:,:,2)=[-1 1 0; -1 1 0;-1 1 0];%initialisation of weights
w(:,:,3)=[0 0 0;1 1 1;1-1 -1 -1];%initialisation of weights
w(:,:,4)=w(:,:,3)';%initialisation of weights
AC=[];



%Convolutional Layer
for fer=1:30
for epoch=1:800
[z,target_value]=imgx(epoch);

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
 for r=1:1:4
 for i=1:2:26
     
  for j=1:2:26
      a=max(c1(i,j,1),c1(i+1,j,r));
      b=max(c1(i,j+1,r),c1(i+1,j+1,r));
      f(t,s,r)=max(a,b);
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

%back propagation
lr=0.005;
y_one_hot = zeros( 1, 10 );
r=target_value;
if(r==0)
    target_vector=[1 0 0 0 0 0 0 0 0 0]';
else
for i = 1:9
    rows = i == r;
    y_one_hot( i+1,rows) = 1;
end
target_vector=y_one_hot(:,1);
end

[d_output,d_hidden_layer,error_at_input]=back_propagation(y,target_vector,W2,hidden_op,W1);
W2=W2+((hidden_op*d_output')').*lr;
W1=W1+((input)*d_hidden_layer')'.*lr;

%back_propagation at convo layer
input_delta_error=reshape(error_at_input,13,13,4);

 for p=1:4
                k=1;
                m=1;
            for i=1:1:13 
    
                for j=1:1:13
        
                    pooling_error(m,k,p)= input_delta_error(i,j);
                    pooling_error(m+1,k,p)= input_delta_error(i,j);
                    pooling_error(m,k+1,p)= input_delta_error(i,j);
                    pooling_error(m+1,k+1,p)=input_delta_error(i,j);
                    k=k+2;
                end
                    k=1;
                    m=m+2;
            end
 end
dele_dash=zeros(28,28,4);
for k=1:4
        for i=1:26
            for j=1:26
                if(pooling_error(i,j,k)>0)
                    dele_dash(i,j,k)=pooling_error(i,j,k);
                else
                    dele_dash(i,j,k)=0;
                end
            end
        end
end

k=1;EW=[];

ew=0;
for k=1:4
    for a=0:2
        for b=0:2
            for i=1:26
                for j=1:26
                ew=ew+dele_dash(i,j,k)*im1(i+a,j+b);
                end
            end
                EW=[EW ew];
        end
    end
end


final_ew=reshape(EW,9,4)';
for i=1:4
w(:,:,i) = w(:,:,i)+ lr.*reshape(final_ew(i,:),3,3)';
end



 end
[yout,A]=test1(W1,W2,w);
AC=[AC A];

 
end
[accuracy,confusion_matrix]=test32(W1,W2)