function [L,W,dist]=GL(X,K,KNN)
[N,~]=size(X); dist=zeros(N,N);
for i=1:N
    for j=i+1:N
        dist(i,j)=norm(X(i,:)-X(j,:));
        %dist(i,j)=abs(X(i,:)-X(j,:));
    end
end
dist=dist+dist';
tau=zeros(N,1);
for i=1:N
    sorted=sort(dist(i,:)); tau(i)=sorted(K+1);
end
W=zeros(N,N);
if KNN==1
    for i=1:N
        for j=i+1:N
            if dist(i,j)<=max(tau(i),tau(j))
                W(i,j)=exp(-dist(i,j)^2/tau(i)/tau(j)/2);
            end              
        end
    end
else
    for i=1:N
        for j=i+1:N
            W(i,j)=exp(-dist(i,j)^2/tau(i)/tau(j)/2);
                      
        end
    end
end
W=W+W';
for i=1:N
    W(i,i)=1;
end
DD=zeros(N,N); 
for i=1:N
    DD(i,i)=1/sqrt(sum(W(i,:)));
end
L=round((eye(N)-DD*W*DD)/mean(tau)^2,10); 
% L=round(eye(N)-DD*W*DD,10); 



% D=zeros(N,N);
% for i=1:N
%     D(i,i)=sum(W(i,:));
% end
% %L=round(eye(N)-D^(-1)*W,10);
% L=round(D-W,10);

