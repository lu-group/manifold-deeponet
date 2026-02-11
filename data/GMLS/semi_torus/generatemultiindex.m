function index =  generatemultiindex(N,dim)

% input
% N : max degree of polynomials
% dim : dimension

%N = 5;
%dim = 3;

P = nchoosek(N+dim,N);
index = zeros(dim,P);

Ntotal = (N+1)^dim;
allindex = zeros(dim,Ntotal);


for i=1:dim
    nskip = (N+1)^(dim-i);
    for k=1:Ntotal/nskip/(N+1)
        for j=1:N+1;
            allindex(i,(k-1)*nskip*(N+1)+(j-1)*nskip+1:(k-1)*nskip*(N+1)+j*nskip) = j-1;
        end
    end
    
end

index1 = find(sum(allindex)<=N);
index = allindex(:,index1);