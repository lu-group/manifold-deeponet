function [Psi,Lambda,FakeZero] = RBFEig(LapMatrix,nvars,SvalThr)
%%% Inputs
    %%% LapMatrix      - Laplace Matrix N*N
    %%% nvars          - # of eigens
    %%% PhiInv         - determine the rank of matrix
    %%% SvalThr        - threashold for singular values
    
%%% Outputs
    %%% Psi            - eigenvectors
    %%% Lambda         - eigenvalues
    %%% FakeZero       - dimension of null space
    
%% detect dominant singular values     
%[~,S,~] = svd(PhiInv);
[~,S,~] = svd(LapMatrix);
S = diag(S);
S = sort(S,'ascend');
FakeZero = length(find(S<SvalThr));

N=size(LapMatrix,2);
%% eigs for non-symmetric matrix
%%% find eigenvalues closest to 1 since all eigenvalues theoretically are
%%% negative definite.
[psi,lambda] = eig(LapMatrix);

%[psi,lambda] = eigs(LapMatrix,nvars,1);
lambda = diag(lambda); % diagonal to vector
[~,perm] = sort(abs(lambda),'ascend');
lambda = lambda(perm);
psi = psi(:,perm);

%%% collect the true eigenvalues and eigenvectors
%% real part 
if FakeZero==0
    FakeZero=1;
end
Lambda = (lambda(FakeZero:nvars));
Psi = (psi(:,FakeZero:nvars));


a=find(Lambda>1e-4);
ntrue=1:length(Lambda);
ntrue(a)=[];
Lambda=Lambda(ntrue);
Psi =Psi(:,ntrue);
%Lambda(1)=0;
%Psi(:,1)=ones(N,1);

end