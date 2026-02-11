function [P0_tilde,T0_tilde] = geod_mean_normal_svd_v4_pick_V2(x,k,d,indxB)

%%% Inputs
    %%% x       - N-by-[n_R] data set with N data points in R^n_R
    %%% k       - number of nearest neighbors to use
    %%% d       - intrinsic dimension, must be integer
    %%% indxB   - the indices of points to be estimated
        
%%% Outputs
    %%% P0_tilde   - Tangential projection matrix, N*n*n

    
%%% Created by Shixiao Willing Jiang, Updated on Jun. 03, 2022    

N = size(x,1);
n = size(x,2);
NB = length(indxB);


if d > n % d <= n
    disp('d must be <= n');
    return;
end
if d == n
    P0_tilde = zeros(n,n,NB);  
    for i = 1:NB
        P0_tilde(:,:,i) = eye(n);
    end
    return;
end
if k <= (d+1)*d/2 % k > D = (d+1)*d/2
    disp('k must be > D = (d+1)*d/2');
    return;
end

[~,inds] = knnCPU(x,x(indxB,:),k); % NB*k
    
%% local SVD 

P0_tilde = zeros(n,n,NB);    
T0_tilde=zeros(n,NB,d);
for i = 1:NB
    
    temp = x(inds(i,:),:)'; % n*k
    tempmean = mean(temp,2); % n*1
    temp_rough = temp - repmat(tempmean,1,k); % n*k
    
    [U,S,~] = svd(temp_rough);
    [Sv,IX] = sort(diag(S),'descend');
    U = U(:,IX);
    
    %%% rough tangent space for representation of geodesic normal
    %%% coordiante
    t_rough = U(:,1:d); % n*d
    
    %%% need to determine if the surface locally is flat enough based on
    %%% singular value,
    if Sv(d+1)/Sv(d) < 1e-10 % then zero curvature, trust svd t_rough result
        
        P0_tilde(:,:,i) = t_rough*t_rough';
        
    else % curvature is nonzero
        
        %%% compute gamma(s) - gamma(0)
        tempx0 = temp(:,1); % n*1 x(indxB(i),:)'
        temp = temp - repmat(tempx0,1,k); % n*k
        %%% approximate si,sj
        si = temp'*t_rough; % k*d, including 1st zero
        %%% normalization of si to Euclidean distance
        norm_si = sqrt(sum(si.^2,2)); % k*1
        closepts = find(norm_si<1e-10); % avoid pts too close to x0
        norm_temp = sqrt(sum(temp.^2,1)); % 1*k
        si = si.*repmat(norm_temp'./norm_si,1,d);
        if ~isempty(closepts)
            si(closepts,1:d) = zeros(length(closepts),d);
        end
        %     keyboard,
        
        %%% compute aij = 2*si*sj (i~=j) and si^2 (i==j)
        D = d*(d+1)/2;
        aij = zeros(k,D); % aij is k*D
        for ii = 1:d
            aij(:,ii) = si(:,ii).^2; % i == j
        end
        ncount = d+1;
        for ii = 1:d
            for jj = ii+1:d
                aij(:,ncount) = 2*si(:,ii).*si(:,jj); % i ~= j
                ncount = ncount + 1;
            end
        end
        
        %%% compute Delta = 2*(x_knn - x0)
        Delta = 2*temp'; % k*n
        %%% estimate Hession terms, YD = Xij = d^2 gamma(s)/dsi dsj
        YD = (aij'*aij)\(aij'*Delta); % D*n
        %%% estimate the t_est = Delta - aij*YD
        t_est = (Delta - aij*YD)'; % n*k
        
        %%% svd for t_est
        [U2,S2,~] = svd(t_est);
        [~,IX2] = sort(diag(S2),'descend');
        U2 = U2(:,IX2);
        tangent = U2(:,1:d); % n*d
        T0_tilde(:,i,:)=tangent;
        %     keyboard,
        
        P0_tilde(:,:,i) = tangent*tangent';
    end
end

P0_tilde = permute(P0_tilde,[3,2,1]); % P0 is N*n*n

    
end


