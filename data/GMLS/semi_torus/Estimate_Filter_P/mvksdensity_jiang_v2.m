function [fx,pts,bw] = mvksdensity_jiang_v2(x,pts,bw,maxMem)

%%% Inputs
    %%% x       - Mxd data set with N data points in R^d
    %%% pts     - Nxd points for estimation
    %%% bw      - 1xd bandwidth
    
%%% Outputs
    %%% fx      - Nx1 pdf
    %%% pts     - Nxd is the input, pts 

    if (nargin<4)
        maxMem = 2;
    end

    M = size(x,1);
    N = size(pts,1);   
    d = size(x,2);
    
        
    %%% compute bandwidth vector
    if (nargin < 3)
        bw = sqrt(var(x,1,1)) * (4/(d+2)/M)^(1/(d+4)); % 1st 1 means divided by N, 2nd 1 mean first dimension
    end    


    
    maxArray = (maxMem*2500)^2;
    blockSize = floor(maxArray/M);
    blocks = floor(N/blockSize);
    
    fx = zeros(N,1);
    
    Nr = sum(x.^2./repmat(bw.^2,M,1), 2);
    Nq = sum(pts.^2./repmat(bw.^2,N,1), 2);
    
    for b = 1:blocks
        dtemp = -2*x*(pts((b-1)*blockSize + 1:b*blockSize,:)./repmat(bw.^2,blockSize,1))';
        dtemp = bsxfun(@plus,dtemp,Nr);
        dtemp = bsxfun(@plus,dtemp,Nq((b-1)*blockSize + 1:b*blockSize)'); % dtemp = M*blockSize
        [row,col]=find(dtemp<40);
        kernel = exp(- dtemp((col-1)*M+row) / 2);
        KERNEL = sparse(col,1:length(col),kernel);
        % fx((b-1)*blockSize + 1:b*blockSize,1) = KERNEL*ones(length(col),1);
        fx((b-1)*blockSize + 1:(b-1)*blockSize + size(KERNEL,1),1) = KERNEL*ones(length(col),1);
    end
    
    if (blocks*blockSize < N)
        dtemp = -2*x*(pts(blocks*blockSize + 1:N,:)./repmat(bw.^2,N-blocks*blockSize,1))';
        dtemp = bsxfun(@plus,dtemp,Nr);
        dtemp = bsxfun(@plus,dtemp,Nq(blocks*blockSize + 1:N)');        
        [row,col]=find(dtemp<40);
        kernel = exp(- dtemp((col-1)*M+row) / 2);
        KERNEL = sparse(col,1:length(col),kernel);
        % fx(blocks*blockSize + 1:N,1) = KERNEL*ones(length(col),1); 
        fx(blocks*blockSize + 1:blocks*blockSize + size(KERNEL,1),1) = KERNEL*ones(length(col),1); 
    end

    
    %%% compute kernel density estimation, fx = zeros(N,1);
    fx = 1/(M*prod(bw)) * 1/(2*pi)^(d/2) * fx;
    
    
end


