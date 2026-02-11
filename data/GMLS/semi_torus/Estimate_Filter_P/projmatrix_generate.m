function Pmatrix = projmatrix_generate(ProjMat_INDEX,x, P0,d,LowPassFilter,RoughKernel)


%%% parameters for low pass filtering
%%% fixed bandwidth only
%%% operator == 3 is Laplace-Beltrami, operator == 4 is Kolmogorov backward
% nvarsDM = LowPassFilter.nvarsDM;
k = LowPassFilter.k;
operator = LowPassFilter.operator;
DinvThr = LowPassFilter.DinvThr;
% nlow = LowPassFilter.nlow; %%% number of low modes used
% epsilon = 0.004;
% dim = 1;

%%% parameters for tangent to surface
kk = RoughKernel.kk;
indxB = RoughKernel.indxB;


%%% 1 == true P0
%%% 2 == rough kernel P0_tilde
%%% 3 == P0_low
if ProjMat_INDEX == 1

    %%% give to Pmatrix
    Pmatrix = P0;

elseif ProjMat_INDEX == 2
    if isfield(RoughKernel,'x3') == 0 
        %%% estimate tangent and projection matrix P
%         [~,P0_tilde] = detect_s1_normal_plane_v1(x,indxB,kk,d);
        % [tangent,P0_tilde] = detect_s1_normal_plane_v1(x,RoughKernel.indxB, RoughKernel.kk,d);
        [P0_tilde] = geod_mean_normal_svd_v4_pick_V2(x,kk,d,indxB);
    elseif isfield(RoughKernel,'x3') == 1
        N1 = size(x,1);
        N3 = size(RoughKernel.x3,1);
        n = size(x,2);
        X3 = zeros(N1+N3,n);
        X3(1:N1,:) = x;
        X3(N1+1:N1+N3,:) = RoughKernel.x3;
%         [~,P0_tilde] = detect_s1_normal_plane_v1(X3,indxB,kk,d);
        [P0_tilde] = geod_mean_normal_svd_v4_pick_V2(X3,kk,d,indxB);
    end
    
    %%% give to Pmatrix
    Pmatrix = P0_tilde;
    
    
    
% elseif ProjMat_INDEX == 3
%     if isfield(RoughKernel,'x3') == 0 
%         %%% estimate tangent and projection matrix P
%         [~,P0_tilde] = detect_s1_normal_plane_v1(x,indxB,kk,d);
%         % [tangent,P0_tilde] = detect_s1_normal_plane_v1(x,RoughKernel.indxB, RoughKernel.kk,d);
%     elseif isfield(RoughKernel,'x3') == 1
%         N1 = size(x,1);
%         N3 = size(RoughKernel.x3,1);
%         n = size(x,2);
%         X3 = zeros(N1+N3,n);
%         X3(1:N1,:) = x;
%         X3(N1+1:N1+N3,:) = RoughKernel.x3;
%         [~,P0_tilde] = detect_s1_normal_plane_v1(X3,indxB,kk,d);
%     end
%     
%     %%% low-pass filter using DM 
%     [dlow] = FBDM_v3_matrix(x,k,operator,DinvThr);
%     [P0_low] = P0_filter_v6_pinv(P0_tilde,dlow,d);
%     
%     %%% give to Pmatrix
%     Pmatrix = P0_low;
    
end

