function P0 = P0Cheat_ellipse(theta,x,am)

N = size(x,1);
n = size(x,2);

P0 = zeros(n,n,N);

%%% tangent direction
tvec = zeros(N,n);
tvec(:,1) = - sin(theta);
tvec(:,2) = am*cos(theta);


for ii = 1:N
    P0(:,:,ii) = tvec(ii,:)'*pinv(tvec(ii,:)*tvec(ii,:)')*tvec(ii,:);
end
P0 = permute(P0,[3,2,1]); % P0 is N*n*n

end