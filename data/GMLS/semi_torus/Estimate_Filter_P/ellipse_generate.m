function [x,theta] = ellipse_generate(DATA_INDEX,am,N)

%%% intrinsic data
%%% 1 == well-sampled data
%%% 2 == random data
if DATA_INDEX == 1 % well-sampled
    theta = [0:2*pi/N:2*pi-2*pi/N]';    
elseif DATA_INDEX == 2 % random
    theta = rand(N,1)*2*pi;
    theta = sort(theta,'ascend');
end

%%% extrinsic data
x(:,1) = cos(theta);
x(:,2) = am*sin(theta);

 

end