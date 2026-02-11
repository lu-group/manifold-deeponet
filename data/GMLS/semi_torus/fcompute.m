function [f,c,u] = fcompute(THET,PHI,a)
%%% Input      
%%% THET        - intrinsic theta
%%% PHI         - intrinsic phi

%%% Output
%%% f           - force = div(c*grad u)-u
%%% u           - true solution

%%% analytic results
c = 1.1 + sin(THET).^2.*cos(PHI).^2; % nonconstant diffusion coefficient

%%%%% analytic results
% derivative of c 
c_t = 2*sin(THET).*cos(THET).*cos(PHI).^2;   % this will give you div(c \grad)
c_p = -2*sin(THET).^2.*sin(PHI).*cos(PHI);

% x = [(a+cos(THETA)).*cos(PHI), (a+cos(THETA)).*sin(PHI), sin(THETA)];
u = sin(PHI).*sin(THET);     % true solution 
u_t = sin(PHI).*cos(THET);
u_tt = sin(PHI).*(-sin(THET));
u_p = cos(PHI).*sin(THET);
u_pp = -sin(PHI).*sin(THET);

% g = [1 0; 0 (a+cos(THETA))^2];   % Riemannian metric
gsin = sin(THET);
gcos = a + cos(THET);

% Analytic Lu = f = -div(c \grad)+u
f = -( (-gsin.*c.*u_t + gcos.*c_t.*u_t + gcos.*c.*u_tt) + c_p./gcos.*u_p + c./gcos.*u_pp ) ./ gcos+u;

end