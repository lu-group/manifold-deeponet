function [C] = kappa_u_xyz_v2(X, a1, a2, b1, b2)
    %%% Input      
    %%% X           - ambient data
    
    %%% Output
    %%% f           - force = -div(c*grad u)+u
    %%% c           - diffusion coefficients c
    %%% u           - true solution
    
    %% picewise kappa
    N=size(X,1);C=zeros(N,1); 
    for i=1:N
        x=X(i,1); 
        y=X(i,2); 
        z=X(i,3); 
    if x<=0 && y<=0
    C(i)=a1*x+b1*y+10;
    elseif x>0 && y<=0
    C(i)=a2*x+b1*y+10;
    elseif x<=0 && y>0
    C(i)=a1*x+b2*y+10;
    elseif x>0 && y>0
    C(i)=a2*x+b2*y+10;
    end
    end
    
    %% kappa function
    % syms x y z           % x,y,z cordinates in R^3
    % different expression of c could be chosen, but make sure c is positive for all x,y,z on torus
    %% Linear
    % c(x,y,z)=a*x+b*y+6+a1; 
    % %% exponential
    %c(x,y,z)=a*exp(x)+b*exp(y)+b1;
    %% higher order
    % c(x,y,z)=a1*x^2+b1*y^2+ a2*x + b2*y + 10;
    % 
    % cH=matlabFunction(c);
    % C=cH(X(:,1),X(:,2),X(:,3));
    
    
    end