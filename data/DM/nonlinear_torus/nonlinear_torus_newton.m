clear all, close all

% Train
Nk = 200;
m=50;
Nm=m^2;
R=2; r=1;
flag = 1;
isTest = false;

% x - Theta and Phi, y - X Y Z, 
if flag == 1
    dx2=2*pi/m; test2=[dx2/2:dx2:2*pi-dx2/2]; [X,Y]=meshgrid(test2);
    x(1:Nm,1)=reshape(X,Nm,1); x(1:Nm,2)=reshape(Y,Nm,1);
elseif flag==2
    rng(0) %default
    test = rand(Nm,2);
    THET = 2*pi*(test(:,1));
    PHI = 2*pi*(test(:,2));
    x(1:Nm,1)=reshape(THET,Nm,1); x(1:Nm,2)=reshape(PHI,Nm,1);
end
y=zeros(Nm,3); y(:,1)=(2+cos(x(:,1))).*cos(x(:,2)); y(:,2)=(2+cos(x(:,1))).*sin(x(:,2)); y(:,3)=sin(x(:,1)); 

mc=26; Nmc=mc^2; 
dxc=2*pi/mc; testc=[dxc/2:dxc:2*pi-dxc/2]; [tc, pc]=meshgrid(testc);
xc(1:Nmc,1)=reshape(tc,Nmc,1); xc(1:Nmc,2)=reshape(pc,Nmc,1);
yc=zeros(Nmc,3); yc(:,1)=(2+cos(xc(:,1))).*cos(xc(:,2)); yc(:,2)=(2+cos(xc(:,1))).*sin(xc(:,2)); yc(:,3)=sin(xc(:,1));

us = zeros(Nk, Nm);
kappas_torus = zeros(Nk, Nm);
kappas = zeros(Nk, Nmc);
coefs = [];
for i = 1:Nk
    a = 1 + rand;
    disp([i, a]);
    u = a*cos(x(:,1));
    kappa = a*(R + r*cos(xc(:,1)));
    kappa_torus = a*(R + r*cos(x(:,1)));
    f = 3/2*u.^2+2*kappa_torus.*u-1/2*kappa_torus.^2 +u;

    X=y;
    N=Nm;
    c=kappa_torus;
    k=floor(1.5*sqrt(N));
    [d2,inds]=knnCPU(y,y,k);
    [~,epsilon]=qestfind(X,k);
    W=exp(-d2.^2/(4*epsilon)); 
    W=sparse(reshape(double(inds'),N*k,1),repmat(1:N,k,1),reshape(double(W'),N*k,1),N,N,N*k)';
    q=sum(W,2); K=W*diag(1./q).*(sqrt(c)*sqrt(c)'); 
    DD=diag(sum(K,2)); 
    L=(K-DD)/epsilon; 

    % Using the Newton's method 
    u0=u; % initial guess
    F0=(L)*u0+3/2*u0.^2+2*kappa_torus.*u0-1/2*kappa_torus.^2;
    tol=1e-5; max_it=20;it=0;
    err=max(abs(F0));
    while err>tol & it<max_it
        it = it + 1;
        grad_F=L+3*diag(u0)+2*kappa_torus.*eye(N); 
        du=-grad_F\(F0);
        u=du+u0;
        u0=u;
        F0=(L)*u0+3/2*u0.^2+2*kappa_torus.*u0-1/2*kappa_torus.^2;
    end
    err=max(abs(F0));
    % disp(['The residual of the discrete semiliear PDE is ', num2str(err)])
    us(i, :) = u0; 
    kappas(i, :) = kappa;
    kappas_torus(i, :) = kappa_torus;
    coefs = [coefs; i, a];
end

if isTest == false
    writematrix(coefs, "data/coefs_train.txt");
    writematrix(x, "data/tpt_train.txt")
    writematrix(xc, "data/tp_train.txt")
    writematrix(y, "data/Xt_train.txt")
    writematrix(yc, "data/X_train.txt")
    writematrix(kappas, "data/kappas_train.txt")
    writematrix(kappas_torus, "data/kappast_train.txt")
    writematrix(us, "data/us_train.txt")

%% test data
% save the data
else
    writematrix(coefs, "data/coefs_test.txt");
    writematrix(x, "data/tpt_test.txt")
    writematrix(xc, "data/tp_test.txt")
    writematrix(y, "data/Xt_test.txt")
    writematrix(yc, "data/X_test.txt")
    writematrix(kappas, "data/kappas_test.txt")
    writematrix(kappas_torus, "data/kappast_test.txt")
    writematrix(us, "data/us_test.txt")
end

clear all, close all

% Test
Nk = 1000;
m=50;
Nm=m^2;
R=2; r=1;
flag = 1;
isTest = true;

% x - Theta and Phi, y - X Y Z, 
if flag == 1
    dx2=2*pi/m; test2=[dx2/2:dx2:2*pi-dx2/2]; [X,Y]=meshgrid(test2);
    x(1:Nm,1)=reshape(X,Nm,1); x(1:Nm,2)=reshape(Y,Nm,1);
elseif flag==2
    rng(0) %default
    test = rand(Nm,2);
    THET = 2*pi*(test(:,1));
    PHI = 2*pi*(test(:,2));
    x(1:Nm,1)=reshape(THET,Nm,1); x(1:Nm,2)=reshape(PHI,Nm,1);
end
y=zeros(Nm,3); y(:,1)=(2+cos(x(:,1))).*cos(x(:,2)); y(:,2)=(2+cos(x(:,1))).*sin(x(:,2)); y(:,3)=sin(x(:,1)); 

mc=26; Nmc=mc^2; 
dxc=2*pi/mc; testc=[dxc/2:dxc:2*pi-dxc/2]; [tc, pc]=meshgrid(testc);
xc(1:Nmc,1)=reshape(tc,Nmc,1); xc(1:Nmc,2)=reshape(pc,Nmc,1);
yc=zeros(Nmc,3); yc(:,1)=(2+cos(xc(:,1))).*cos(xc(:,2)); yc(:,2)=(2+cos(xc(:,1))).*sin(xc(:,2)); yc(:,3)=sin(xc(:,1));

us = zeros(Nk, Nm);
kappas_torus = zeros(Nk, Nm);
kappas = zeros(Nk, Nmc);
coefs = [];
for i = 1:Nk
    a = 1 + rand;
    disp([i, a]);
    u = a*cos(x(:,1));
    kappa = a*(R + r*cos(xc(:,1)));
    kappa_torus = a*(R + r*cos(x(:,1)));
    f = 3/2*u.^2+2*kappa_torus.*u-1/2*kappa_torus.^2 +u;

    X=y;
    N=Nm;
    c=kappa_torus;
    k=floor(1.5*sqrt(N));
    [d2,inds]=knnCPU(y,y,k);
    [~,epsilon]=qestfind(X,k);
    W=exp(-d2.^2/(4*epsilon)); 
    W=sparse(reshape(double(inds'),N*k,1),repmat(1:N,k,1),reshape(double(W'),N*k,1),N,N,N*k)';
    q=sum(W,2); K=W*diag(1./q).*(sqrt(c)*sqrt(c)'); 
    DD=diag(sum(K,2)); 
    L=(K-DD)/epsilon; 

    % Using the Newton's method 
    u0=u; % initial guess
    F0=(L)*u0+3/2*u0.^2+2*kappa_torus.*u0-1/2*kappa_torus.^2;
    tol=1e-5; max_it=20;it=0;
    err=max(abs(F0));
    while err>tol & it<max_it
        it = it + 1;
        grad_F=L+3*diag(u0)+2*kappa_torus.*eye(N); 
        du=-grad_F\(F0);
        u=du+u0;
        u0=u;
        F0=(L)*u0+3/2*u0.^2+2*kappa_torus.*u0-1/2*kappa_torus.^2;
    end
    err=max(abs(F0));
    % disp(['The residual of the discrete semiliear PDE is ', num2str(err)])
    us(i, :) = u0; 
    kappas(i, :) = kappa;
    kappas_torus(i, :) = kappa_torus;
    coefs = [coefs; i, a];
end

if isTest == false
    writematrix(coefs, "data/coefs_train.txt");
    writematrix(x, "data/tpt_train.txt")
    writematrix(xc, "data/tp_train.txt")
    writematrix(y, "data/Xt_train.txt")
    writematrix(yc, "data/X_train.txt")
    writematrix(kappas, "data/kappas_train.txt")
    writematrix(kappas_torus, "data/kappast_train.txt")
    writematrix(us, "data/us_train.txt")

%% test data
% save the data
else
    writematrix(coefs, "data/coefs_test.txt");
    writematrix(x, "data/tpt_test.txt")
    writematrix(xc, "data/tp_test.txt")
    writematrix(y, "data/Xt_test.txt")
    writematrix(yc, "data/X_test.txt")
    writematrix(kappas, "data/kappas_test.txt")
    writematrix(kappas_torus, "data/kappast_test.txt")
    writematrix(us, "data/us_test.txt")
end