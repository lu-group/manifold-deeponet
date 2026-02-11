function [MATS] = MatAssem_Lap_Eqn_On_A_Surface(a1,b1,a2,b2,c)
%MatAssem_Heat_Eqn_On_A_Surface

% Copyright (c) 02-02-2015,  Shawn W. Walker

% define domain (2-D  closed surface in 3-D)
Gamma = Domain('triangle',3); % surface domain

%gf=GeoFunc(Gamma);
% define finite element spaces
Vh = Element(Gamma,lagrange_deg1_dim2,1); % piecewise linear on a surface mesh
gf = GeoFunc(Gamma);
% define functions on FE spaces
v = Test(Vh);
u = Trial(Vh);

% define FEM matrices
Mass_Matrix = Bilinear(Vh,Vh);
Mass_Matrix = Mass_Matrix + Integral(Gamma, v.val * u.val );

Stiff_Matrix = Bilinear(Vh,Vh);
kappa=a1*gf.X(1)^2+b1*gf.X(2)^2+a2*gf.X(1)+b2*gf.X(2)+3+c;
% kappa=gf.X(1)^2+gf.X(2)+gf.X(3);
% kappa=a*gf.X(1)+2*b*gf.X(2)+3+c;
force=0.1*(gf.X(1)+gf.X(2)+gf.X(3)).^2;

Stiff_Matrix = Stiff_Matrix + Integral(Gamma, kappa*v.grad' * u.grad );
%Stiff_Matrix2 = Stiff_Matrix + Integral(Gamma, u.grad' * (gf.Tangent_Space_Proj*[1;1;1])*v.val);
%Body_Force_Matrix = Linear(Vh);
%Body_Force_Matrix = Body_Force_Matrix + Integral(Gamma,v.val * sin(gf.X(1)) * cos(gf.X(2)) * sin(gf.X(3)));
RHS=Linear(Vh);
RHS=RHS+Integral(Gamma,force*v.val);

% set the minimum order of accuracy for the quad rule
Quadrature_Order = 3;
% define geometry representation - Domain, (default to piecewise linear)
G1 = GeoElement(Gamma);
% define a set of matrices
MATS = Matrices(Quadrature_Order,G1);

% collect all of the matrices together
MATS = MATS.Append_Matrix(Mass_Matrix);
MATS = MATS.Append_Matrix(Stiff_Matrix);
%MATS = MATS.Append_Matrix(Stiff_Matrix2);
MATS = MATS.Append_Matrix(RHS);
end