clear all, close all

a1=0.446783749429806;
b1=0.306349472016557*2;
a2=0.508508655381127;
b2=0.51077156417211*2;
c=0.817627708322262;
%% Complie Mex file
m_handle = @MatAssem_Lap_Eqn_On_A_Surface;
MEX_File = 'DEMO_mex_Lap_Eqn_On_A_Surface';
[status, Path_To_Mex] = Convert_Form_Definition_to_MEX(m_handle,{a1,b1,a2,b2,c},MEX_File);
if status~=0
    disp('Compile did not succeed.');
    return;
end

objfile="bunny_8k_smooth.obj";
FEM_Soln = Execute_lap_Eqn_On_A_Surface(objfile);

ObjMesh=readObj(objfile);
TRI = ObjMesh.f.v;
VTX=ObjMesh.v;
figure(2)
trimesh(TRI, VTX(:,1)', VTX(:,3)', VTX(:,2)',FEM_Soln);
view([-30 10])
colorbar;
view([-90, 30, 10]);
colormap jet

x=VTX(:,1);y=VTX(:,2);z=VTX(:,3);
X=VTX;
% writematrix(VTX, "data"+"/Xt_8k.txt");
% writematrix(TRI, "data"+"/TRI_8k.txt");
addpath ./DM
k=40;
N=size(VTX,1);
f=10 * (x + y + z).^2-10;
% kappa=x.^2+y+z;
kappa=a1*x.^2+b1*y.^2+a2*x+b2*y+3+c;
[~,epsilon]=qestfind(X,k);
L_DM=dmapsgauss(X,k,kappa,epsilon); 
L=-L_DM+eye(N);
u_DM=L\f;
err=max(abs(u_DM-FEM_Soln))
mean_error = mean(abs((u_DM-FEM_Soln)./FEM_Soln))
max_error = max(abs((u_DM-FEM_Soln)./FEM_Soln))

figure(3)
trimesh(TRI, VTX(:,1)', VTX(:,3)', VTX(:,2)',u_DM);
view([-30 10])
colorbar;
view([-90, 30, 10]);
colormap jet