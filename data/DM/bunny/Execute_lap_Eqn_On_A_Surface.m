function [Soln] = Execute_lap_Eqn_On_A_Surface(objfile)
ObjMesh=readObj(objfile);
TRI=ObjMesh.f.v;
% vertices
VTX=ObjMesh.v;
Mesh = MeshTriangle(TRI,VTX, 'Gamma');
clear TRI VTX NEW_VTX;

% define function spaces (i.e. the DoFmaps)
Vh_DoFmap = uint32(Mesh.ConnectivityList);

% assemble
tic
[FEM] = DEMO_mex_Lap_Eqn_On_A_Surface([],Mesh.Points,uint32(Mesh.ConnectivityList),[],[],Vh_DoFmap);
toc
% put FEM into a nice object to make accessing the matrices easier
LB_Mats = FEMatrixAccessor('Laplace-Beltrami',FEM);
clear FEM;
Mass  = LB_Mats.Get_Matrix('Mass_Matrix');
Stiff = LB_Mats.Get_Matrix('Stiff_Matrix');
RHS  = LB_Mats.Get_Matrix('RHS');

disp('Solve the Laplacian Equation On A Surface:');
%Solve -div(kappa \nabla u)+u =f;
A=Stiff+Mass;
Soln = A \ RHS;
   
end