/*
============================================================================================
   Methods for a C++ Class that does generic finite element assembly.

   NOTE: portions of this code are automatically generated!

   Copyright (c) 04-08-2010,  Shawn W. Walker
============================================================================================
*/

#define GFA Generic_FEM_Assembly

/***************************************************************************************/
/* constructor */
GFA::GFA (const mxArray *prhs[], const mxArray *Subset_Elem)
{
    /*------------ BEGIN: Auto Generate ------------*/
    // setup inputs
    Domain_Gamma_embedded_in_Gamma_restricted_to_Gamma.Setup_Data(prhs[PRHS_Gamma_Mesh_Subdomains], prhs[PRHS_Gamma_Mesh_DoFmap], Subset_Elem);
    Domain_Gamma_embedded_in_Gamma_restricted_to_d_Gamma.Setup_Data(prhs[PRHS_Gamma_Mesh_Subdomains], prhs[PRHS_Gamma_Mesh_DoFmap], Subset_Elem);
    Domain_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma.Setup_Data(prhs[PRHS_Gamma_Mesh_Subdomains], prhs[PRHS_Gamma_Mesh_DoFmap], Subset_Elem);

    geom_Gamma_embedded_in_Gamma_restricted_to_Gamma.Setup_Mesh_Geometry(prhs[PRHS_Gamma_Mesh_Vertices], prhs[PRHS_Gamma_Mesh_DoFmap], prhs[PRHS_Gamma_Mesh_Orient]);
    geom_Gamma_embedded_in_Gamma_restricted_to_Gamma.Domain = &Domain_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma.Setup_Mesh_Geometry(prhs[PRHS_Gamma_Mesh_Vertices], prhs[PRHS_Gamma_Mesh_DoFmap], prhs[PRHS_Gamma_Mesh_Orient]);
    geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma.Domain = &Domain_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma.Setup_Mesh_Geometry(prhs[PRHS_Gamma_Mesh_Vertices], prhs[PRHS_Gamma_Mesh_DoFmap], prhs[PRHS_Gamma_Mesh_Orient]);
    geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma.Domain = &Domain_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;

    Setup_Data(prhs);
    /*------------   END: Auto Generate ------------*/
}
/***************************************************************************************/


/***************************************************************************************/
/* destructor */
GFA::~GFA ()
{
    // clear it
    mxFree(Sparse_Data_CG_M.name);
    mxFree(Sparse_Data_Div_Matrix.name);
    mxFree(Sparse_Data_L2_div_sig_soln_sq.name);
    mxFree(Sparse_Data_L2_p_soln_sq.name);
    mxFree(Sparse_Data_L2_sig_soln_sq.name);
    mxFree(Sparse_Data_N_DOT_sigma.name);
    mxFree(Sparse_Data_P_Mass_Matrix.name);
    mxFree(Sparse_Data_Proj_P_MAT.name);
    mxFree(Sparse_Data_Proj_Sigma_MAT.name);
    mxFree(Sparse_Data_RHS_Matrix.name);
    mxFree(Sparse_Data_Sigma_Mass_Matrix.name);
}
/***************************************************************************************/


/***************************************************************************************/
/* setup matrix data into a nice struct for internal use */
void GFA::Setup_Data(const mxArray *prhs[]) // input from MATLAB
{
    // access previously assembled matrices (if they exist)
    /*------------ BEGIN: Auto Generate ------------*/
    Access_Previous_FEM_Matrix(prhs[PRHS_OLD_FEM], "CG_M", 0, Sparse_Data_CG_M);
    Access_Previous_FEM_Matrix(prhs[PRHS_OLD_FEM], "Div_Matrix", 1, Sparse_Data_Div_Matrix);
    Access_Previous_FEM_Matrix(prhs[PRHS_OLD_FEM], "L2_div_sig_soln_sq", 2, Sparse_Data_L2_div_sig_soln_sq);
    Access_Previous_FEM_Matrix(prhs[PRHS_OLD_FEM], "L2_p_soln_sq", 3, Sparse_Data_L2_p_soln_sq);
    Access_Previous_FEM_Matrix(prhs[PRHS_OLD_FEM], "L2_sig_soln_sq", 4, Sparse_Data_L2_sig_soln_sq);
    Access_Previous_FEM_Matrix(prhs[PRHS_OLD_FEM], "N_DOT_sigma", 5, Sparse_Data_N_DOT_sigma);
    Access_Previous_FEM_Matrix(prhs[PRHS_OLD_FEM], "P_Mass_Matrix", 6, Sparse_Data_P_Mass_Matrix);
    Access_Previous_FEM_Matrix(prhs[PRHS_OLD_FEM], "Proj_P_MAT", 7, Sparse_Data_Proj_P_MAT);
    Access_Previous_FEM_Matrix(prhs[PRHS_OLD_FEM], "Proj_Sigma_MAT", 8, Sparse_Data_Proj_Sigma_MAT);
    Access_Previous_FEM_Matrix(prhs[PRHS_OLD_FEM], "RHS_Matrix", 9, Sparse_Data_RHS_Matrix);
    Access_Previous_FEM_Matrix(prhs[PRHS_OLD_FEM], "Sigma_Mass_Matrix", 10, Sparse_Data_Sigma_Mass_Matrix);
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // setup access to FE basis functions
    P_Space_phi_restricted_to_Gamma.Setup_Function_Space(prhs[PRHS_P_Space_DoFmap]);
    P_Space_phi_restricted_to_Gamma.Mesh = &geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    Sigma_Space_phi_restricted_to_Gamma.Setup_Function_Space(prhs[PRHS_Sigma_Space_DoFmap]);
    Sigma_Space_phi_restricted_to_Gamma.Mesh = &geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    V_Space_phi_restricted_to_Gamma.Setup_Function_Space(prhs[PRHS_V_Space_DoFmap]);
    V_Space_phi_restricted_to_Gamma.Mesh = &geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    P_Space_phi_restricted_to_d_Gamma.Setup_Function_Space(prhs[PRHS_P_Space_DoFmap]);
    P_Space_phi_restricted_to_d_Gamma.Mesh = &geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Sigma_Space_phi_restricted_to_d_Gamma.Setup_Function_Space(prhs[PRHS_Sigma_Space_DoFmap]);
    Sigma_Space_phi_restricted_to_d_Gamma.Mesh = &geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // setup correct number of components for CONSTANT basis functions
    L2_div_sig_soln_sq_Gamma_col_constant_phi.Num_Comp = 1;
    L2_div_sig_soln_sq_Gamma_row_constant_phi.Num_Comp = 1;
    L2_p_soln_sq_Gamma_col_constant_phi.Num_Comp = 1;
    L2_p_soln_sq_Gamma_row_constant_phi.Num_Comp = 1;
    L2_sig_soln_sq_Gamma_col_constant_phi.Num_Comp = 1;
    L2_sig_soln_sq_Gamma_row_constant_phi.Num_Comp = 1;
    N_DOT_sigma_Gamma_col_constant_phi.Num_Comp = 1;
    N_DOT_sigma_Gamma_row_constant_phi.Num_Comp = 1;
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // setup access to external FE functions
    p_coef_restricted_to_Gamma.Setup_Function_Space(prhs[PRHS_p_coef_Values], &P_Space_phi_restricted_to_Gamma);
    sig_coef_restricted_to_Gamma.Setup_Function_Space(prhs[PRHS_sig_coef_Values], &Sigma_Space_phi_restricted_to_Gamma);
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // setup the base matrices to compute
    Base_Matrix_CG_M = new Base_CG_M_Data_Type(&V_Space_phi_restricted_to_Gamma, &V_Space_phi_restricted_to_Gamma);
    Base_Matrix_Div_Matrix = new Base_Div_Matrix_Data_Type(&P_Space_phi_restricted_to_Gamma, &Sigma_Space_phi_restricted_to_Gamma);
    Base_Matrix_L2_div_sig_soln_sq = new Base_L2_div_sig_soln_sq_Data_Type(&L2_div_sig_soln_sq_Gamma_row_constant_phi, &L2_div_sig_soln_sq_Gamma_col_constant_phi);
    Base_Matrix_L2_p_soln_sq = new Base_L2_p_soln_sq_Data_Type(&L2_p_soln_sq_Gamma_row_constant_phi, &L2_p_soln_sq_Gamma_col_constant_phi);
    Base_Matrix_L2_sig_soln_sq = new Base_L2_sig_soln_sq_Data_Type(&L2_sig_soln_sq_Gamma_row_constant_phi, &L2_sig_soln_sq_Gamma_col_constant_phi);
    Base_Matrix_N_DOT_sigma = new Base_N_DOT_sigma_Data_Type(&N_DOT_sigma_Gamma_row_constant_phi, &N_DOT_sigma_Gamma_col_constant_phi);
    Base_Matrix_P_Mass_Matrix = new Base_P_Mass_Matrix_Data_Type(&P_Space_phi_restricted_to_Gamma, &P_Space_phi_restricted_to_Gamma);
    Base_Matrix_Proj_P_MAT = new Base_Proj_P_MAT_Data_Type(&V_Space_phi_restricted_to_Gamma, &P_Space_phi_restricted_to_Gamma);
    Base_Matrix_Proj_Sigma_MAT = new Base_Proj_Sigma_MAT_Data_Type(&V_Space_phi_restricted_to_Gamma, &Sigma_Space_phi_restricted_to_Gamma);
    Base_Matrix_RHS_Matrix = new Base_RHS_Matrix_Data_Type(&Sigma_Space_phi_restricted_to_d_Gamma, &P_Space_phi_restricted_to_d_Gamma);
    Base_Matrix_Sigma_Mass_Matrix = new Base_Sigma_Mass_Matrix_Data_Type(&Sigma_Space_phi_restricted_to_Gamma, &Sigma_Space_phi_restricted_to_Gamma);
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // pass pointers around
    Base_Matrix_CG_M->geom_Gamma_embedded_in_Gamma_restricted_to_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    Base_Matrix_CG_M->geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_CG_M->geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_CG_M->P_Space_phi_restricted_to_Gamma = &P_Space_phi_restricted_to_Gamma;
    Base_Matrix_CG_M->P_Space_phi_restricted_to_d_Gamma = &P_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_CG_M->Sigma_Space_phi_restricted_to_Gamma = &Sigma_Space_phi_restricted_to_Gamma;
    Base_Matrix_CG_M->Sigma_Space_phi_restricted_to_d_Gamma = &Sigma_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_CG_M->V_Space_phi_restricted_to_Gamma = &V_Space_phi_restricted_to_Gamma;
    Base_Matrix_CG_M->p_coef_restricted_to_Gamma = &p_coef_restricted_to_Gamma;
    Base_Matrix_CG_M->sig_coef_restricted_to_Gamma = &sig_coef_restricted_to_Gamma;
    Base_Matrix_Div_Matrix->geom_Gamma_embedded_in_Gamma_restricted_to_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    Base_Matrix_Div_Matrix->geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_Div_Matrix->geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_Div_Matrix->P_Space_phi_restricted_to_Gamma = &P_Space_phi_restricted_to_Gamma;
    Base_Matrix_Div_Matrix->P_Space_phi_restricted_to_d_Gamma = &P_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_Div_Matrix->Sigma_Space_phi_restricted_to_Gamma = &Sigma_Space_phi_restricted_to_Gamma;
    Base_Matrix_Div_Matrix->Sigma_Space_phi_restricted_to_d_Gamma = &Sigma_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_Div_Matrix->V_Space_phi_restricted_to_Gamma = &V_Space_phi_restricted_to_Gamma;
    Base_Matrix_Div_Matrix->p_coef_restricted_to_Gamma = &p_coef_restricted_to_Gamma;
    Base_Matrix_Div_Matrix->sig_coef_restricted_to_Gamma = &sig_coef_restricted_to_Gamma;
    Base_Matrix_L2_div_sig_soln_sq->geom_Gamma_embedded_in_Gamma_restricted_to_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    Base_Matrix_L2_div_sig_soln_sq->geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_L2_div_sig_soln_sq->geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_L2_div_sig_soln_sq->P_Space_phi_restricted_to_Gamma = &P_Space_phi_restricted_to_Gamma;
    Base_Matrix_L2_div_sig_soln_sq->P_Space_phi_restricted_to_d_Gamma = &P_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_L2_div_sig_soln_sq->Sigma_Space_phi_restricted_to_Gamma = &Sigma_Space_phi_restricted_to_Gamma;
    Base_Matrix_L2_div_sig_soln_sq->Sigma_Space_phi_restricted_to_d_Gamma = &Sigma_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_L2_div_sig_soln_sq->V_Space_phi_restricted_to_Gamma = &V_Space_phi_restricted_to_Gamma;
    Base_Matrix_L2_div_sig_soln_sq->p_coef_restricted_to_Gamma = &p_coef_restricted_to_Gamma;
    Base_Matrix_L2_div_sig_soln_sq->sig_coef_restricted_to_Gamma = &sig_coef_restricted_to_Gamma;
    Base_Matrix_L2_p_soln_sq->geom_Gamma_embedded_in_Gamma_restricted_to_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    Base_Matrix_L2_p_soln_sq->geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_L2_p_soln_sq->geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_L2_p_soln_sq->P_Space_phi_restricted_to_Gamma = &P_Space_phi_restricted_to_Gamma;
    Base_Matrix_L2_p_soln_sq->P_Space_phi_restricted_to_d_Gamma = &P_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_L2_p_soln_sq->Sigma_Space_phi_restricted_to_Gamma = &Sigma_Space_phi_restricted_to_Gamma;
    Base_Matrix_L2_p_soln_sq->Sigma_Space_phi_restricted_to_d_Gamma = &Sigma_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_L2_p_soln_sq->V_Space_phi_restricted_to_Gamma = &V_Space_phi_restricted_to_Gamma;
    Base_Matrix_L2_p_soln_sq->p_coef_restricted_to_Gamma = &p_coef_restricted_to_Gamma;
    Base_Matrix_L2_p_soln_sq->sig_coef_restricted_to_Gamma = &sig_coef_restricted_to_Gamma;
    Base_Matrix_L2_sig_soln_sq->geom_Gamma_embedded_in_Gamma_restricted_to_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    Base_Matrix_L2_sig_soln_sq->geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_L2_sig_soln_sq->geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_L2_sig_soln_sq->P_Space_phi_restricted_to_Gamma = &P_Space_phi_restricted_to_Gamma;
    Base_Matrix_L2_sig_soln_sq->P_Space_phi_restricted_to_d_Gamma = &P_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_L2_sig_soln_sq->Sigma_Space_phi_restricted_to_Gamma = &Sigma_Space_phi_restricted_to_Gamma;
    Base_Matrix_L2_sig_soln_sq->Sigma_Space_phi_restricted_to_d_Gamma = &Sigma_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_L2_sig_soln_sq->V_Space_phi_restricted_to_Gamma = &V_Space_phi_restricted_to_Gamma;
    Base_Matrix_L2_sig_soln_sq->p_coef_restricted_to_Gamma = &p_coef_restricted_to_Gamma;
    Base_Matrix_L2_sig_soln_sq->sig_coef_restricted_to_Gamma = &sig_coef_restricted_to_Gamma;
    Base_Matrix_N_DOT_sigma->geom_Gamma_embedded_in_Gamma_restricted_to_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    Base_Matrix_N_DOT_sigma->geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_N_DOT_sigma->geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_N_DOT_sigma->P_Space_phi_restricted_to_Gamma = &P_Space_phi_restricted_to_Gamma;
    Base_Matrix_N_DOT_sigma->P_Space_phi_restricted_to_d_Gamma = &P_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_N_DOT_sigma->Sigma_Space_phi_restricted_to_Gamma = &Sigma_Space_phi_restricted_to_Gamma;
    Base_Matrix_N_DOT_sigma->Sigma_Space_phi_restricted_to_d_Gamma = &Sigma_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_N_DOT_sigma->V_Space_phi_restricted_to_Gamma = &V_Space_phi_restricted_to_Gamma;
    Base_Matrix_N_DOT_sigma->p_coef_restricted_to_Gamma = &p_coef_restricted_to_Gamma;
    Base_Matrix_N_DOT_sigma->sig_coef_restricted_to_Gamma = &sig_coef_restricted_to_Gamma;
    Base_Matrix_P_Mass_Matrix->geom_Gamma_embedded_in_Gamma_restricted_to_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    Base_Matrix_P_Mass_Matrix->geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_P_Mass_Matrix->geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_P_Mass_Matrix->P_Space_phi_restricted_to_Gamma = &P_Space_phi_restricted_to_Gamma;
    Base_Matrix_P_Mass_Matrix->P_Space_phi_restricted_to_d_Gamma = &P_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_P_Mass_Matrix->Sigma_Space_phi_restricted_to_Gamma = &Sigma_Space_phi_restricted_to_Gamma;
    Base_Matrix_P_Mass_Matrix->Sigma_Space_phi_restricted_to_d_Gamma = &Sigma_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_P_Mass_Matrix->V_Space_phi_restricted_to_Gamma = &V_Space_phi_restricted_to_Gamma;
    Base_Matrix_P_Mass_Matrix->p_coef_restricted_to_Gamma = &p_coef_restricted_to_Gamma;
    Base_Matrix_P_Mass_Matrix->sig_coef_restricted_to_Gamma = &sig_coef_restricted_to_Gamma;
    Base_Matrix_Proj_P_MAT->geom_Gamma_embedded_in_Gamma_restricted_to_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    Base_Matrix_Proj_P_MAT->geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_Proj_P_MAT->geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_Proj_P_MAT->P_Space_phi_restricted_to_Gamma = &P_Space_phi_restricted_to_Gamma;
    Base_Matrix_Proj_P_MAT->P_Space_phi_restricted_to_d_Gamma = &P_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_Proj_P_MAT->Sigma_Space_phi_restricted_to_Gamma = &Sigma_Space_phi_restricted_to_Gamma;
    Base_Matrix_Proj_P_MAT->Sigma_Space_phi_restricted_to_d_Gamma = &Sigma_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_Proj_P_MAT->V_Space_phi_restricted_to_Gamma = &V_Space_phi_restricted_to_Gamma;
    Base_Matrix_Proj_P_MAT->p_coef_restricted_to_Gamma = &p_coef_restricted_to_Gamma;
    Base_Matrix_Proj_P_MAT->sig_coef_restricted_to_Gamma = &sig_coef_restricted_to_Gamma;
    Base_Matrix_Proj_Sigma_MAT->geom_Gamma_embedded_in_Gamma_restricted_to_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    Base_Matrix_Proj_Sigma_MAT->geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_Proj_Sigma_MAT->geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_Proj_Sigma_MAT->P_Space_phi_restricted_to_Gamma = &P_Space_phi_restricted_to_Gamma;
    Base_Matrix_Proj_Sigma_MAT->P_Space_phi_restricted_to_d_Gamma = &P_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_Proj_Sigma_MAT->Sigma_Space_phi_restricted_to_Gamma = &Sigma_Space_phi_restricted_to_Gamma;
    Base_Matrix_Proj_Sigma_MAT->Sigma_Space_phi_restricted_to_d_Gamma = &Sigma_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_Proj_Sigma_MAT->V_Space_phi_restricted_to_Gamma = &V_Space_phi_restricted_to_Gamma;
    Base_Matrix_Proj_Sigma_MAT->p_coef_restricted_to_Gamma = &p_coef_restricted_to_Gamma;
    Base_Matrix_Proj_Sigma_MAT->sig_coef_restricted_to_Gamma = &sig_coef_restricted_to_Gamma;
    Base_Matrix_RHS_Matrix->geom_Gamma_embedded_in_Gamma_restricted_to_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    Base_Matrix_RHS_Matrix->geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_RHS_Matrix->geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_RHS_Matrix->P_Space_phi_restricted_to_Gamma = &P_Space_phi_restricted_to_Gamma;
    Base_Matrix_RHS_Matrix->P_Space_phi_restricted_to_d_Gamma = &P_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_RHS_Matrix->Sigma_Space_phi_restricted_to_Gamma = &Sigma_Space_phi_restricted_to_Gamma;
    Base_Matrix_RHS_Matrix->Sigma_Space_phi_restricted_to_d_Gamma = &Sigma_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_RHS_Matrix->V_Space_phi_restricted_to_Gamma = &V_Space_phi_restricted_to_Gamma;
    Base_Matrix_RHS_Matrix->p_coef_restricted_to_Gamma = &p_coef_restricted_to_Gamma;
    Base_Matrix_RHS_Matrix->sig_coef_restricted_to_Gamma = &sig_coef_restricted_to_Gamma;
    Base_Matrix_Sigma_Mass_Matrix->geom_Gamma_embedded_in_Gamma_restricted_to_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    Base_Matrix_Sigma_Mass_Matrix->geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_Sigma_Mass_Matrix->geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma = &geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    Base_Matrix_Sigma_Mass_Matrix->P_Space_phi_restricted_to_Gamma = &P_Space_phi_restricted_to_Gamma;
    Base_Matrix_Sigma_Mass_Matrix->P_Space_phi_restricted_to_d_Gamma = &P_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_Sigma_Mass_Matrix->Sigma_Space_phi_restricted_to_Gamma = &Sigma_Space_phi_restricted_to_Gamma;
    Base_Matrix_Sigma_Mass_Matrix->Sigma_Space_phi_restricted_to_d_Gamma = &Sigma_Space_phi_restricted_to_d_Gamma;
    Base_Matrix_Sigma_Mass_Matrix->V_Space_phi_restricted_to_Gamma = &V_Space_phi_restricted_to_Gamma;
    Base_Matrix_Sigma_Mass_Matrix->p_coef_restricted_to_Gamma = &p_coef_restricted_to_Gamma;
    Base_Matrix_Sigma_Mass_Matrix->sig_coef_restricted_to_Gamma = &sig_coef_restricted_to_Gamma;
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // setup the block matrices to assemble
    Block_Assemble_Matrix_CG_M = new Block_Assemble_CG_M_Data_Type(&Sparse_Data_CG_M, Base_Matrix_CG_M);
    Block_Assemble_Matrix_Div_Matrix = new Block_Assemble_Div_Matrix_Data_Type(&Sparse_Data_Div_Matrix, Base_Matrix_Div_Matrix);
    Block_Assemble_Matrix_L2_div_sig_soln_sq = new Block_Assemble_L2_div_sig_soln_sq_Data_Type(&Sparse_Data_L2_div_sig_soln_sq, Base_Matrix_L2_div_sig_soln_sq);
    Block_Assemble_Matrix_L2_p_soln_sq = new Block_Assemble_L2_p_soln_sq_Data_Type(&Sparse_Data_L2_p_soln_sq, Base_Matrix_L2_p_soln_sq);
    Block_Assemble_Matrix_L2_sig_soln_sq = new Block_Assemble_L2_sig_soln_sq_Data_Type(&Sparse_Data_L2_sig_soln_sq, Base_Matrix_L2_sig_soln_sq);
    Block_Assemble_Matrix_N_DOT_sigma = new Block_Assemble_N_DOT_sigma_Data_Type(&Sparse_Data_N_DOT_sigma, Base_Matrix_N_DOT_sigma);
    Block_Assemble_Matrix_P_Mass_Matrix = new Block_Assemble_P_Mass_Matrix_Data_Type(&Sparse_Data_P_Mass_Matrix, Base_Matrix_P_Mass_Matrix);
    Block_Assemble_Matrix_Proj_P_MAT = new Block_Assemble_Proj_P_MAT_Data_Type(&Sparse_Data_Proj_P_MAT, Base_Matrix_Proj_P_MAT);
    Block_Assemble_Matrix_Proj_Sigma_MAT = new Block_Assemble_Proj_Sigma_MAT_Data_Type(&Sparse_Data_Proj_Sigma_MAT, Base_Matrix_Proj_Sigma_MAT);
    Block_Assemble_Matrix_RHS_Matrix = new Block_Assemble_RHS_Matrix_Data_Type(&Sparse_Data_RHS_Matrix, Base_Matrix_RHS_Matrix);
    Block_Assemble_Matrix_Sigma_Mass_Matrix = new Block_Assemble_Sigma_Mass_Matrix_Data_Type(&Sparse_Data_Sigma_Mass_Matrix, Base_Matrix_Sigma_Mass_Matrix);
    /*------------   END: Auto Generate ------------*/
}
/***************************************************************************************/


/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* here is the main calling routine for assembling all FEM matrices */
void GFA::Assemble_Matrices ()
{
    // BEGIN: assemble matrices over the Integration Domain: Gamma

    if (Domain_Gamma_embedded_in_Gamma_restricted_to_Gamma.Sub_Assem_List.empty())
        {
        mexPrintf("This integration domain is empty: Gamma\n");
        mexPrintf(" ... so no assembly necessary.\n");
        }

    // loop through each element
    for (std::vector<unsigned int>::iterator DoI_Ind = Domain_Gamma_embedded_in_Gamma_restricted_to_Gamma.Sub_Assem_List.begin();
              DoI_Ind != Domain_Gamma_embedded_in_Gamma_restricted_to_Gamma.Sub_Assem_List.end(); ++DoI_Ind)
        {
        Domain_Gamma_embedded_in_Gamma_restricted_to_Gamma.Read_Embed_Data(*DoI_Ind);

        // get the local simplex transformation
        geom_Gamma_embedded_in_Gamma_restricted_to_Gamma.Compute_Local_Transformation();

        // perform pre-computations with FE basis functions
        // NOTE: this must come before the external FE coefficient functions
        P_Space_phi_restricted_to_Gamma.Transform_Basis_Functions();
        Sigma_Space_phi_restricted_to_Gamma.Transform_Basis_Functions();
        V_Space_phi_restricted_to_Gamma.Transform_Basis_Functions();

        // perform pre-computations with external FE coefficient functions
        p_coef_restricted_to_Gamma.Compute_Func();
        sig_coef_restricted_to_Gamma.Compute_Func();

        // loop through the FE matrices to compute
        Base_Matrix_CG_M->Tabulate_Tensor(geom_Gamma_embedded_in_Gamma_restricted_to_Gamma);
        Base_Matrix_Div_Matrix->Tabulate_Tensor(geom_Gamma_embedded_in_Gamma_restricted_to_Gamma);
        Base_Matrix_L2_div_sig_soln_sq->Tabulate_Tensor(geom_Gamma_embedded_in_Gamma_restricted_to_Gamma);
        Base_Matrix_L2_p_soln_sq->Tabulate_Tensor(geom_Gamma_embedded_in_Gamma_restricted_to_Gamma);
        Base_Matrix_L2_sig_soln_sq->Tabulate_Tensor(geom_Gamma_embedded_in_Gamma_restricted_to_Gamma);
        Base_Matrix_N_DOT_sigma->Tabulate_Tensor(geom_Gamma_embedded_in_Gamma_restricted_to_Gamma);
        Base_Matrix_P_Mass_Matrix->Tabulate_Tensor(geom_Gamma_embedded_in_Gamma_restricted_to_Gamma);
        Base_Matrix_Proj_P_MAT->Tabulate_Tensor(geom_Gamma_embedded_in_Gamma_restricted_to_Gamma);
        Base_Matrix_Proj_Sigma_MAT->Tabulate_Tensor(geom_Gamma_embedded_in_Gamma_restricted_to_Gamma);
        Base_Matrix_Sigma_Mass_Matrix->Tabulate_Tensor(geom_Gamma_embedded_in_Gamma_restricted_to_Gamma);

        // loop through the block FE matrices to assemble
        Block_Assemble_Matrix_CG_M->Add_Entries_To_Global_Matrix_Gamma(Base_Matrix_CG_M);
        Block_Assemble_Matrix_Div_Matrix->Add_Entries_To_Global_Matrix_Gamma(Base_Matrix_Div_Matrix);
        Block_Assemble_Matrix_L2_div_sig_soln_sq->Add_Entries_To_Global_Matrix_Gamma(Base_Matrix_L2_div_sig_soln_sq);
        Block_Assemble_Matrix_L2_p_soln_sq->Add_Entries_To_Global_Matrix_Gamma(Base_Matrix_L2_p_soln_sq);
        Block_Assemble_Matrix_L2_sig_soln_sq->Add_Entries_To_Global_Matrix_Gamma(Base_Matrix_L2_sig_soln_sq);
        Block_Assemble_Matrix_N_DOT_sigma->Add_Entries_To_Global_Matrix_Gamma(Base_Matrix_N_DOT_sigma);
        Block_Assemble_Matrix_P_Mass_Matrix->Add_Entries_To_Global_Matrix_Gamma(Base_Matrix_P_Mass_Matrix);
        Block_Assemble_Matrix_Proj_P_MAT->Add_Entries_To_Global_Matrix_Gamma(Base_Matrix_Proj_P_MAT);
        Block_Assemble_Matrix_Proj_Sigma_MAT->Add_Entries_To_Global_Matrix_Gamma(Base_Matrix_Proj_Sigma_MAT);
        Block_Assemble_Matrix_Sigma_Mass_Matrix->Add_Entries_To_Global_Matrix_Gamma(Base_Matrix_Sigma_Mass_Matrix);
        }
    // END: assemble matrices over the Integration Domain: Gamma

    // BEGIN: assemble matrices over the Integration Domain: d_Gamma

    if (Domain_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma.Sub_Assem_List.empty())
        {
        mexPrintf("This integration domain is empty: d_Gamma\n");
        mexPrintf(" ... so no assembly necessary.\n");
        }

    // loop through each element
    for (std::vector<unsigned int>::iterator DoI_Ind = Domain_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma.Sub_Assem_List.begin();
              DoI_Ind != Domain_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma.Sub_Assem_List.end(); ++DoI_Ind)
        {
        Domain_Gamma_embedded_in_Gamma_restricted_to_d_Gamma.Read_Embed_Data(*DoI_Ind);
        Domain_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma.Read_Embed_Data(*DoI_Ind);

        // get the local simplex transformation
        geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma.Compute_Local_Transformation();
        geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma.Compute_Local_Transformation();

        // perform pre-computations with FE basis functions
        // NOTE: this must come before the external FE coefficient functions
        P_Space_phi_restricted_to_d_Gamma.Transform_Basis_Functions();
        Sigma_Space_phi_restricted_to_d_Gamma.Transform_Basis_Functions();


        // loop through the FE matrices to compute
        Base_Matrix_RHS_Matrix->Tabulate_Tensor(geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma);

        // loop through the block FE matrices to assemble
        Block_Assemble_Matrix_RHS_Matrix->Add_Entries_To_Global_Matrix_d_Gamma(Base_Matrix_RHS_Matrix);
        }
    // END: assemble matrices over the Integration Domain: d_Gamma

}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* this outputs all FEM matrices as MATLAB sparse matrices */
void GFA::Output_Matrices (mxArray* plhs[])
{
    // declare internal matrix data storage pointer
    mxArray* Sparse_ptr;

    // create sparse MATLAB matrices and pass them back to MATLAB

    Sparse_ptr = Block_Assemble_Matrix_CG_M->MAT->export_matrix();
    Output_Matrix(0, Sparse_ptr, mxCreateString(Block_Assemble_Matrix_CG_M->Name), plhs[0]);
    delete(Block_Assemble_Matrix_CG_M);

    Sparse_ptr = Block_Assemble_Matrix_Div_Matrix->MAT->export_matrix();
    Output_Matrix(1, Sparse_ptr, mxCreateString(Block_Assemble_Matrix_Div_Matrix->Name), plhs[0]);
    delete(Block_Assemble_Matrix_Div_Matrix);

    Sparse_ptr = Block_Assemble_Matrix_L2_div_sig_soln_sq->MAT->export_matrix();
    Output_Matrix(2, Sparse_ptr, mxCreateString(Block_Assemble_Matrix_L2_div_sig_soln_sq->Name), plhs[0]);
    delete(Block_Assemble_Matrix_L2_div_sig_soln_sq);

    Sparse_ptr = Block_Assemble_Matrix_L2_p_soln_sq->MAT->export_matrix();
    Output_Matrix(3, Sparse_ptr, mxCreateString(Block_Assemble_Matrix_L2_p_soln_sq->Name), plhs[0]);
    delete(Block_Assemble_Matrix_L2_p_soln_sq);

    Sparse_ptr = Block_Assemble_Matrix_L2_sig_soln_sq->MAT->export_matrix();
    Output_Matrix(4, Sparse_ptr, mxCreateString(Block_Assemble_Matrix_L2_sig_soln_sq->Name), plhs[0]);
    delete(Block_Assemble_Matrix_L2_sig_soln_sq);

    Sparse_ptr = Block_Assemble_Matrix_N_DOT_sigma->MAT->export_matrix();
    Output_Matrix(5, Sparse_ptr, mxCreateString(Block_Assemble_Matrix_N_DOT_sigma->Name), plhs[0]);
    delete(Block_Assemble_Matrix_N_DOT_sigma);

    Sparse_ptr = Block_Assemble_Matrix_P_Mass_Matrix->MAT->export_matrix();
    Output_Matrix(6, Sparse_ptr, mxCreateString(Block_Assemble_Matrix_P_Mass_Matrix->Name), plhs[0]);
    delete(Block_Assemble_Matrix_P_Mass_Matrix);

    Sparse_ptr = Block_Assemble_Matrix_Proj_P_MAT->MAT->export_matrix();
    Output_Matrix(7, Sparse_ptr, mxCreateString(Block_Assemble_Matrix_Proj_P_MAT->Name), plhs[0]);
    delete(Block_Assemble_Matrix_Proj_P_MAT);

    Sparse_ptr = Block_Assemble_Matrix_Proj_Sigma_MAT->MAT->export_matrix();
    Output_Matrix(8, Sparse_ptr, mxCreateString(Block_Assemble_Matrix_Proj_Sigma_MAT->Name), plhs[0]);
    delete(Block_Assemble_Matrix_Proj_Sigma_MAT);

    Sparse_ptr = Block_Assemble_Matrix_RHS_Matrix->MAT->export_matrix();
    Output_Matrix(9, Sparse_ptr, mxCreateString(Block_Assemble_Matrix_RHS_Matrix->Name), plhs[0]);
    delete(Block_Assemble_Matrix_RHS_Matrix);

    Sparse_ptr = Block_Assemble_Matrix_Sigma_Mass_Matrix->MAT->export_matrix();
    Output_Matrix(10, Sparse_ptr, mxCreateString(Block_Assemble_Matrix_Sigma_Mass_Matrix->Name), plhs[0]);
    delete(Block_Assemble_Matrix_Sigma_Mass_Matrix);

}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/


/***************************************************************************************/
/* setup sparse FEM matrices to be output to MATLAB */
#define NUM_Matrix_Fieldnames (sizeof(Matrix_Fieldnames)/sizeof(*Matrix_Fieldnames))
void GFA::Init_Output_Matrices (mxArray* plhs[])               // output
{
    // // declare internal matrix data storage pointers
    // mxArray *Sparse_ptr;

    // declare constant arrays (see 'Generic_FEM_Assembly.h')
    const char *Matrix_Fieldnames[] = {OUT_MAT_str, OUT_FEM_NAME_str};

    // declare parameters for outputing structures to MATLAB
    mwSize Matrix_dims[2] = {1, 1}; // just initialize to a 1x1 struct

    // set the number of MATLAB structs to create
    Matrix_dims[1] = NUM_FEM_MAT;
    /*** setup LHS argument of MATLAB calling function (i.e. the output structs) ***/
    //                            2, 1xN struct,    X sub-fields,  field names
    plhs[0] = mxCreateStructArray(2, Matrix_dims, NUM_Matrix_Fieldnames, Matrix_Fieldnames);
}
/***************************************************************************************/


/***************************************************************************************/
/* output sparse FEM matrix to MATLAB */
void GFA::Output_Matrix(mwIndex index, mxArray* Sparse_ptr, mxArray* Matrix_Name,   // input
                        mxArray* mxOUT) // output
{
	// point the output MATLAB structure fields to the correct data blocks!
	mxSetFieldByNumber(mxOUT,index,mxGetFieldNumber(mxOUT,OUT_MAT_str), Sparse_ptr);
	mxSetFieldByNumber(mxOUT,index,mxGetFieldNumber(mxOUT,OUT_FEM_NAME_str), Matrix_Name);
}
/***************************************************************************************/


/***************************************************************************************/
/* this verifies the incoming FEM matrix index and name matches, and accesses the data
   (if appropriate) */
void GFA::Access_Previous_FEM_Matrix(
                          const mxArray* OLD_FEM, const char* Matrix_Name,  // inputs
						  const int& Array_Index,                           // inputs
						  PTR_TO_SPARSE&  Data)                             // outputs
{
	if (!mxIsEmpty(OLD_FEM))
		{
		const int Num_Prev_Matrices = (const int) mxGetNumberOfElements(OLD_FEM);
		if (Array_Index >= Num_Prev_Matrices)
			mexErrMsgTxt("Index exceeds the number of incoming FEM matrices!");

        // determine which index of the subdomain array is the one we want
		const mxArray* mxMAT_Name = mxGetField(OLD_FEM, (mwIndex)Array_Index, OUT_FEM_NAME_str);

		/* Copy the string data over... */
        mwSize name_len = mxGetNumberOfElements(mxMAT_Name) + 1;
        char* name_in   = (char*) mxCalloc(name_len, sizeof(char));
        if (mxGetString(mxMAT_Name, name_in, name_len) != 0)
            mexErrMsgIdAndTxt("MATLAB:explore:invalidStringArray","Could not convert Matrix_Name string data.");

        // if they match, then access the data
        const bool name_equal = (strcmp(Matrix_Name,name_in)==0);
        mxFree(name_in);
        if (name_equal)
            Read_Sparse_Ptr(OLD_FEM, Array_Index, Data);
		else // fail
			{
			mexPrintf("ERROR: The Matrix_Name: %s\n",Matrix_Name);
			mexPrintf("ERROR:     does not match: %s\n",name_in);
			mexPrintf("ERROR:     at OLD_FEM index: %d\n",Array_Index+1); // put into MATLAB style
			mexErrMsgTxt("Check your OLD_FEM data!");
			}
        }
	else
		Clear_Sparse_Ptr(Data);
}
/***************************************************************************************/


/***************************************************************************************/
/* read sparse pointer data, so we can reuse sparse data structure in assembly */
void GFA::Read_Sparse_Ptr (const mxArray* OLD_FEM, const int& Array_Index,   // inputs
						   PTR_TO_SPARSE&  Data)                             // outputs
{
	// indicate that there is a matrix
	Data.valid = true;

	// store the matrix name as a string
	const mxArray* String_ptr = mxGetField(OLD_FEM, (mwIndex)Array_Index, OUT_FEM_NAME_str);
	unsigned int buflen = ((unsigned int)mxGetN(String_ptr))*sizeof(mxChar) + 1;
	Data.name = (char*) mxMalloc((size_t)buflen);
	// copy name over
	const int status = mxGetString(String_ptr, Data.name, (mwSize)buflen);
	if (status==1) mexErrMsgTxt("FEM matrix string name not read in correctly!");

	// get pointer to sparse MATLAB matrix
	const mxArray* Sparse_ptr = mxGetField(OLD_FEM, (mwIndex)Array_Index, OUT_MAT_str);
	// store CSC matrix format info
	Data.m  = (int) mxGetM(Sparse_ptr);
	Data.n  = (int) mxGetN(Sparse_ptr);
	Data.jc = (mwIndex*) mxGetJc(Sparse_ptr);
	Data.ir = (mwIndex*) mxGetIr(Sparse_ptr);
	Data.pr =  (double*) mxGetPr(Sparse_ptr);
}
/***************************************************************************************/


/***************************************************************************************/
/* clear sparse pointer data */
void GFA::Clear_Sparse_Ptr (PTR_TO_SPARSE&  Data)
{
	// clear pointers
	Data.valid = false;
	Data.name  = NULL;
	Data.m     = 0;
	Data.n     = 0;
	Data.jc    = NULL;
	Data.ir    = NULL;
	Data.pr    = NULL;
}
/***************************************************************************************/

#undef GFA

/***/
