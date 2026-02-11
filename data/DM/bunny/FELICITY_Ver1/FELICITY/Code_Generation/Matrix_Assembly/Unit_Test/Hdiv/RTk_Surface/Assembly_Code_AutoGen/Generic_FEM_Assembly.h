/*
============================================================================================
   Header file for a C++ Class that contains methods for generic finite element assembly.

   NOTE: portions of this code are automatically generated!

   Copyright (c) 06-20-2012,  Shawn W. Walker
============================================================================================
*/

/*------------ BEGIN: Auto Generate ------------*/
// set the number of FEM matrices to assemble
#define NUM_FEM_MAT    11

// In MATLAB, the output FEM matrix entry list should look like:
//            FEM.MAT
//               .Type
//
// Here, we define the strings that makes these variable names
#define OUT_MAT_str      "MAT"
#define OUT_FEM_NAME_str "Type"
/*------------   END: Auto Generate ------------*/

/***************************************************************************************/
/*** C++ class ***/
class Generic_FEM_Assembly
{
public:
    //Generic_FEM_Assembly (); // constructor
    Generic_FEM_Assembly (const mxArray *[], const mxArray *); // constructor
    ~Generic_FEM_Assembly (); // DE-structor

    /*------------ BEGIN: Auto Generate ------------*/
    // create access routines
    const Data_Type_p_coef_restricted_to_Gamma* Get_p_coef_restricted_to_Gamma_ptr() const { return &(p_coef_restricted_to_Gamma); }
    const Data_Type_sig_coef_restricted_to_Gamma* Get_sig_coef_restricted_to_Gamma_ptr() const { return &(sig_coef_restricted_to_Gamma); }

    void Setup_Data (const mxArray*[]);
    void Assemble_Matrices ();
    void Output_Matrices (mxArray*[]);
    void Init_Output_Matrices (mxArray*[]);
    void Output_Matrix (mwIndex, mxArray*, mxArray*, mxArray*);
    void Access_Previous_FEM_Matrix (const mxArray*, const char*, const int&, PTR_TO_SPARSE&);
    void Read_Sparse_Ptr (const mxArray*, const int&, PTR_TO_SPARSE&);
    void Clear_Sparse_Ptr (PTR_TO_SPARSE&);

private:
    // these variables are defined from inputs coming from MATLAB

    // classes for (sub)domain(s) and topological entities
    CLASS_Domain_Gamma_embedded_in_Gamma_restricted_to_Gamma    Domain_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    CLASS_Domain_Gamma_embedded_in_Gamma_restricted_to_d_Gamma    Domain_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    CLASS_Domain_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma    Domain_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;

    // mesh geometry classes (can be higher order)
    CLASS_geom_Gamma_embedded_in_Gamma_restricted_to_Gamma   geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    CLASS_geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma   geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    CLASS_geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma   geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // pointers to each specific FEM matrix
    Base_CG_M_Data_Type*    Base_Matrix_CG_M;
    Base_Div_Matrix_Data_Type*    Base_Matrix_Div_Matrix;
    Base_L2_div_sig_soln_sq_Data_Type*    Base_Matrix_L2_div_sig_soln_sq;
    Base_L2_p_soln_sq_Data_Type*    Base_Matrix_L2_p_soln_sq;
    Base_L2_sig_soln_sq_Data_Type*    Base_Matrix_L2_sig_soln_sq;
    Base_N_DOT_sigma_Data_Type*    Base_Matrix_N_DOT_sigma;
    Base_P_Mass_Matrix_Data_Type*    Base_Matrix_P_Mass_Matrix;
    Base_Proj_P_MAT_Data_Type*    Base_Matrix_Proj_P_MAT;
    Base_Proj_Sigma_MAT_Data_Type*    Base_Matrix_Proj_Sigma_MAT;
    Base_RHS_Matrix_Data_Type*    Base_Matrix_RHS_Matrix;
    Base_Sigma_Mass_Matrix_Data_Type*    Base_Matrix_Sigma_Mass_Matrix;

    Block_Assemble_CG_M_Data_Type*    Block_Assemble_Matrix_CG_M;
    PTR_TO_SPARSE    Sparse_Data_CG_M;
    Block_Assemble_Div_Matrix_Data_Type*    Block_Assemble_Matrix_Div_Matrix;
    PTR_TO_SPARSE    Sparse_Data_Div_Matrix;
    Block_Assemble_L2_div_sig_soln_sq_Data_Type*    Block_Assemble_Matrix_L2_div_sig_soln_sq;
    PTR_TO_SPARSE    Sparse_Data_L2_div_sig_soln_sq;
    Block_Assemble_L2_p_soln_sq_Data_Type*    Block_Assemble_Matrix_L2_p_soln_sq;
    PTR_TO_SPARSE    Sparse_Data_L2_p_soln_sq;
    Block_Assemble_L2_sig_soln_sq_Data_Type*    Block_Assemble_Matrix_L2_sig_soln_sq;
    PTR_TO_SPARSE    Sparse_Data_L2_sig_soln_sq;
    Block_Assemble_N_DOT_sigma_Data_Type*    Block_Assemble_Matrix_N_DOT_sigma;
    PTR_TO_SPARSE    Sparse_Data_N_DOT_sigma;
    Block_Assemble_P_Mass_Matrix_Data_Type*    Block_Assemble_Matrix_P_Mass_Matrix;
    PTR_TO_SPARSE    Sparse_Data_P_Mass_Matrix;
    Block_Assemble_Proj_P_MAT_Data_Type*    Block_Assemble_Matrix_Proj_P_MAT;
    PTR_TO_SPARSE    Sparse_Data_Proj_P_MAT;
    Block_Assemble_Proj_Sigma_MAT_Data_Type*    Block_Assemble_Matrix_Proj_Sigma_MAT;
    PTR_TO_SPARSE    Sparse_Data_Proj_Sigma_MAT;
    Block_Assemble_RHS_Matrix_Data_Type*    Block_Assemble_Matrix_RHS_Matrix;
    PTR_TO_SPARSE    Sparse_Data_RHS_Matrix;
    Block_Assemble_Sigma_Mass_Matrix_Data_Type*    Block_Assemble_Matrix_Sigma_Mass_Matrix;
    PTR_TO_SPARSE    Sparse_Data_Sigma_Mass_Matrix;
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // create variables for accessing and evaulating FEM (local) basis functions
    Data_Type_P_Space_phi_restricted_to_Gamma      P_Space_phi_restricted_to_Gamma;
    Data_Type_Sigma_Space_phi_restricted_to_Gamma      Sigma_Space_phi_restricted_to_Gamma;
    Data_Type_V_Space_phi_restricted_to_Gamma      V_Space_phi_restricted_to_Gamma;
    Data_Type_P_Space_phi_restricted_to_d_Gamma      P_Space_phi_restricted_to_d_Gamma;
    Data_Type_Sigma_Space_phi_restricted_to_d_Gamma      Sigma_Space_phi_restricted_to_d_Gamma;
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // create variables for accessing CONSTANT basis functions
    Data_Type_CONST_ONE_phi      L2_div_sig_soln_sq_Gamma_col_constant_phi;
    Data_Type_CONST_ONE_phi      L2_div_sig_soln_sq_Gamma_row_constant_phi;
    Data_Type_CONST_ONE_phi      L2_p_soln_sq_Gamma_col_constant_phi;
    Data_Type_CONST_ONE_phi      L2_p_soln_sq_Gamma_row_constant_phi;
    Data_Type_CONST_ONE_phi      L2_sig_soln_sq_Gamma_col_constant_phi;
    Data_Type_CONST_ONE_phi      L2_sig_soln_sq_Gamma_row_constant_phi;
    Data_Type_CONST_ONE_phi      N_DOT_sigma_Gamma_col_constant_phi;
    Data_Type_CONST_ONE_phi      N_DOT_sigma_Gamma_row_constant_phi;
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // create variables for accessing external functions
    Data_Type_p_coef_restricted_to_Gamma      p_coef_restricted_to_Gamma;
    Data_Type_sig_coef_restricted_to_Gamma      sig_coef_restricted_to_Gamma;
    /*------------   END: Auto Generate ------------*/

};

/***/
