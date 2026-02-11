/*
============================================================================================
   This file contains an implementation of a derived C++ Class from the abstract base class
   in 'Local_FE_Matrix.cc'.

   NOTE: portions of this code are automatically generated!

   Copyright (c) 06-14-2016,  Shawn W. Walker
============================================================================================
*/

/*------------ BEGIN: Auto Generate ------------*/
// FE matrix is:
//
// \int_{Gamma}  (sig_coef_t1_div + (2778046668940015*cos(geomGamma_X1*pi)*sin(geomGamma_X1*pi)*((43905379114243015*cos(geomGamma_X1*pi)^2*cos(geomGamma_X2*pi)^4)/17592186044416 - (70025138674774545*cos(geomGamma_X1*pi)^2*cos(geomGamma_X2*pi)^2)/70368744177664 + (1713638851887625*cos(geomGamma_X1*pi)^4*cos(geomGamma_X2*pi)^2)/8796093022208 - (1713638851887625*cos(geomGamma_X1*pi)^2*cos(geomGamma_X2*pi)^6)/2199023255552 - (1713638851887625*cos(geomGamma_X1*pi)^4*cos(geomGamma_X2*pi)^4)/2199023255552 + (6205324372715265*cos(geomGamma_X1*pi)^2)/35184372088832 - (2432922343284323*cos(geomGamma_X2*pi)^2)/140737488355328 + (1713638851887625*cos(geomGamma_X1*pi)^4)/8796093022208 - (78178156151995515*cos(geomGamma_X2*pi)^4)/70368744177664 + (5140916555662875*cos(geomGamma_X2*pi)^6)/8796093022208 + 3903946575782639/140737488355328))/(140737488355328*((2778046668940015*cos(geomGamma_X1*pi)^2)/281474976710656 - (2778046668940015*cos(geomGamma_X1*pi)^2*cos(geomGamma_X2*pi)^2)/140737488355328 + (2778046668940015*cos(geomGamma_X2*pi)^2)/281474976710656 + 1)^2))^2,  ... etc... (see individual submatrices, i.e. Tab_Tensor routines.)
//
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
// define the name of the FEM matrix (should be the same as the filename of this file)
#define SpecificFEM        Base_L2_div_sig_soln_sq_Data_Type
#define SpecificFEM_str   "L2_div_sig_soln_sq"

// the row function space is Type = constant_one, Name = "constant_one"

// set the number of cartesian tuple components (m*n) = 1 * 1
#define ROW_NC  1
// NOTE: the (i,j) tuple component is accessed by the linear index k = i + (j-1)*m
// set the number of basis functions on each element
#define ROW_NB  1

// the col function space is Type = constant_one, Name = "constant_one"

// set the number of cartesian tuple components (m*n) = 1 * 1
#define COL_NC  1
// NOTE: the (i,j) tuple component is accessed by the linear index k = i + (j-1)*m
// set the number of basis functions on each element
#define COL_NB  1
/*------------   END: Auto Generate ------------*/


/***************************************************************************************/
/* C++ (Specific) Local FE matrix class definition */
class SpecificFEM: public Base_FE_MATRIX_Class // derive from base class
{
public:

    // data structure for sub-matrices of the global FE matrix:
    // row/col offsets for inserting into global FE matrix
    int     Row_Shift[ROW_NC];
    int     Col_Shift[COL_NC];

    /*------------ BEGIN: Auto Generate ------------*/
    // access local mesh geometry info
    const CLASS_geom_Gamma_embedded_in_Gamma_restricted_to_Gamma*  geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;
    const CLASS_geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma*  geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    const CLASS_geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma*  geom_d_Gamma_embedded_in_Gamma_restricted_to_d_Gamma;
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // pointers for accessing FE basis functions
    const Data_Type_P_Space_phi_restricted_to_Gamma*  P_Space_phi_restricted_to_Gamma;
    const Data_Type_P_Space_phi_restricted_to_d_Gamma*  P_Space_phi_restricted_to_d_Gamma;
    const Data_Type_Sigma_Space_phi_restricted_to_Gamma*  Sigma_Space_phi_restricted_to_Gamma;
    const Data_Type_Sigma_Space_phi_restricted_to_d_Gamma*  Sigma_Space_phi_restricted_to_d_Gamma;
    const Data_Type_V_Space_phi_restricted_to_Gamma*  V_Space_phi_restricted_to_Gamma;
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // pointers for accessing coefficient functions
    const Data_Type_p_coef_restricted_to_Gamma*  p_coef_restricted_to_Gamma;
    const Data_Type_sig_coef_restricted_to_Gamma*  sig_coef_restricted_to_Gamma;
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // data tables to hold local FE tensors (matrices)
    double  FE_Tensor_0[ROW_NB*COL_NB];
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // constructor
    SpecificFEM (const ABSTRACT_FEM_Function_Class*, const ABSTRACT_FEM_Function_Class*);
    ~SpecificFEM (); // destructor
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    void Tabulate_Tensor(const CLASS_geom_Gamma_embedded_in_Gamma_restricted_to_Gamma&);
    /*------------   END: Auto Generate ------------*/
private:
};


/***************************************************************************************/
/* constructor */
SpecificFEM::SpecificFEM (const ABSTRACT_FEM_Function_Class* row_basis_func,
                          const ABSTRACT_FEM_Function_Class* col_basis_func) :
Base_FE_MATRIX_Class () // call the base class constructor
{
    // set the 'Name' of the form for this local FE matrix
    Form_Name = (char*) SpecificFEM_str; // this should be similar to the Class identifier
    // record the type of form
    Form_Type = real;
    // record the number of sub-matrices (in local FE matrix)
    Num_Sub_Matrices = 1;

    // BEGIN: simple error check
    if (row_basis_func->Num_Comp!=ROW_NC)
        {
        mexPrintf("ERROR: The number of components for the row FE space is NOT correct!\n");
        mexPrintf("The FE matrix '%s' expects %d row components.\n",SpecificFEM_str,ROW_NC);
        mexPrintf("Actual number of row components is %d.\n",row_basis_func->Num_Comp);
        mexErrMsgTxt("Please report this error!\n");
        }
    if (col_basis_func->Num_Comp!=COL_NC)
        {
        mexPrintf("ERROR: The number of components for the column FE space is NOT correct!\n");
        mexPrintf("The FE matrix '%s' expects %d col components.\n",SpecificFEM_str,ROW_NC);
        mexPrintf("Actual number of col components is %d.\n",col_basis_func->Num_Comp);
        mexErrMsgTxt("Please report this error!\n");
        }
    if (row_basis_func->Num_Basis!=ROW_NB)
        {
        mexPrintf("ERROR: The number of basis functions in the row FEM space is NOT correct!\n");
        mexPrintf("The FEM matrix '%s' expects %d row basis functions.\n",SpecificFEM_str,ROW_NB);
        mexPrintf("Actual number of row components is %d.\n",row_basis_func->Num_Basis);
        mexErrMsgTxt("Please report this error!\n");
        }
    if (col_basis_func->Num_Basis!=COL_NB)
        {
        mexPrintf("ERROR: The number of basis functions in the column FEM space is NOT correct!\n");
        mexPrintf("The FEM matrix '%s' expects %d col basis functions.\n",SpecificFEM_str,COL_NB);
        mexPrintf("Actual number of row components is %d.\n",col_basis_func->Num_Basis);
        mexErrMsgTxt("Please report this error!\n");
        }
    //   END: simple error check

    // record the size of the global matrix
    global_num_row = row_basis_func->Num_Comp*row_basis_func->Num_Nodes;
    global_num_col = col_basis_func->Num_Comp*col_basis_func->Num_Nodes;

    /*------------ BEGIN: Auto Generate ------------*/
    // input information for offsetting the sub-matrices
    Row_Shift[0] = 0*row_basis_func->Num_Nodes;
    Col_Shift[0] = 0*col_basis_func->Num_Nodes;
    /*------------   END: Auto Generate ------------*/
}
/***************************************************************************************/


/***************************************************************************************/
/* destructor: should not usually need to be modified */
SpecificFEM::~SpecificFEM ()
{
}
/***************************************************************************************/

/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* Tabulate the tensor for the local element contribution */
void SpecificFEM::Tabulate_Tensor(const CLASS_geom_Gamma_embedded_in_Gamma_restricted_to_Gamma& Mesh)
{
    const unsigned int NQ = 12;

    // Compute single entry using quadrature

    // Loop quadrature points for integral
    double  A0_value = 0.0; // initialize
    for (unsigned int qp = 0; qp < NQ; qp++)
        {
        const double  t2 = geom_Gamma_embedded_in_Gamma_restricted_to_Gamma->Map_PHI[qp].v[0]*3.141592653589793;
        const double  t3 = cos(t2);
        const double  t6 = geom_Gamma_embedded_in_Gamma_restricted_to_Gamma->Map_PHI[qp].v[1]*3.141592653589793;
        const double  t4 = cos(t6);
        const double  t5 = t3*t3;
        const double  t7 = t4*t4;
        const double  t8 = t7*t7;
        const double  t9 = t5*t5;
        const double  t10 = sig_coef_restricted_to_Gamma->Func_vv_Div[0][qp].a+t3*sin(t2)*1.0/pow(t5*9.869604401089358+t7*9.869604401089358-t5*t7*1.973920880217872E1+1.0,2.0)*(t5*1.763659262427173E2-t7*1.728695297689116E1-t8*1.110978418978384E3+t9*1.948181820680048E2-t5*t7*9.95117072119094E2+t5*t8*2.495731855233488E3+t7*t8*5.844545462040145E2+t7*t9*1.948181820680048E2-t8*t9*7.792727282720193E2-t5*t7*t8*7.792727282720193E2+2.773920880217872E1)*1.973920880217872E1;
        const double  integrand_0 = t10*t10;
        A0_value += integrand_0 * Mesh.Map_Det_Jac_w_Weight[qp].a;
        }
    FE_Tensor_0[0] = A0_value;
}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/

// remove those macros!
#undef SpecificFEM
#undef SpecificFEM_str

#undef ROW_NC
#undef ROW_NB
#undef COL_NC
#undef COL_NB

/***/

