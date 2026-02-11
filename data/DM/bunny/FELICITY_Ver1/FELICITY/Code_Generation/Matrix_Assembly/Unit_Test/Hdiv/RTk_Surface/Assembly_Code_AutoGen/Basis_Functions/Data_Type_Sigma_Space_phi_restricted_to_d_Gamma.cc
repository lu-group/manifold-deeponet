/*
============================================================================================
   This class contains data about a given FE function space, and methods for computing
   transformations of the local basis functions.

   This code references the header files:

   matrix_vector_defn.h
   matrix_vector_ops.h
   geometric_computations.h
   basis_function_computations.h


   NOTE: portions of this code are automatically generated!

   Copyright (c) 01-15-2018,  Shawn W. Walker
============================================================================================
*/

/*------------ BEGIN: Auto Generate ------------*/
// define the name of the FE basis function (should be the same as the filename of this file)
#define SpecificFUNC        Data_Type_Sigma_Space_phi_restricted_to_d_Gamma
#define SpecificFUNC_str   "Data_Type_Sigma_Space_phi_restricted_to_d_Gamma"

// set the type of function space
#define SPACE_type  "CG - raviart_thomas_deg1_dim2"
// set the name of function space
#define SPACE_name  "Sigma_Space"

// set the Subdomain topological dimension
#define SUB_TD  2
// set the Domain of Integration (DoI) topological dimension
#define DOI_TD  1
// set the geometric dimension
#define GD  3
// set the number of cartesian tuple components (m*n) = 1 * 1
#define NC  1
// NOTE: the (i,j) tuple component is accessed by the linear index k = i + (j-1)*m
// set the number of quad points
#define NQ  5
// set the number of basis functions
#define NB  8
/*------------   END: Auto Generate ------------*/

/* C++ (Specific) FE Function class definition */
class SpecificFUNC: public ABSTRACT_FEM_Function_Class // derive from base class
{
public:
    int*     Elem_DoF[NB];    // element DoF list

    // data structure containing information on the function evaluations.
    // Note: this data is evaluated at several quadrature points!
    // vector valued H(div) basis functions
    VEC_3x1 Func_vv_Value[NB][NQ];

    // constructor
    SpecificFUNC ();
    ~SpecificFUNC (); // destructor
    void Setup_Function_Space(const mxArray*);
    void Get_Local_to_Global_DoFmap(const int&, int*) const;
                   // need the "const" to ENSURE that nothing in this object will change!
    void Transform_Basis_Functions();
    const CLASS_geom_Gamma_embedded_in_Gamma_restricted_to_d_Gamma*  Mesh;

private:
    void Get_Basis_Sign_Change(const double Mesh_Orient[SUB_TD+1], double Basis_Sign[NB]) const;
    void Map_Basis_p1();
    void Map_Basis_n1();
    void Map_Basis_p2();
    void Map_Basis_n2();
    void Map_Basis_p3();
    void Map_Basis_n3();
};

/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* constructor */
SpecificFUNC::SpecificFUNC () :
ABSTRACT_FEM_Function_Class () // call the base class constructor
{
    Name       = (char*) SpecificFUNC_str;
    Type       = (char*) SPACE_type;
    Space_Name = (char*) SPACE_name;
    Sub_TopDim = SUB_TD;
    DoI_TopDim = DOI_TD;
    GeoDim     = GD;
    Num_Basis  = NB;
    Num_Comp   = NC;
    Num_QP     = NQ;
    Mesh       = NULL;

    // init DoF information to NULL
    for (int basis_i = 0; (basis_i < Num_Basis); basis_i++)
        Elem_DoF[basis_i] = NULL;

    // init everything to zero
    for (int qp_i = 0; (qp_i < Num_QP); qp_i++)
    for (int basis_i = 0; (basis_i < Num_Basis); basis_i++)
        Func_vv_Value[basis_i][qp_i].Set_To_Zero();
}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/


/***************************************************************************************/
/* DE-structor */
SpecificFUNC::~SpecificFUNC ()
{
}
/***************************************************************************************/


/***************************************************************************************/
/* put incoming function data from MATLAB into a nice struct  */
void SpecificFUNC::Setup_Function_Space(const mxArray* Elem)          // inputs
{
    Init_Function_Space(Elem);

    // split up the columns of the element data
    Elem_DoF[0] = (int *) mxGetPr(Elem);
    for (int basis_i = 1; (basis_i < Num_Basis); basis_i++)
        Elem_DoF[basis_i] = Elem_DoF[basis_i-1] + Num_Elem;
}
/***************************************************************************************/


/***************************************************************************************/
/* get the local DoFs on the given element.
   Note: elem_index is in the   C-style (i.e. 0 <= elem_index <= Num_Elem - 1),
         Indices is in the MATLAB-style (i.e. 1 <= Indices[:] <= max(Elem_DoF)). */
void SpecificFUNC::Get_Local_to_Global_DoFmap(const int& elem_index, int* Indices) const  // inputs
{
    /* error check: */
    if (elem_index < 0)
        {
        mexPrintf("ERROR: Given element index #%d is not positive. It must be > 0!\n",elem_index+1);
        mexPrintf("ERROR: There is an issue with the Finite Element Space = %s!\n",Space_Name);
        mexErrMsgTxt("ERROR: Make sure your inputs are valid!");
        }
    else if (elem_index >= Num_Elem)
        {
        mexPrintf("ERROR: Given element index #%d exceeds the number of elements in the finite element (FE) space.\n",elem_index+1);
		mexPrintf("It must be <= %d!  OR  Your FE space DoFmap is not defined correctly!\n",Num_Elem);
		mexPrintf("   For example, the number of rows in DoFmap should *equal*\n");
		mexPrintf("       the number of mesh elements in the (sub)-domain.\n");
        mexPrintf("ERROR: There is an issue with the Finite Element Space = %s!\n",Space_Name);
        mexErrMsgTxt("ERROR: Make sure your inputs are valid!");
        }

    // get local to global index map for the current element
    for (int basis_i = 0; (basis_i < Num_Basis); basis_i++)
        {
        int DoF_index = Elem_DoF[basis_i][elem_index] - 1; // shifted for C - style indexing
        Indices[basis_i] = DoF_index;
        }
}
/***************************************************************************************/

/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* get sign changes to apply to basis functions to account for changes in orientation
  (with respect to the "standard" reference orientation) on the current mesh element.  */
void SpecificFUNC::Get_Basis_Sign_Change(const double Mesh_Orient[SUB_TD+1], double Basis_Sign[NB]) const
{
    // get facet (edge) orientation "signature" (takes values from 0 to 7)
    const int Facet_Orientation_Signature = (int) ( (Mesh_Orient[0] < 0.0) * 1 +
                                                    (Mesh_Orient[1] < 0.0) * 2 +
                                                    (Mesh_Orient[2] < 0.0) * 4 );

    // BEGIN: determine which basis functions must change their signs
    /* 2-D triangle element has 3 facets */
    switch (Facet_Orientation_Signature)
    {
    case 1: // (-1, 1, 1) facet orientation
        Basis_Sign[0] = -1.0;
        Basis_Sign[1] = -1.0;
        break;
    case 2: // (1, -1, 1) facet orientation
        Basis_Sign[2] = -1.0;
        Basis_Sign[3] = -1.0;
        break;
    case 3: // (-1, -1, 1) facet orientation
        Basis_Sign[0] = -1.0;
        Basis_Sign[1] = -1.0;
        Basis_Sign[2] = -1.0;
        Basis_Sign[3] = -1.0;
        break;
    case 4: // (1, 1, -1) facet orientation
        Basis_Sign[4] = -1.0;
        Basis_Sign[5] = -1.0;
        break;
    case 5: // (-1, 1, -1) facet orientation
        Basis_Sign[0] = -1.0;
        Basis_Sign[1] = -1.0;
        Basis_Sign[4] = -1.0;
        Basis_Sign[5] = -1.0;
        break;
    case 6: // (1, -1, -1) facet orientation
        Basis_Sign[2] = -1.0;
        Basis_Sign[3] = -1.0;
        Basis_Sign[4] = -1.0;
        Basis_Sign[5] = -1.0;
        break;
    case 7: // (-1, -1, -1) facet orientation
        Basis_Sign[0] = -1.0;
        Basis_Sign[1] = -1.0;
        Basis_Sign[2] = -1.0;
        Basis_Sign[3] = -1.0;
        Basis_Sign[4] = -1.0;
        Basis_Sign[5] = -1.0;
        break;
    default: ; // Facet_Orientation_Signature==0 (1, 1, 1) facet orientation
        // do nothing; everything is positive already
    }
    // END: determine which basis functions must change their signs
}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* compute the correct local transformation */
void SpecificFUNC::Transform_Basis_Functions()
{
    // read in the embedding info
    const int Entity_Index = Mesh->Domain->DoI_Entity_Index;

    // if orientation is positive
    if (Entity_Index > 0)
        {
        // pick the correct one
        if (Entity_Index==1) Map_Basis_p1();
        else if (Entity_Index==2) Map_Basis_p2();
        else if (Entity_Index==3) Map_Basis_p3();
        else mexErrMsgTxt("ERROR: Entity_Index outside valid range or is zero!");
        }
    else // else it is negative
        {
        // pick the correct one
        if (Entity_Index==-1) Map_Basis_n1();
        else if (Entity_Index==-2) Map_Basis_n2();
        else if (Entity_Index==-3) Map_Basis_n3();
        else mexErrMsgTxt("ERROR: Entity_Index outside valid range or is zero!");
        }
}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/


/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* map basis functions from the standard reference element
       to an actual element in the Domain.     */
void SpecificFUNC::Map_Basis_p1()
{

/*------------ BEGIN: Auto Generate ------------*/
    /* call sub-routine to determine sign flips for H(div) basis functions */
    double Basis_Sign[NB] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // init to all positive (no sign change)
    Get_Basis_Sign_Change(Mesh->Orientation, Basis_Sign);
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
// Local Element defined on Subdomain: CG, raviart_thomas_deg1_dim2
// the Subdomain             has topological dimension = 2
// the Domain of Integration has topological dimension = 1
// geometric dimension = 3
// Number of Quadrature Points = 5

    // get "Val" of basis functions
    VEC_2x1 phi_Val[NQ][NB];

    phi_Val[0][0].Set_Equal_To(3.45468351824816899E+00, 2.63856019567822730E-01);
    phi_Val[0][1].Set_Equal_To(-1.54850367230950492E+00, -1.70035865506486583E-01);
    phi_Val[0][2].Set_Equal_To(1.70035865506486583E-01, -1.70035865506486583E-01);
    phi_Val[0][3].Set_Equal_To(9.38201540613361473E-02, -9.38201540613361473E-02);
    phi_Val[0][4].Set_Equal_To(-1.90617984593866385E+00, 1.90617984593866385E+00);
    phi_Val[0][5].Set_Equal_To(3.45468351824816899E+00, -3.45468351824816899E+00);
    phi_Val[0][6].Set_Equal_To(3.57676173629158878E-01, -3.57676173629158878E-01);
    phi_Val[0][7].Set_Equal_To(-3.57676173629158878E-01, 3.57676173629158878E-01);
    phi_Val[1][0].Set_Equal_To(1.65683701606274680E+00, 9.58570914254302520E-01);
    phi_Val[1][1].Set_Equal_To(-1.18367705957063751E-01, -4.97040224359985572E-01);
    phi_Val[1][2].Set_Equal_To(4.97040224359985572E-01, -4.97040224359985572E-01);
    phi_Val[1][3].Set_Equal_To(4.61530689894316892E-01, -4.61530689894316892E-01);
    phi_Val[1][4].Set_Equal_To(-1.53846931010568322E+00, 1.53846931010568322E+00);
    phi_Val[1][5].Set_Equal_To(1.65683701606274680E+00, -1.65683701606274680E+00);
    phi_Val[1][6].Set_Equal_To(1.42010160414861941E+00, -1.42010160414861941E+00);
    phi_Val[1][7].Set_Equal_To(-1.42010160414861941E+00, 1.42010160414861941E+00);
    phi_Val[2][0].Set_Equal_To(0.00000000000000000E+00, 1.00000000000000000E+00);
    phi_Val[2][1].Set_Equal_To(1.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][2].Set_Equal_To(0.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][3].Set_Equal_To(1.00000000000000000E+00, -1.00000000000000000E+00);
    phi_Val[2][4].Set_Equal_To(-1.00000000000000000E+00, 1.00000000000000000E+00);
    phi_Val[2][5].Set_Equal_To(0.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][6].Set_Equal_To(2.00000000000000000E+00, -2.00000000000000000E+00);
    phi_Val[2][7].Set_Equal_To(-2.00000000000000000E+00, 2.00000000000000000E+00);
    phi_Val[3][0].Set_Equal_To(-4.97040224359985572E-01, -1.18367705957063404E-01);
    phi_Val[3][1].Set_Equal_To(9.58570914254302631E-01, 1.65683701606274636E+00);
    phi_Val[3][2].Set_Equal_To(-1.65683701606274636E+00, 1.65683701606274636E+00);
    phi_Val[3][3].Set_Equal_To(1.53846931010568300E+00, -1.53846931010568300E+00);
    phi_Val[3][4].Set_Equal_To(-4.61530689894317003E-01, 4.61530689894317003E-01);
    phi_Val[3][5].Set_Equal_To(-4.97040224359985572E-01, 4.97040224359985572E-01);
    phi_Val[3][6].Set_Equal_To(1.42010160414861963E+00, -1.42010160414861963E+00);
    phi_Val[3][7].Set_Equal_To(-1.42010160414861963E+00, 1.42010160414861963E+00);
    phi_Val[4][0].Set_Equal_To(-1.70035865506486583E-01, -1.54850367230950492E+00);
    phi_Val[4][1].Set_Equal_To(2.63856019567822730E-01, 3.45468351824816899E+00);
    phi_Val[4][2].Set_Equal_To(-3.45468351824816899E+00, 3.45468351824816899E+00);
    phi_Val[4][3].Set_Equal_To(1.90617984593866385E+00, -1.90617984593866385E+00);
    phi_Val[4][4].Set_Equal_To(-9.38201540613361473E-02, 9.38201540613361473E-02);
    phi_Val[4][5].Set_Equal_To(-1.70035865506486583E-01, 1.70035865506486583E-01);
    phi_Val[4][6].Set_Equal_To(3.57676173629158878E-01, -3.57676173629158878E-01);
    phi_Val[4][7].Set_Equal_To(-3.57676173629158878E-01, 3.57676173629158878E-01);


// // set of quadrature points
// static const double Quad_Points[NQ][SUB_TD] = { \
//     4.69100770306680737E-02, \
//     2.30765344947158446E-01, \
//     5.00000000000000000E-01, \
//     7.69234655052841498E-01, \
//     9.53089922969331926E-01  \
//     };

// // set of quadrature weights
// static const double Quad_Weights[NQ] = { \
//     1.18463442528094529E-01, \
//     2.39314335249683374E-01, \
//     2.84444444444444333E-01, \
//     2.39314335249683430E-01, \
//     1.18463442528094445E-01  \
//     };
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
    /*** compute basis function quantities ***/
    // map basis vectors over (indexing is in the C style)
    // loop through quad points
    for (int qp_i = 0; (qp_i < Num_QP); qp_i++)
        {
        // evaluate for each basis function
        for (int basis_i = 0; (basis_i < Num_Basis); basis_i++)
            {
            VEC_2x1 vv_temp;
            // pre-multiply by sign change
            const double INV_DET_SIGN_FLIP = Basis_Sign[basis_i] * Mesh->Map_Inv_Det_Jac[qp_i].a;
            // pre-multiply by 1/det(Jac)
            Scalar_Mult_Vector(phi_Val[qp_i][basis_i], INV_DET_SIGN_FLIP, vv_temp);
            Mat_Vec(Mesh->Map_PHI_Grad[qp_i], vv_temp, Func_vv_Value[basis_i][qp_i]);
            }
        }
/*------------   END: Auto Generate ------------*/
}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* map basis functions from the standard reference element
       to an actual element in the Domain.     */
void SpecificFUNC::Map_Basis_n1()
{

/*------------ BEGIN: Auto Generate ------------*/
    /* call sub-routine to determine sign flips for H(div) basis functions */
    double Basis_Sign[NB] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // init to all positive (no sign change)
    Get_Basis_Sign_Change(Mesh->Orientation, Basis_Sign);
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
// Local Element defined on Subdomain: CG, raviart_thomas_deg1_dim2
// the Subdomain             has topological dimension = 2
// the Domain of Integration has topological dimension = 1
// geometric dimension = 3
// Number of Quadrature Points = 5

    // get "Val" of basis functions
    VEC_2x1 phi_Val[NQ][NB];

    phi_Val[0][0].Set_Equal_To(-1.70035865506486583E-01, -1.54850367230950492E+00);
    phi_Val[0][1].Set_Equal_To(2.63856019567822730E-01, 3.45468351824816899E+00);
    phi_Val[0][2].Set_Equal_To(-3.45468351824816899E+00, 3.45468351824816899E+00);
    phi_Val[0][3].Set_Equal_To(1.90617984593866385E+00, -1.90617984593866385E+00);
    phi_Val[0][4].Set_Equal_To(-9.38201540613361473E-02, 9.38201540613361473E-02);
    phi_Val[0][5].Set_Equal_To(-1.70035865506486583E-01, 1.70035865506486583E-01);
    phi_Val[0][6].Set_Equal_To(3.57676173629158878E-01, -3.57676173629158878E-01);
    phi_Val[0][7].Set_Equal_To(-3.57676173629158878E-01, 3.57676173629158878E-01);
    phi_Val[1][0].Set_Equal_To(-4.97040224359985572E-01, -1.18367705957063751E-01);
    phi_Val[1][1].Set_Equal_To(9.58570914254302520E-01, 1.65683701606274680E+00);
    phi_Val[1][2].Set_Equal_To(-1.65683701606274680E+00, 1.65683701606274680E+00);
    phi_Val[1][3].Set_Equal_To(1.53846931010568322E+00, -1.53846931010568322E+00);
    phi_Val[1][4].Set_Equal_To(-4.61530689894316892E-01, 4.61530689894316892E-01);
    phi_Val[1][5].Set_Equal_To(-4.97040224359985572E-01, 4.97040224359985572E-01);
    phi_Val[1][6].Set_Equal_To(1.42010160414861941E+00, -1.42010160414861941E+00);
    phi_Val[1][7].Set_Equal_To(-1.42010160414861941E+00, 1.42010160414861941E+00);
    phi_Val[2][0].Set_Equal_To(0.00000000000000000E+00, 1.00000000000000000E+00);
    phi_Val[2][1].Set_Equal_To(1.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][2].Set_Equal_To(0.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][3].Set_Equal_To(1.00000000000000000E+00, -1.00000000000000000E+00);
    phi_Val[2][4].Set_Equal_To(-1.00000000000000000E+00, 1.00000000000000000E+00);
    phi_Val[2][5].Set_Equal_To(0.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][6].Set_Equal_To(2.00000000000000000E+00, -2.00000000000000000E+00);
    phi_Val[2][7].Set_Equal_To(-2.00000000000000000E+00, 2.00000000000000000E+00);
    phi_Val[3][0].Set_Equal_To(1.65683701606274636E+00, 9.58570914254302631E-01);
    phi_Val[3][1].Set_Equal_To(-1.18367705957063404E-01, -4.97040224359985572E-01);
    phi_Val[3][2].Set_Equal_To(4.97040224359985572E-01, -4.97040224359985572E-01);
    phi_Val[3][3].Set_Equal_To(4.61530689894317003E-01, -4.61530689894317003E-01);
    phi_Val[3][4].Set_Equal_To(-1.53846931010568300E+00, 1.53846931010568300E+00);
    phi_Val[3][5].Set_Equal_To(1.65683701606274636E+00, -1.65683701606274636E+00);
    phi_Val[3][6].Set_Equal_To(1.42010160414861963E+00, -1.42010160414861963E+00);
    phi_Val[3][7].Set_Equal_To(-1.42010160414861963E+00, 1.42010160414861963E+00);
    phi_Val[4][0].Set_Equal_To(3.45468351824816899E+00, 2.63856019567822730E-01);
    phi_Val[4][1].Set_Equal_To(-1.54850367230950492E+00, -1.70035865506486583E-01);
    phi_Val[4][2].Set_Equal_To(1.70035865506486583E-01, -1.70035865506486583E-01);
    phi_Val[4][3].Set_Equal_To(9.38201540613361473E-02, -9.38201540613361473E-02);
    phi_Val[4][4].Set_Equal_To(-1.90617984593866385E+00, 1.90617984593866385E+00);
    phi_Val[4][5].Set_Equal_To(3.45468351824816899E+00, -3.45468351824816899E+00);
    phi_Val[4][6].Set_Equal_To(3.57676173629158878E-01, -3.57676173629158878E-01);
    phi_Val[4][7].Set_Equal_To(-3.57676173629158878E-01, 3.57676173629158878E-01);


// // set of quadrature points
// static const double Quad_Points[NQ][SUB_TD] = { \
//     4.69100770306680737E-02, \
//     2.30765344947158446E-01, \
//     5.00000000000000000E-01, \
//     7.69234655052841498E-01, \
//     9.53089922969331926E-01  \
//     };

// // set of quadrature weights
// static const double Quad_Weights[NQ] = { \
//     1.18463442528094529E-01, \
//     2.39314335249683374E-01, \
//     2.84444444444444333E-01, \
//     2.39314335249683430E-01, \
//     1.18463442528094445E-01  \
//     };
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
    /*** compute basis function quantities ***/
    // map basis vectors over (indexing is in the C style)
    // loop through quad points
    for (int qp_i = 0; (qp_i < Num_QP); qp_i++)
        {
        // evaluate for each basis function
        for (int basis_i = 0; (basis_i < Num_Basis); basis_i++)
            {
            VEC_2x1 vv_temp;
            // pre-multiply by sign change
            const double INV_DET_SIGN_FLIP = Basis_Sign[basis_i] * Mesh->Map_Inv_Det_Jac[qp_i].a;
            // pre-multiply by 1/det(Jac)
            Scalar_Mult_Vector(phi_Val[qp_i][basis_i], INV_DET_SIGN_FLIP, vv_temp);
            Mat_Vec(Mesh->Map_PHI_Grad[qp_i], vv_temp, Func_vv_Value[basis_i][qp_i]);
            }
        }
/*------------   END: Auto Generate ------------*/
}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* map basis functions from the standard reference element
       to an actual element in the Domain.     */
void SpecificFUNC::Map_Basis_p2()
{

/*------------ BEGIN: Auto Generate ------------*/
    /* call sub-routine to determine sign flips for H(div) basis functions */
    double Basis_Sign[NB] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // init to all positive (no sign change)
    Get_Basis_Sign_Change(Mesh->Orientation, Basis_Sign);
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
// Local Element defined on Subdomain: CG, raviart_thomas_deg1_dim2
// the Subdomain             has topological dimension = 2
// the Domain of Integration has topological dimension = 1
// geometric dimension = 3
// Number of Quadrature Points = 5

    // get "Val" of basis functions
    VEC_2x1 phi_Val[NQ][NB];

    phi_Val[0][0].Set_Equal_To(0.00000000000000000E+00, -1.90617984593866385E+00);
    phi_Val[0][1].Set_Equal_To(0.00000000000000000E+00, 3.45468351824816899E+00);
    phi_Val[0][2].Set_Equal_To(-3.71853953781599156E+00, 3.45468351824816899E+00);
    phi_Val[0][3].Set_Equal_To(1.71853953781599156E+00, -1.54850367230950492E+00);
    phi_Val[0][4].Set_Equal_To(0.00000000000000000E+00, 1.70035865506486583E-01);
    phi_Val[0][5].Set_Equal_To(0.00000000000000000E+00, 9.38201540613361473E-02);
    phi_Val[0][6].Set_Equal_To(0.00000000000000000E+00, 3.57676173629158878E-01);
    phi_Val[0][7].Set_Equal_To(0.00000000000000000E+00, 7.15352347258317756E-01);
    phi_Val[1][0].Set_Equal_To(0.00000000000000000E+00, -1.53846931010568322E+00);
    phi_Val[1][1].Set_Equal_To(0.00000000000000000E+00, 1.65683701606274680E+00);
    phi_Val[1][2].Set_Equal_To(-2.61540793031704943E+00, 1.65683701606274680E+00);
    phi_Val[1][3].Set_Equal_To(6.15407930317049323E-01, -1.18367705957063751E-01);
    phi_Val[1][4].Set_Equal_To(0.00000000000000000E+00, 4.97040224359985572E-01);
    phi_Val[1][5].Set_Equal_To(0.00000000000000000E+00, 4.61530689894316892E-01);
    phi_Val[1][6].Set_Equal_To(0.00000000000000000E+00, 1.42010160414861941E+00);
    phi_Val[1][7].Set_Equal_To(0.00000000000000000E+00, 2.84020320829723882E+00);
    phi_Val[2][0].Set_Equal_To(0.00000000000000000E+00, -1.00000000000000000E+00);
    phi_Val[2][1].Set_Equal_To(0.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][2].Set_Equal_To(-1.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][3].Set_Equal_To(-1.00000000000000000E+00, 1.00000000000000000E+00);
    phi_Val[2][4].Set_Equal_To(0.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][5].Set_Equal_To(0.00000000000000000E+00, 1.00000000000000000E+00);
    phi_Val[2][6].Set_Equal_To(0.00000000000000000E+00, 2.00000000000000000E+00);
    phi_Val[2][7].Set_Equal_To(0.00000000000000000E+00, 4.00000000000000000E+00);
    phi_Val[3][0].Set_Equal_To(0.00000000000000000E+00, -4.61530689894317003E-01);
    phi_Val[3][1].Set_Equal_To(0.00000000000000000E+00, -4.97040224359985572E-01);
    phi_Val[3][2].Set_Equal_To(6.15407930317048990E-01, -4.97040224359985572E-01);
    phi_Val[3][3].Set_Equal_To(-2.61540793031704899E+00, 9.58570914254302631E-01);
    phi_Val[3][4].Set_Equal_To(0.00000000000000000E+00, -1.65683701606274636E+00);
    phi_Val[3][5].Set_Equal_To(0.00000000000000000E+00, 1.53846931010568300E+00);
    phi_Val[3][6].Set_Equal_To(0.00000000000000000E+00, 1.42010160414861963E+00);
    phi_Val[3][7].Set_Equal_To(0.00000000000000000E+00, 2.84020320829723927E+00);
    phi_Val[4][0].Set_Equal_To(0.00000000000000000E+00, -9.38201540613361473E-02);
    phi_Val[4][1].Set_Equal_To(0.00000000000000000E+00, -1.70035865506486583E-01);
    phi_Val[4][2].Set_Equal_To(1.71853953781599156E+00, -1.70035865506486583E-01);
    phi_Val[4][3].Set_Equal_To(-3.71853953781599156E+00, 2.63856019567822730E-01);
    phi_Val[4][4].Set_Equal_To(0.00000000000000000E+00, -3.45468351824816899E+00);
    phi_Val[4][5].Set_Equal_To(0.00000000000000000E+00, 1.90617984593866385E+00);
    phi_Val[4][6].Set_Equal_To(0.00000000000000000E+00, 3.57676173629158878E-01);
    phi_Val[4][7].Set_Equal_To(0.00000000000000000E+00, 7.15352347258317756E-01);


// // set of quadrature points
// static const double Quad_Points[NQ][SUB_TD] = { \
//     4.69100770306680737E-02, \
//     2.30765344947158446E-01, \
//     5.00000000000000000E-01, \
//     7.69234655052841498E-01, \
//     9.53089922969331926E-01  \
//     };

// // set of quadrature weights
// static const double Quad_Weights[NQ] = { \
//     1.18463442528094529E-01, \
//     2.39314335249683374E-01, \
//     2.84444444444444333E-01, \
//     2.39314335249683430E-01, \
//     1.18463442528094445E-01  \
//     };
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
    /*** compute basis function quantities ***/
    // map basis vectors over (indexing is in the C style)
    // loop through quad points
    for (int qp_i = 0; (qp_i < Num_QP); qp_i++)
        {
        // evaluate for each basis function
        for (int basis_i = 0; (basis_i < Num_Basis); basis_i++)
            {
            VEC_2x1 vv_temp;
            // pre-multiply by sign change
            const double INV_DET_SIGN_FLIP = Basis_Sign[basis_i] * Mesh->Map_Inv_Det_Jac[qp_i].a;
            // pre-multiply by 1/det(Jac)
            Scalar_Mult_Vector(phi_Val[qp_i][basis_i], INV_DET_SIGN_FLIP, vv_temp);
            Mat_Vec(Mesh->Map_PHI_Grad[qp_i], vv_temp, Func_vv_Value[basis_i][qp_i]);
            }
        }
/*------------   END: Auto Generate ------------*/
}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* map basis functions from the standard reference element
       to an actual element in the Domain.     */
void SpecificFUNC::Map_Basis_n2()
{

/*------------ BEGIN: Auto Generate ------------*/
    /* call sub-routine to determine sign flips for H(div) basis functions */
    double Basis_Sign[NB] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // init to all positive (no sign change)
    Get_Basis_Sign_Change(Mesh->Orientation, Basis_Sign);
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
// Local Element defined on Subdomain: CG, raviart_thomas_deg1_dim2
// the Subdomain             has topological dimension = 2
// the Domain of Integration has topological dimension = 1
// geometric dimension = 3
// Number of Quadrature Points = 5

    // get "Val" of basis functions
    VEC_2x1 phi_Val[NQ][NB];

    phi_Val[0][0].Set_Equal_To(0.00000000000000000E+00, -9.38201540613361473E-02);
    phi_Val[0][1].Set_Equal_To(0.00000000000000000E+00, -1.70035865506486583E-01);
    phi_Val[0][2].Set_Equal_To(1.71853953781599156E+00, -1.70035865506486583E-01);
    phi_Val[0][3].Set_Equal_To(-3.71853953781599156E+00, 2.63856019567822730E-01);
    phi_Val[0][4].Set_Equal_To(0.00000000000000000E+00, -3.45468351824816899E+00);
    phi_Val[0][5].Set_Equal_To(0.00000000000000000E+00, 1.90617984593866385E+00);
    phi_Val[0][6].Set_Equal_To(0.00000000000000000E+00, 3.57676173629158878E-01);
    phi_Val[0][7].Set_Equal_To(0.00000000000000000E+00, 7.15352347258317756E-01);
    phi_Val[1][0].Set_Equal_To(0.00000000000000000E+00, -4.61530689894316892E-01);
    phi_Val[1][1].Set_Equal_To(0.00000000000000000E+00, -4.97040224359985572E-01);
    phi_Val[1][2].Set_Equal_To(6.15407930317049323E-01, -4.97040224359985572E-01);
    phi_Val[1][3].Set_Equal_To(-2.61540793031704943E+00, 9.58570914254302520E-01);
    phi_Val[1][4].Set_Equal_To(0.00000000000000000E+00, -1.65683701606274680E+00);
    phi_Val[1][5].Set_Equal_To(0.00000000000000000E+00, 1.53846931010568322E+00);
    phi_Val[1][6].Set_Equal_To(0.00000000000000000E+00, 1.42010160414861941E+00);
    phi_Val[1][7].Set_Equal_To(0.00000000000000000E+00, 2.84020320829723882E+00);
    phi_Val[2][0].Set_Equal_To(0.00000000000000000E+00, -1.00000000000000000E+00);
    phi_Val[2][1].Set_Equal_To(0.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][2].Set_Equal_To(-1.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][3].Set_Equal_To(-1.00000000000000000E+00, 1.00000000000000000E+00);
    phi_Val[2][4].Set_Equal_To(0.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][5].Set_Equal_To(0.00000000000000000E+00, 1.00000000000000000E+00);
    phi_Val[2][6].Set_Equal_To(0.00000000000000000E+00, 2.00000000000000000E+00);
    phi_Val[2][7].Set_Equal_To(0.00000000000000000E+00, 4.00000000000000000E+00);
    phi_Val[3][0].Set_Equal_To(0.00000000000000000E+00, -1.53846931010568300E+00);
    phi_Val[3][1].Set_Equal_To(0.00000000000000000E+00, 1.65683701606274636E+00);
    phi_Val[3][2].Set_Equal_To(-2.61540793031704899E+00, 1.65683701606274636E+00);
    phi_Val[3][3].Set_Equal_To(6.15407930317048990E-01, -1.18367705957063404E-01);
    phi_Val[3][4].Set_Equal_To(0.00000000000000000E+00, 4.97040224359985572E-01);
    phi_Val[3][5].Set_Equal_To(0.00000000000000000E+00, 4.61530689894317003E-01);
    phi_Val[3][6].Set_Equal_To(0.00000000000000000E+00, 1.42010160414861963E+00);
    phi_Val[3][7].Set_Equal_To(0.00000000000000000E+00, 2.84020320829723927E+00);
    phi_Val[4][0].Set_Equal_To(0.00000000000000000E+00, -1.90617984593866385E+00);
    phi_Val[4][1].Set_Equal_To(0.00000000000000000E+00, 3.45468351824816899E+00);
    phi_Val[4][2].Set_Equal_To(-3.71853953781599156E+00, 3.45468351824816899E+00);
    phi_Val[4][3].Set_Equal_To(1.71853953781599156E+00, -1.54850367230950492E+00);
    phi_Val[4][4].Set_Equal_To(0.00000000000000000E+00, 1.70035865506486583E-01);
    phi_Val[4][5].Set_Equal_To(0.00000000000000000E+00, 9.38201540613361473E-02);
    phi_Val[4][6].Set_Equal_To(0.00000000000000000E+00, 3.57676173629158878E-01);
    phi_Val[4][7].Set_Equal_To(0.00000000000000000E+00, 7.15352347258317756E-01);


// // set of quadrature points
// static const double Quad_Points[NQ][SUB_TD] = { \
//     4.69100770306680737E-02, \
//     2.30765344947158446E-01, \
//     5.00000000000000000E-01, \
//     7.69234655052841498E-01, \
//     9.53089922969331926E-01  \
//     };

// // set of quadrature weights
// static const double Quad_Weights[NQ] = { \
//     1.18463442528094529E-01, \
//     2.39314335249683374E-01, \
//     2.84444444444444333E-01, \
//     2.39314335249683430E-01, \
//     1.18463442528094445E-01  \
//     };
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
    /*** compute basis function quantities ***/
    // map basis vectors over (indexing is in the C style)
    // loop through quad points
    for (int qp_i = 0; (qp_i < Num_QP); qp_i++)
        {
        // evaluate for each basis function
        for (int basis_i = 0; (basis_i < Num_Basis); basis_i++)
            {
            VEC_2x1 vv_temp;
            // pre-multiply by sign change
            const double INV_DET_SIGN_FLIP = Basis_Sign[basis_i] * Mesh->Map_Inv_Det_Jac[qp_i].a;
            // pre-multiply by 1/det(Jac)
            Scalar_Mult_Vector(phi_Val[qp_i][basis_i], INV_DET_SIGN_FLIP, vv_temp);
            Mat_Vec(Mesh->Map_PHI_Grad[qp_i], vv_temp, Func_vv_Value[basis_i][qp_i]);
            }
        }
/*------------   END: Auto Generate ------------*/
}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* map basis functions from the standard reference element
       to an actual element in the Domain.     */
void SpecificFUNC::Map_Basis_p3()
{

/*------------ BEGIN: Auto Generate ------------*/
    /* call sub-routine to determine sign flips for H(div) basis functions */
    double Basis_Sign[NB] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // init to all positive (no sign change)
    Get_Basis_Sign_Change(Mesh->Orientation, Basis_Sign);
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
// Local Element defined on Subdomain: CG, raviart_thomas_deg1_dim2
// the Subdomain             has topological dimension = 2
// the Domain of Integration has topological dimension = 1
// geometric dimension = 3
// Number of Quadrature Points = 5

    // get "Val" of basis functions
    VEC_2x1 phi_Val[NQ][NB];

    phi_Val[0][0].Set_Equal_To(-1.70035865506486583E-01, 0.00000000000000000E+00);
    phi_Val[0][1].Set_Equal_To(-9.38201540613361473E-02, 0.00000000000000000E+00);
    phi_Val[0][2].Set_Equal_To(1.90617984593866385E+00, 0.00000000000000000E+00);
    phi_Val[0][3].Set_Equal_To(-3.45468351824816899E+00, 0.00000000000000000E+00);
    phi_Val[0][4].Set_Equal_To(2.63856019567822730E-01, -3.71853953781599156E+00);
    phi_Val[0][5].Set_Equal_To(-1.70035865506486583E-01, 1.71853953781599156E+00);
    phi_Val[0][6].Set_Equal_To(7.15352347258317756E-01, 0.00000000000000000E+00);
    phi_Val[0][7].Set_Equal_To(3.57676173629158878E-01, 0.00000000000000000E+00);
    phi_Val[1][0].Set_Equal_To(-4.97040224359985572E-01, 0.00000000000000000E+00);
    phi_Val[1][1].Set_Equal_To(-4.61530689894316892E-01, 0.00000000000000000E+00);
    phi_Val[1][2].Set_Equal_To(1.53846931010568322E+00, 0.00000000000000000E+00);
    phi_Val[1][3].Set_Equal_To(-1.65683701606274680E+00, 0.00000000000000000E+00);
    phi_Val[1][4].Set_Equal_To(9.58570914254302520E-01, -2.61540793031704943E+00);
    phi_Val[1][5].Set_Equal_To(-4.97040224359985572E-01, 6.15407930317049323E-01);
    phi_Val[1][6].Set_Equal_To(2.84020320829723882E+00, 0.00000000000000000E+00);
    phi_Val[1][7].Set_Equal_To(1.42010160414861941E+00, 0.00000000000000000E+00);
    phi_Val[2][0].Set_Equal_To(0.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][1].Set_Equal_To(-1.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][2].Set_Equal_To(1.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][3].Set_Equal_To(0.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][4].Set_Equal_To(1.00000000000000000E+00, -1.00000000000000000E+00);
    phi_Val[2][5].Set_Equal_To(0.00000000000000000E+00, -1.00000000000000000E+00);
    phi_Val[2][6].Set_Equal_To(4.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][7].Set_Equal_To(2.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[3][0].Set_Equal_To(1.65683701606274636E+00, 0.00000000000000000E+00);
    phi_Val[3][1].Set_Equal_To(-1.53846931010568300E+00, 0.00000000000000000E+00);
    phi_Val[3][2].Set_Equal_To(4.61530689894317003E-01, 0.00000000000000000E+00);
    phi_Val[3][3].Set_Equal_To(4.97040224359985572E-01, 0.00000000000000000E+00);
    phi_Val[3][4].Set_Equal_To(-1.18367705957063404E-01, 6.15407930317048990E-01);
    phi_Val[3][5].Set_Equal_To(1.65683701606274636E+00, -2.61540793031704899E+00);
    phi_Val[3][6].Set_Equal_To(2.84020320829723927E+00, 0.00000000000000000E+00);
    phi_Val[3][7].Set_Equal_To(1.42010160414861963E+00, 0.00000000000000000E+00);
    phi_Val[4][0].Set_Equal_To(3.45468351824816899E+00, 0.00000000000000000E+00);
    phi_Val[4][1].Set_Equal_To(-1.90617984593866385E+00, 0.00000000000000000E+00);
    phi_Val[4][2].Set_Equal_To(9.38201540613361473E-02, 0.00000000000000000E+00);
    phi_Val[4][3].Set_Equal_To(1.70035865506486583E-01, 0.00000000000000000E+00);
    phi_Val[4][4].Set_Equal_To(-1.54850367230950492E+00, 1.71853953781599156E+00);
    phi_Val[4][5].Set_Equal_To(3.45468351824816899E+00, -3.71853953781599156E+00);
    phi_Val[4][6].Set_Equal_To(7.15352347258317756E-01, 0.00000000000000000E+00);
    phi_Val[4][7].Set_Equal_To(3.57676173629158878E-01, 0.00000000000000000E+00);


// // set of quadrature points
// static const double Quad_Points[NQ][SUB_TD] = { \
//     4.69100770306680737E-02, \
//     2.30765344947158446E-01, \
//     5.00000000000000000E-01, \
//     7.69234655052841498E-01, \
//     9.53089922969331926E-01  \
//     };

// // set of quadrature weights
// static const double Quad_Weights[NQ] = { \
//     1.18463442528094529E-01, \
//     2.39314335249683374E-01, \
//     2.84444444444444333E-01, \
//     2.39314335249683430E-01, \
//     1.18463442528094445E-01  \
//     };
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
    /*** compute basis function quantities ***/
    // map basis vectors over (indexing is in the C style)
    // loop through quad points
    for (int qp_i = 0; (qp_i < Num_QP); qp_i++)
        {
        // evaluate for each basis function
        for (int basis_i = 0; (basis_i < Num_Basis); basis_i++)
            {
            VEC_2x1 vv_temp;
            // pre-multiply by sign change
            const double INV_DET_SIGN_FLIP = Basis_Sign[basis_i] * Mesh->Map_Inv_Det_Jac[qp_i].a;
            // pre-multiply by 1/det(Jac)
            Scalar_Mult_Vector(phi_Val[qp_i][basis_i], INV_DET_SIGN_FLIP, vv_temp);
            Mat_Vec(Mesh->Map_PHI_Grad[qp_i], vv_temp, Func_vv_Value[basis_i][qp_i]);
            }
        }
/*------------   END: Auto Generate ------------*/
}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* map basis functions from the standard reference element
       to an actual element in the Domain.     */
void SpecificFUNC::Map_Basis_n3()
{

/*------------ BEGIN: Auto Generate ------------*/
    /* call sub-routine to determine sign flips for H(div) basis functions */
    double Basis_Sign[NB] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // init to all positive (no sign change)
    Get_Basis_Sign_Change(Mesh->Orientation, Basis_Sign);
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
// Local Element defined on Subdomain: CG, raviart_thomas_deg1_dim2
// the Subdomain             has topological dimension = 2
// the Domain of Integration has topological dimension = 1
// geometric dimension = 3
// Number of Quadrature Points = 5

    // get "Val" of basis functions
    VEC_2x1 phi_Val[NQ][NB];

    phi_Val[0][0].Set_Equal_To(3.45468351824816899E+00, 0.00000000000000000E+00);
    phi_Val[0][1].Set_Equal_To(-1.90617984593866385E+00, 0.00000000000000000E+00);
    phi_Val[0][2].Set_Equal_To(9.38201540613361473E-02, 0.00000000000000000E+00);
    phi_Val[0][3].Set_Equal_To(1.70035865506486583E-01, 0.00000000000000000E+00);
    phi_Val[0][4].Set_Equal_To(-1.54850367230950492E+00, 1.71853953781599156E+00);
    phi_Val[0][5].Set_Equal_To(3.45468351824816899E+00, -3.71853953781599156E+00);
    phi_Val[0][6].Set_Equal_To(7.15352347258317756E-01, 0.00000000000000000E+00);
    phi_Val[0][7].Set_Equal_To(3.57676173629158878E-01, 0.00000000000000000E+00);
    phi_Val[1][0].Set_Equal_To(1.65683701606274680E+00, 0.00000000000000000E+00);
    phi_Val[1][1].Set_Equal_To(-1.53846931010568322E+00, 0.00000000000000000E+00);
    phi_Val[1][2].Set_Equal_To(4.61530689894316892E-01, 0.00000000000000000E+00);
    phi_Val[1][3].Set_Equal_To(4.97040224359985572E-01, 0.00000000000000000E+00);
    phi_Val[1][4].Set_Equal_To(-1.18367705957063751E-01, 6.15407930317049323E-01);
    phi_Val[1][5].Set_Equal_To(1.65683701606274680E+00, -2.61540793031704943E+00);
    phi_Val[1][6].Set_Equal_To(2.84020320829723882E+00, 0.00000000000000000E+00);
    phi_Val[1][7].Set_Equal_To(1.42010160414861941E+00, 0.00000000000000000E+00);
    phi_Val[2][0].Set_Equal_To(0.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][1].Set_Equal_To(-1.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][2].Set_Equal_To(1.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][3].Set_Equal_To(0.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][4].Set_Equal_To(1.00000000000000000E+00, -1.00000000000000000E+00);
    phi_Val[2][5].Set_Equal_To(0.00000000000000000E+00, -1.00000000000000000E+00);
    phi_Val[2][6].Set_Equal_To(4.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[2][7].Set_Equal_To(2.00000000000000000E+00, 0.00000000000000000E+00);
    phi_Val[3][0].Set_Equal_To(-4.97040224359985572E-01, 0.00000000000000000E+00);
    phi_Val[3][1].Set_Equal_To(-4.61530689894317003E-01, 0.00000000000000000E+00);
    phi_Val[3][2].Set_Equal_To(1.53846931010568300E+00, 0.00000000000000000E+00);
    phi_Val[3][3].Set_Equal_To(-1.65683701606274636E+00, 0.00000000000000000E+00);
    phi_Val[3][4].Set_Equal_To(9.58570914254302631E-01, -2.61540793031704899E+00);
    phi_Val[3][5].Set_Equal_To(-4.97040224359985572E-01, 6.15407930317048990E-01);
    phi_Val[3][6].Set_Equal_To(2.84020320829723927E+00, 0.00000000000000000E+00);
    phi_Val[3][7].Set_Equal_To(1.42010160414861963E+00, 0.00000000000000000E+00);
    phi_Val[4][0].Set_Equal_To(-1.70035865506486583E-01, 0.00000000000000000E+00);
    phi_Val[4][1].Set_Equal_To(-9.38201540613361473E-02, 0.00000000000000000E+00);
    phi_Val[4][2].Set_Equal_To(1.90617984593866385E+00, 0.00000000000000000E+00);
    phi_Val[4][3].Set_Equal_To(-3.45468351824816899E+00, 0.00000000000000000E+00);
    phi_Val[4][4].Set_Equal_To(2.63856019567822730E-01, -3.71853953781599156E+00);
    phi_Val[4][5].Set_Equal_To(-1.70035865506486583E-01, 1.71853953781599156E+00);
    phi_Val[4][6].Set_Equal_To(7.15352347258317756E-01, 0.00000000000000000E+00);
    phi_Val[4][7].Set_Equal_To(3.57676173629158878E-01, 0.00000000000000000E+00);


// // set of quadrature points
// static const double Quad_Points[NQ][SUB_TD] = { \
//     4.69100770306680737E-02, \
//     2.30765344947158446E-01, \
//     5.00000000000000000E-01, \
//     7.69234655052841498E-01, \
//     9.53089922969331926E-01  \
//     };

// // set of quadrature weights
// static const double Quad_Weights[NQ] = { \
//     1.18463442528094529E-01, \
//     2.39314335249683374E-01, \
//     2.84444444444444333E-01, \
//     2.39314335249683430E-01, \
//     1.18463442528094445E-01  \
//     };
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
    /*** compute basis function quantities ***/
    // map basis vectors over (indexing is in the C style)
    // loop through quad points
    for (int qp_i = 0; (qp_i < Num_QP); qp_i++)
        {
        // evaluate for each basis function
        for (int basis_i = 0; (basis_i < Num_Basis); basis_i++)
            {
            VEC_2x1 vv_temp;
            // pre-multiply by sign change
            const double INV_DET_SIGN_FLIP = Basis_Sign[basis_i] * Mesh->Map_Inv_Det_Jac[qp_i].a;
            // pre-multiply by 1/det(Jac)
            Scalar_Mult_Vector(phi_Val[qp_i][basis_i], INV_DET_SIGN_FLIP, vv_temp);
            Mat_Vec(Mesh->Map_PHI_Grad[qp_i], vv_temp, Func_vv_Value[basis_i][qp_i]);
            }
        }
/*------------   END: Auto Generate ------------*/
}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/

// remove those macros!
#undef SpecificFUNC
#undef SpecificFUNC_str

#undef SPACE_type
#undef SPACE_name
#undef SUB_TD
#undef DOI_TD
#undef GD
#undef NC
#undef NB
#undef NQ

/***/
