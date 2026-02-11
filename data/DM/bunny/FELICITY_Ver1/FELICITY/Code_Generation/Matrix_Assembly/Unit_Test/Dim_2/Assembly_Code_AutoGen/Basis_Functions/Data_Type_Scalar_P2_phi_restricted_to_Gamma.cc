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
#define SpecificFUNC        Data_Type_Scalar_P2_phi_restricted_to_Gamma
#define SpecificFUNC_str   "Data_Type_Scalar_P2_phi_restricted_to_Gamma"

// set the type of function space
#define SPACE_type  "CG - lagrange_deg2_dim2"
// set the name of function space
#define SPACE_name  "Scalar_P2"

// set the Subdomain topological dimension
#define SUB_TD  2
// set the Domain of Integration (DoI) topological dimension
#define DOI_TD  2
// set the geometric dimension
#define GD  3
// set the number of cartesian tuple components (m*n) = 1 * 1
#define NC  1
// NOTE: the (i,j) tuple component is accessed by the linear index k = i + (j-1)*m
// set the number of quad points
#define NQ  19
// set the number of basis functions
#define NB  6
/*------------   END: Auto Generate ------------*/

/* C++ (Specific) FE Function class definition */
class SpecificFUNC: public ABSTRACT_FEM_Function_Class // derive from base class
{
public:
    int*     Elem_DoF[NB];    // element DoF list

    // data structure containing information on the function evaluations.
    // Note: this data is evaluated at several quadrature points!
    // local function evaluated at a quadrature point in reference element
    // (this is a pointer because it will change depending on the local mesh entity)
    SCALAR  (*Func_f_Value)[NB][NQ];
    // intrinsic gradient of basis function
    VEC_3x1 Func_f_Grad[NB][NQ];

    // constructor
    SpecificFUNC ();
    ~SpecificFUNC (); // destructor
    void Setup_Function_Space(const mxArray*);
    void Get_Local_to_Global_DoFmap(const int&, int*) const;
                   // need the "const" to ENSURE that nothing in this object will change!
    void Transform_Basis_Functions();
    const CLASS_geom_Gamma_embedded_in_Gamma_restricted_to_Gamma*  Mesh;

private:
    void Map_Basis_p1();
    SCALAR Value_p1[NB][NQ];
    void Basis_Value_p1(SCALAR V[NB][NQ]);
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
        Func_f_Grad[basis_i][qp_i].Set_To_Zero();
    // init basis function values on local mesh entities
    Basis_Value_p1(Value_p1);
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
/* compute the correct local transformation */
void SpecificFUNC::Transform_Basis_Functions()
{
    Map_Basis_p1();
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
// Local Element defined on Subdomain: CG, lagrange_deg2_dim2
// the Subdomain             has topological dimension = 2
// the Domain of Integration has topological dimension = 2
// geometric dimension = 3
// Number of Quadrature Points = 19

    // get "Grad" of basis functions
    VEC_2x1 phi_Grad[NQ][NB];

    phi_Grad[0][0].Set_Equal_To(-3.33333333333333315E-01, -3.33333333333333315E-01);
    phi_Grad[0][1].Set_Equal_To(3.33333333333333315E-01, 0.00000000000000000E+00);
    phi_Grad[0][2].Set_Equal_To(0.00000000000000000E+00, 3.33333333333333315E-01);
    phi_Grad[0][3].Set_Equal_To(1.33333333333333326E+00, 1.33333333333333326E+00);
    phi_Grad[0][4].Set_Equal_To(-1.33333333333333326E+00, 0.00000000000000000E+00);
    phi_Grad[0][5].Set_Equal_To(0.00000000000000000E+00, -1.33333333333333326E+00);
    phi_Grad[1][0].Set_Equal_To(9.17460153589900962E-01, 9.17460153589900962E-01);
    phi_Grad[1][1].Set_Equal_To(9.58730076794950481E-01, 0.00000000000000000E+00);
    phi_Grad[1][2].Set_Equal_To(0.00000000000000000E+00, 9.58730076794950481E-01);
    phi_Grad[1][3].Set_Equal_To(1.95873007679495048E+00, 1.95873007679495048E+00);
    phi_Grad[1][4].Set_Equal_To(-1.95873007679495048E+00, -1.87619023038485144E+00);
    phi_Grad[1][5].Set_Equal_To(-1.87619023038485144E+00, -1.95873007679495048E+00);
    phi_Grad[2][0].Set_Equal_To(-9.58730076794950592E-01, -9.58730076794950592E-01);
    phi_Grad[2][1].Set_Equal_To(9.58730076794950481E-01, 0.00000000000000000E+00);
    phi_Grad[2][2].Set_Equal_To(0.00000000000000000E+00, -9.17460153589900962E-01);
    phi_Grad[2][3].Set_Equal_To(8.25398464100989826E-02, 1.95873007679495048E+00);
    phi_Grad[2][4].Set_Equal_To(-8.25398464100989826E-02, 1.87619023038485166E+00);
    phi_Grad[2][5].Set_Equal_To(5.55111512312578270E-17, -1.95873007679495048E+00);
    phi_Grad[3][0].Set_Equal_To(-9.58730076794950592E-01, -9.58730076794950592E-01);
    phi_Grad[3][1].Set_Equal_To(-9.17460153589900962E-01, 0.00000000000000000E+00);
    phi_Grad[3][2].Set_Equal_To(0.00000000000000000E+00, 9.58730076794950481E-01);
    phi_Grad[3][3].Set_Equal_To(1.95873007679495048E+00, 8.25398464100989826E-02);
    phi_Grad[3][4].Set_Equal_To(-1.95873007679495048E+00, 5.55111512312578270E-17);
    phi_Grad[3][5].Set_Equal_To(1.87619023038485166E+00, -8.25398464100989826E-02);
    phi_Grad[4][0].Set_Equal_To(4.96716731943493084E-01, 4.96716731943493084E-01);
    phi_Grad[4][1].Set_Equal_To(7.48358365971746542E-01, 0.00000000000000000E+00);
    phi_Grad[4][2].Set_Equal_To(0.00000000000000000E+00, 7.48358365971746542E-01);
    phi_Grad[4][3].Set_Equal_To(1.74835836597174654E+00, 1.74835836597174654E+00);
    phi_Grad[4][4].Set_Equal_To(-1.74835836597174654E+00, -1.24507509791523963E+00);
    phi_Grad[4][5].Set_Equal_To(-1.24507509791523963E+00, -1.74835836597174654E+00);
    phi_Grad[5][0].Set_Equal_To(-7.48358365971746542E-01, -7.48358365971746542E-01);
    phi_Grad[5][1].Set_Equal_To(7.48358365971746542E-01, 0.00000000000000000E+00);
    phi_Grad[5][2].Set_Equal_To(0.00000000000000000E+00, -4.96716731943493084E-01);
    phi_Grad[5][3].Set_Equal_To(5.03283268056506916E-01, 1.74835836597174654E+00);
    phi_Grad[5][4].Set_Equal_To(-5.03283268056506916E-01, 1.24507509791523963E+00);
    phi_Grad[5][5].Set_Equal_To(0.00000000000000000E+00, -1.74835836597174654E+00);
    phi_Grad[6][0].Set_Equal_To(-7.48358365971746542E-01, -7.48358365971746542E-01);
    phi_Grad[6][1].Set_Equal_To(-4.96716731943493084E-01, 0.00000000000000000E+00);
    phi_Grad[6][2].Set_Equal_To(0.00000000000000000E+00, 7.48358365971746542E-01);
    phi_Grad[6][3].Set_Equal_To(1.74835836597174654E+00, 5.03283268056506916E-01);
    phi_Grad[6][4].Set_Equal_To(-1.74835836597174654E+00, 0.00000000000000000E+00);
    phi_Grad[6][5].Set_Equal_To(1.24507509791523963E+00, -5.03283268056506916E-01);
    phi_Grad[7][0].Set_Equal_To(-1.49437171504773825E+00, -1.49437171504773825E+00);
    phi_Grad[7][1].Set_Equal_To(-2.47185857523869124E-01, 0.00000000000000000E+00);
    phi_Grad[7][2].Set_Equal_To(0.00000000000000000E+00, -2.47185857523869124E-01);
    phi_Grad[7][3].Set_Equal_To(7.52814142476130876E-01, 7.52814142476130876E-01);
    phi_Grad[7][4].Set_Equal_To(-7.52814142476130876E-01, 1.74155757257160748E+00);
    phi_Grad[7][5].Set_Equal_To(1.74155757257160748E+00, -7.52814142476130876E-01);
    phi_Grad[8][0].Set_Equal_To(2.47185857523869346E-01, 2.47185857523869346E-01);
    phi_Grad[8][1].Set_Equal_To(-2.47185857523869124E-01, 0.00000000000000000E+00);
    phi_Grad[8][2].Set_Equal_To(0.00000000000000000E+00, 1.49437171504773847E+00);
    phi_Grad[8][3].Set_Equal_To(2.49437171504773847E+00, 7.52814142476130876E-01);
    phi_Grad[8][4].Set_Equal_To(-2.49437171504773847E+00, -1.74155757257160793E+00);
    phi_Grad[8][5].Set_Equal_To(-2.22044604925031308E-16, -7.52814142476130876E-01);
    phi_Grad[9][0].Set_Equal_To(2.47185857523869346E-01, 2.47185857523869346E-01);
    phi_Grad[9][1].Set_Equal_To(1.49437171504773847E+00, 0.00000000000000000E+00);
    phi_Grad[9][2].Set_Equal_To(0.00000000000000000E+00, -2.47185857523869124E-01);
    phi_Grad[9][3].Set_Equal_To(7.52814142476130876E-01, 2.49437171504773847E+00);
    phi_Grad[9][4].Set_Equal_To(-7.52814142476130876E-01, -2.22044604925031308E-16);
    phi_Grad[9][5].Set_Equal_To(-1.74155757257160793E+00, -2.49437171504773847E+00);
    phi_Grad[10][0].Set_Equal_To(-2.64216389284437847E+00, -2.64216389284437847E+00);
    phi_Grad[10][1].Set_Equal_To(-8.21081946422189124E-01, 0.00000000000000000E+00);
    phi_Grad[10][2].Set_Equal_To(0.00000000000000000E+00, -8.21081946422189124E-01);
    phi_Grad[10][3].Set_Equal_To(1.78918053577810848E-01, 1.78918053577810848E-01);
    phi_Grad[10][4].Set_Equal_To(-1.78918053577810848E-01, 3.46324583926656748E+00);
    phi_Grad[10][5].Set_Equal_To(3.46324583926656748E+00, -1.78918053577810848E-01);
    phi_Grad[11][0].Set_Equal_To(8.21081946422189346E-01, 8.21081946422189346E-01);
    phi_Grad[11][1].Set_Equal_To(-8.21081946422189124E-01, 0.00000000000000000E+00);
    phi_Grad[11][2].Set_Equal_To(0.00000000000000000E+00, 2.64216389284437847E+00);
    phi_Grad[11][3].Set_Equal_To(3.64216389284437847E+00, 1.78918053577810848E-01);
    phi_Grad[11][4].Set_Equal_To(-3.64216389284437847E+00, -3.46324583926656793E+00);
    phi_Grad[11][5].Set_Equal_To(-1.66533453693773481E-16, -1.78918053577810848E-01);
    phi_Grad[12][0].Set_Equal_To(8.21081946422189346E-01, 8.21081946422189346E-01);
    phi_Grad[12][1].Set_Equal_To(2.64216389284437847E+00, 0.00000000000000000E+00);
    phi_Grad[12][2].Set_Equal_To(0.00000000000000000E+00, -8.21081946422189124E-01);
    phi_Grad[12][3].Set_Equal_To(1.78918053577810848E-01, 3.64216389284437847E+00);
    phi_Grad[12][4].Set_Equal_To(-1.78918053577810848E-01, -1.66533453693773481E-16);
    phi_Grad[12][5].Set_Equal_To(-3.46324583926656793E+00, -3.64216389284437847E+00);
    phi_Grad[13][0].Set_Equal_To(-1.96479439513799203E+00, -1.96479439513799203E+00);
    phi_Grad[13][1].Set_Equal_To(-8.52646351781054856E-01, 0.00000000000000000E+00);
    phi_Grad[13][2].Set_Equal_To(0.00000000000000000E+00, -1.12148043356937177E-01);
    phi_Grad[13][3].Set_Equal_To(8.87851956643062823E-01, 1.47353648218945144E-01);
    phi_Grad[13][4].Set_Equal_To(-8.87851956643062823E-01, 2.07694243849492910E+00);
    phi_Grad[13][5].Set_Equal_To(2.81744074691904700E+00, -1.47353648218945144E-01);
    phi_Grad[14][0].Set_Equal_To(-1.96479439513799203E+00, -1.96479439513799203E+00);
    phi_Grad[14][1].Set_Equal_To(-1.12148043356937177E-01, 0.00000000000000000E+00);
    phi_Grad[14][2].Set_Equal_To(0.00000000000000000E+00, -8.52646351781054856E-01);
    phi_Grad[14][3].Set_Equal_To(1.47353648218945144E-01, 8.87851956643062823E-01);
    phi_Grad[14][4].Set_Equal_To(-1.47353648218945144E-01, 2.81744074691904700E+00);
    phi_Grad[14][5].Set_Equal_To(2.07694243849492910E+00, -8.87851956643062823E-01);
    phi_Grad[15][0].Set_Equal_To(1.12148043356937177E-01, 1.12148043356937177E-01);
    phi_Grad[15][1].Set_Equal_To(-8.52646351781054856E-01, 0.00000000000000000E+00);
    phi_Grad[15][2].Set_Equal_To(0.00000000000000000E+00, 1.96479439513799203E+00);
    phi_Grad[15][3].Set_Equal_To(2.96479439513799203E+00, 1.47353648218945144E-01);
    phi_Grad[15][4].Set_Equal_To(-2.96479439513799203E+00, -2.07694243849492910E+00);
    phi_Grad[15][5].Set_Equal_To(7.40498308424117679E-01, -1.47353648218945144E-01);
    phi_Grad[16][0].Set_Equal_To(1.12148043356937177E-01, 1.12148043356937177E-01);
    phi_Grad[16][1].Set_Equal_To(1.96479439513799203E+00, 0.00000000000000000E+00);
    phi_Grad[16][2].Set_Equal_To(0.00000000000000000E+00, -8.52646351781054856E-01);
    phi_Grad[16][3].Set_Equal_To(1.47353648218945144E-01, 2.96479439513799203E+00);
    phi_Grad[16][4].Set_Equal_To(-1.47353648218945144E-01, 7.40498308424117679E-01);
    phi_Grad[16][5].Set_Equal_To(-2.07694243849492910E+00, -2.96479439513799203E+00);
    phi_Grad[17][0].Set_Equal_To(8.52646351781054856E-01, 8.52646351781054856E-01);
    phi_Grad[17][1].Set_Equal_To(-1.12148043356937177E-01, 0.00000000000000000E+00);
    phi_Grad[17][2].Set_Equal_To(0.00000000000000000E+00, 1.96479439513799203E+00);
    phi_Grad[17][3].Set_Equal_To(2.96479439513799203E+00, 8.87851956643062823E-01);
    phi_Grad[17][4].Set_Equal_To(-2.96479439513799203E+00, -2.81744074691904700E+00);
    phi_Grad[17][5].Set_Equal_To(-7.40498308424117679E-01, -8.87851956643062823E-01);
    phi_Grad[18][0].Set_Equal_To(8.52646351781054856E-01, 8.52646351781054856E-01);
    phi_Grad[18][1].Set_Equal_To(1.96479439513799203E+00, 0.00000000000000000E+00);
    phi_Grad[18][2].Set_Equal_To(0.00000000000000000E+00, -1.12148043356937177E-01);
    phi_Grad[18][3].Set_Equal_To(8.87851956643062823E-01, 2.96479439513799203E+00);
    phi_Grad[18][4].Set_Equal_To(-8.87851956643062823E-01, -7.40498308424117679E-01);
    phi_Grad[18][5].Set_Equal_To(-2.81744074691904700E+00, -2.96479439513799203E+00);


// // set of quadrature points
// static const double Quad_Points[NQ][SUB_TD] = { \
//     {3.33333333333333315E-01, 3.33333333333333315E-01}, \
//     {4.89682519198737620E-01, 4.89682519198737620E-01}, \
//     {4.89682519198737620E-01, 2.06349616025247456E-02}, \
//     {2.06349616025247456E-02, 4.89682519198737620E-01}, \
//     {4.37089591492936635E-01, 4.37089591492936635E-01}, \
//     {4.37089591492936635E-01, 1.25820817014126729E-01}, \
//     {1.25820817014126729E-01, 4.37089591492936635E-01}, \
//     {1.88203535619032719E-01, 1.88203535619032719E-01}, \
//     {1.88203535619032719E-01, 6.23592928761934617E-01}, \
//     {6.23592928761934617E-01, 1.88203535619032719E-01}, \
//     {4.47295133944527121E-02, 4.47295133944527121E-02}, \
//     {4.47295133944527121E-02, 9.10540973211094617E-01}, \
//     {9.10540973211094617E-01, 4.47295133944527121E-02}, \
//     {3.68384120547362859E-02, 2.21962989160765706E-01}, \
//     {2.21962989160765706E-01, 3.68384120547362859E-02}, \
//     {3.68384120547362859E-02, 7.41198598784498008E-01}, \
//     {7.41198598784498008E-01, 3.68384120547362859E-02}, \
//     {2.21962989160765706E-01, 7.41198598784498008E-01}, \
//     {7.41198598784498008E-01, 2.21962989160765706E-01}  \
//     };

// // set of quadrature weights
// static const double Quad_Weights[NQ] = { \
//     4.85678981413994182E-02, \
//     1.56673501135695357E-02, \
//     1.56673501135695357E-02, \
//     1.56673501135695357E-02, \
//     3.89137705023871391E-02, \
//     3.89137705023871391E-02, \
//     3.89137705023871391E-02, \
//     3.98238694636051244E-02, \
//     3.98238694636051244E-02, \
//     3.98238694636051244E-02, \
//     1.27888378293490156E-02, \
//     1.27888378293490156E-02, \
//     1.27888378293490156E-02, \
//     2.16417696886446881E-02, \
//     2.16417696886446881E-02, \
//     2.16417696886446881E-02, \
//     2.16417696886446881E-02, \
//     2.16417696886446881E-02, \
//     2.16417696886446881E-02  \
//     };
/*------------   END: Auto Generate ------------*/
    // set basis function values to the correct mesh entity
    Func_f_Value = &Value_p1; // point to the correct mesh entity

/*------------ BEGIN: Auto Generate ------------*/
    /*** compute basis function quantities ***/
    // map gradient from local to global coordinates (indexing is in the C style)
    // loop through quad points
    for (int qp_i = 0; (qp_i < Num_QP); qp_i++)
        {
        // multiply local gradient by transpose of Inv_Grad(PHI) matrix
        for (int basis_i = 0; (basis_i < Num_Basis); basis_i++)
            {
            VEC_2x1 VL; // map local gradient to the tangent space
            Mat_Transpose_Vec(Mesh->Map_PHI_Inv_Metric[qp_i], phi_Grad[qp_i][basis_i], VL);
            Mat_Vec(Mesh->Map_PHI_Grad[qp_i], VL, Func_f_Grad[basis_i][qp_i]);
            }
        }
/*------------   END: Auto Generate ------------*/
}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* evaluate basis functions (no derivatives!) on the local reference element. */
void SpecificFUNC::Basis_Value_p1(SCALAR BF_V[NB][NQ])
{

/*------------ BEGIN: Auto Generate ------------*/
// Local Element defined on Subdomain: CG, lagrange_deg2_dim2
// the Subdomain             has topological dimension = 2
// the Domain of Integration has topological dimension = 2
// geometric dimension = 3
// Number of Quadrature Points = 19

    // get "Val" of basis functions
    SCALAR phi_Val[NQ][NB];

    phi_Val[0][0].Set_Equal_To(-1.11111111111111105E-01);
    phi_Val[0][1].Set_Equal_To(-1.11111111111111105E-01);
    phi_Val[0][2].Set_Equal_To(-1.11111111111111105E-01);
    phi_Val[0][3].Set_Equal_To(4.44444444444444420E-01);
    phi_Val[0][4].Set_Equal_To(4.44444444444444420E-01);
    phi_Val[0][5].Set_Equal_To(4.44444444444444420E-01);
    phi_Val[1][0].Set_Equal_To(-1.97833583218494161E-02);
    phi_Val[1][1].Set_Equal_To(-1.01045799810935439E-02);
    phi_Val[1][2].Set_Equal_To(-1.01045799810935439E-02);
    phi_Val[1][3].Set_Equal_To(9.59155878435288156E-01);
    phi_Val[1][4].Set_Equal_To(4.04183199243741756E-02);
    phi_Val[1][5].Set_Equal_To(4.04183199243741756E-02);
    phi_Val[2][0].Set_Equal_To(-1.01045799810935300E-02);
    phi_Val[2][1].Set_Equal_To(-1.01045799810935439E-02);
    phi_Val[2][2].Set_Equal_To(-1.97833583218494057E-02);
    phi_Val[2][3].Set_Equal_To(4.04183199243741478E-02);
    phi_Val[2][4].Set_Equal_To(4.04183199243741478E-02);
    phi_Val[2][5].Set_Equal_To(9.59155878435288156E-01);
    phi_Val[3][0].Set_Equal_To(-1.01045799810935300E-02);
    phi_Val[3][1].Set_Equal_To(-1.97833583218494057E-02);
    phi_Val[3][2].Set_Equal_To(-1.01045799810935439E-02);
    phi_Val[3][3].Set_Equal_To(4.04183199243741478E-02);
    phi_Val[3][4].Set_Equal_To(9.59155878435288156E-01);
    phi_Val[3][5].Set_Equal_To(4.04183199243741478E-02);
    phi_Val[4][0].Set_Equal_To(-9.41590610259220029E-02);
    phi_Val[4][1].Set_Equal_To(-5.49949695100121830E-02);
    phi_Val[4][2].Set_Equal_To(-5.49949695100121830E-02);
    phi_Val[4][3].Set_Equal_To(7.64189243965848863E-01);
    phi_Val[4][4].Set_Equal_To(2.19979878040048732E-01);
    phi_Val[4][5].Set_Equal_To(2.19979878040048732E-01);
    phi_Val[5][0].Set_Equal_To(-5.49949695100121830E-02);
    phi_Val[5][1].Set_Equal_To(-5.49949695100121830E-02);
    phi_Val[5][2].Set_Equal_To(-9.41590610259220029E-02);
    phi_Val[5][3].Set_Equal_To(2.19979878040048732E-01);
    phi_Val[5][4].Set_Equal_To(2.19979878040048732E-01);
    phi_Val[5][5].Set_Equal_To(7.64189243965848863E-01);
    phi_Val[6][0].Set_Equal_To(-5.49949695100121830E-02);
    phi_Val[6][1].Set_Equal_To(-9.41590610259220029E-02);
    phi_Val[6][2].Set_Equal_To(-5.49949695100121830E-02);
    phi_Val[6][3].Set_Equal_To(2.19979878040048732E-01);
    phi_Val[6][4].Set_Equal_To(7.64189243965848863E-01);
    phi_Val[6][5].Set_Equal_To(2.19979878040048732E-01);
    phi_Val[7][0].Set_Equal_To(1.54143352841839831E-01);
    phi_Val[7][1].Set_Equal_To(-1.17362393980023683E-01);
    phi_Val[7][2].Set_Equal_To(-1.17362393980023683E-01);
    phi_Val[7][3].Set_Equal_To(1.41682283278018073E-01);
    phi_Val[7][4].Set_Equal_To(4.69449575920094730E-01);
    phi_Val[7][5].Set_Equal_To(4.69449575920094730E-01);
    phi_Val[8][0].Set_Equal_To(-1.17362393980023669E-01);
    phi_Val[8][1].Set_Equal_To(-1.17362393980023683E-01);
    phi_Val[8][2].Set_Equal_To(1.54143352841839915E-01);
    phi_Val[8][3].Set_Equal_To(4.69449575920094786E-01);
    phi_Val[8][4].Set_Equal_To(4.69449575920094619E-01);
    phi_Val[8][5].Set_Equal_To(1.41682283278018017E-01);
    phi_Val[9][0].Set_Equal_To(-1.17362393980023669E-01);
    phi_Val[9][1].Set_Equal_To(1.54143352841839915E-01);
    phi_Val[9][2].Set_Equal_To(-1.17362393980023683E-01);
    phi_Val[9][3].Set_Equal_To(4.69449575920094786E-01);
    phi_Val[9][4].Set_Equal_To(1.41682283278018017E-01);
    phi_Val[9][5].Set_Equal_To(4.69449575920094619E-01);
    phi_Val[10][0].Set_Equal_To(7.47628754581319943E-01);
    phi_Val[10][1].Set_Equal_To(-4.07280546574436617E-02);
    phi_Val[10][2].Set_Equal_To(-4.07280546574436617E-02);
    phi_Val[10][3].Set_Equal_To(8.00291747401809909E-03);
    phi_Val[10][4].Set_Equal_To(1.62912218629774647E-01);
    phi_Val[10][5].Set_Equal_To(1.62912218629774647E-01);
    phi_Val[11][0].Set_Equal_To(-4.07280546574436270E-02);
    phi_Val[11][1].Set_Equal_To(-4.07280546574436617E-02);
    phi_Val[11][2].Set_Equal_To(7.47628754581320054E-01);
    phi_Val[11][3].Set_Equal_To(1.62912218629774647E-01);
    phi_Val[11][4].Set_Equal_To(1.62912218629774508E-01);
    phi_Val[11][5].Set_Equal_To(8.00291747401809042E-03);
    phi_Val[12][0].Set_Equal_To(-4.07280546574436270E-02);
    phi_Val[12][1].Set_Equal_To(7.47628754581320054E-01);
    phi_Val[12][2].Set_Equal_To(-4.07280546574436617E-02);
    phi_Val[12][3].Set_Equal_To(1.62912218629774647E-01);
    phi_Val[12][4].Set_Equal_To(8.00291747401809042E-03);
    phi_Val[12][5].Set_Equal_To(1.62912218629774508E-01);
    phi_Val[13][0].Set_Equal_To(3.57552126895708478E-01);
    phi_Val[13][1].Set_Equal_To(-3.41242748493072040E-02);
    phi_Val[13][2].Set_Equal_To(-1.23427852046401318E-01);
    phi_Val[13][3].Set_Equal_To(3.27070562224210035E-02);
    phi_Val[13][4].Set_Equal_To(6.58074626191913037E-01);
    phi_Val[13][5].Set_Equal_To(1.09218317585665983E-01);
    phi_Val[14][0].Set_Equal_To(3.57552126895708478E-01);
    phi_Val[14][1].Set_Equal_To(-1.23427852046401318E-01);
    phi_Val[14][2].Set_Equal_To(-3.41242748493072040E-02);
    phi_Val[14][3].Set_Equal_To(3.27070562224210035E-02);
    phi_Val[14][4].Set_Equal_To(1.09218317585665983E-01);
    phi_Val[14][5].Set_Equal_To(6.58074626191913037E-01);
    phi_Val[15][0].Set_Equal_To(-1.23427852046401318E-01);
    phi_Val[15][1].Set_Equal_To(-3.41242748493072040E-02);
    phi_Val[15][2].Set_Equal_To(3.57552126895708478E-01);
    phi_Val[15][3].Set_Equal_To(1.09218317585665983E-01);
    phi_Val[15][4].Set_Equal_To(6.58074626191913037E-01);
    phi_Val[15][5].Set_Equal_To(3.27070562224210035E-02);
    phi_Val[16][0].Set_Equal_To(-1.23427852046401318E-01);
    phi_Val[16][1].Set_Equal_To(3.57552126895708478E-01);
    phi_Val[16][2].Set_Equal_To(-3.41242748493072040E-02);
    phi_Val[16][3].Set_Equal_To(1.09218317585665983E-01);
    phi_Val[16][4].Set_Equal_To(3.27070562224210035E-02);
    phi_Val[16][5].Set_Equal_To(6.58074626191913037E-01);
    phi_Val[17][0].Set_Equal_To(-3.41242748493072040E-02);
    phi_Val[17][1].Set_Equal_To(-1.23427852046401318E-01);
    phi_Val[17][2].Set_Equal_To(3.57552126895708478E-01);
    phi_Val[17][3].Set_Equal_To(6.58074626191913037E-01);
    phi_Val[17][4].Set_Equal_To(1.09218317585665983E-01);
    phi_Val[17][5].Set_Equal_To(3.27070562224210035E-02);
    phi_Val[18][0].Set_Equal_To(-3.41242748493072040E-02);
    phi_Val[18][1].Set_Equal_To(3.57552126895708478E-01);
    phi_Val[18][2].Set_Equal_To(-1.23427852046401318E-01);
    phi_Val[18][3].Set_Equal_To(6.58074626191913037E-01);
    phi_Val[18][4].Set_Equal_To(3.27070562224210035E-02);
    phi_Val[18][5].Set_Equal_To(1.09218317585665983E-01);


// // set of quadrature points
// static const double Quad_Points[NQ][SUB_TD] = { \
//     {3.33333333333333315E-01, 3.33333333333333315E-01}, \
//     {4.89682519198737620E-01, 4.89682519198737620E-01}, \
//     {4.89682519198737620E-01, 2.06349616025247456E-02}, \
//     {2.06349616025247456E-02, 4.89682519198737620E-01}, \
//     {4.37089591492936635E-01, 4.37089591492936635E-01}, \
//     {4.37089591492936635E-01, 1.25820817014126729E-01}, \
//     {1.25820817014126729E-01, 4.37089591492936635E-01}, \
//     {1.88203535619032719E-01, 1.88203535619032719E-01}, \
//     {1.88203535619032719E-01, 6.23592928761934617E-01}, \
//     {6.23592928761934617E-01, 1.88203535619032719E-01}, \
//     {4.47295133944527121E-02, 4.47295133944527121E-02}, \
//     {4.47295133944527121E-02, 9.10540973211094617E-01}, \
//     {9.10540973211094617E-01, 4.47295133944527121E-02}, \
//     {3.68384120547362859E-02, 2.21962989160765706E-01}, \
//     {2.21962989160765706E-01, 3.68384120547362859E-02}, \
//     {3.68384120547362859E-02, 7.41198598784498008E-01}, \
//     {7.41198598784498008E-01, 3.68384120547362859E-02}, \
//     {2.21962989160765706E-01, 7.41198598784498008E-01}, \
//     {7.41198598784498008E-01, 2.21962989160765706E-01}  \
//     };

// // set of quadrature weights
// static const double Quad_Weights[NQ] = { \
//     4.85678981413994182E-02, \
//     1.56673501135695357E-02, \
//     1.56673501135695357E-02, \
//     1.56673501135695357E-02, \
//     3.89137705023871391E-02, \
//     3.89137705023871391E-02, \
//     3.89137705023871391E-02, \
//     3.98238694636051244E-02, \
//     3.98238694636051244E-02, \
//     3.98238694636051244E-02, \
//     1.27888378293490156E-02, \
//     1.27888378293490156E-02, \
//     1.27888378293490156E-02, \
//     2.16417696886446881E-02, \
//     2.16417696886446881E-02, \
//     2.16417696886446881E-02, \
//     2.16417696886446881E-02, \
//     2.16417696886446881E-02, \
//     2.16417696886446881E-02  \
//     };
/*------------   END: Auto Generate ------------*/
    // copy function evaluations over (indexing is in the C style)
    // loop through quad points
    for (int qp_i = 0; (qp_i < Num_QP); qp_i++)
        {
        // evaluate for each basis function
        for (int basis_i = 0; (basis_i < Num_Basis); basis_i++)
            {
            BF_V[basis_i][qp_i].a = phi_Val[qp_i][basis_i].a;
            }
        }
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
