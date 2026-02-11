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
#define SpecificFUNC        Data_Type_Sigma_Space_phi_restricted_to_Gamma
#define SpecificFUNC_str   "Data_Type_Sigma_Space_phi_restricted_to_Gamma"

// set the type of function space
#define SPACE_type  "CG - raviart_thomas_deg1_dim2"
// set the name of function space
#define SPACE_name  "Sigma_Space"

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
#define NQ  12
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
    // divergence of vector valued H(div) basis functions
    SCALAR Func_vv_Div[NB][NQ];

    // constructor
    SpecificFUNC ();
    ~SpecificFUNC (); // destructor
    void Setup_Function_Space(const mxArray*);
    void Get_Local_to_Global_DoFmap(const int&, int*) const;
                   // need the "const" to ENSURE that nothing in this object will change!
    void Transform_Basis_Functions();
    const CLASS_geom_Gamma_embedded_in_Gamma_restricted_to_Gamma*  Mesh;

private:
    void Get_Basis_Sign_Change(const double Mesh_Orient[SUB_TD+1], double Basis_Sign[NB]) const;
    void Map_Basis_p1();
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
    for (int qp_i = 0; (qp_i < Num_QP); qp_i++)
    for (int basis_i = 0; (basis_i < Num_Basis); basis_i++)
        Func_vv_Div[basis_i][qp_i].Set_To_Zero();
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
    /* call sub-routine to determine sign flips for H(div) basis functions */
    double Basis_Sign[NB] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // init to all positive (no sign change)
    Get_Basis_Sign_Change(Mesh->Orientation, Basis_Sign);
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
// Local Element defined on Subdomain: CG, raviart_thomas_deg1_dim2
// the Subdomain             has topological dimension = 2
// the Domain of Integration has topological dimension = 2
// geometric dimension = 3
// Number of Quadrature Points = 12

    // get "Val" of basis functions
    VEC_2x1 phi_Val[NQ][NB];
    // get "Div" of basis functions
    SCALAR phi_Div[NQ][NB];

    phi_Val[0][0].Set_Equal_To(2.61323081218823816E+00, 3.14850506956868836E-01);
    phi_Val[0][1].Set_Equal_To(-1.30661540609411908E+00, -2.20514267969936423E-01);
    phi_Val[0][2].Set_Equal_To(3.14850506956868836E-01, -2.20514267969936423E-01);
    phi_Val[0][3].Set_Equal_To(3.14850506956868836E-01, -9.43362389869324136E-02);
    phi_Val[0][4].Set_Equal_To(-1.30661540609411908E+00, 1.52712967406405564E+00);
    phi_Val[0][5].Set_Equal_To(2.61323081218823816E+00, -2.92808131914510694E+00);
    phi_Val[0][6].Set_Equal_To(1.32308560781961848E+00, -4.09186745943801222E-01);
    phi_Val[0][7].Set_Equal_To(0.00000000000000000E+00, 5.04712115932016037E-01);
    phi_Val[1][0].Set_Equal_To(-2.20514267969936423E-01, -1.30661540609411908E+00);
    phi_Val[1][1].Set_Equal_To(3.14850506956868836E-01, 2.61323081218823816E+00);
    phi_Val[1][2].Set_Equal_To(-2.92808131914510694E+00, 2.61323081218823816E+00);
    phi_Val[1][3].Set_Equal_To(1.52712967406405564E+00, -1.30661540609411908E+00);
    phi_Val[1][4].Set_Equal_To(-9.43362389869324136E-02, 3.14850506956868836E-01);
    phi_Val[1][5].Set_Equal_To(-2.20514267969936423E-01, 3.14850506956868836E-01);
    phi_Val[1][6].Set_Equal_To(5.04712115932016037E-01, 0.00000000000000000E+00);
    phi_Val[1][7].Set_Equal_To(-4.09186745943801222E-01, 1.32308560781961848E+00);
    phi_Val[2][0].Set_Equal_To(-2.20514267969936423E-01, -9.43362389869324136E-02);
    phi_Val[2][1].Set_Equal_To(-9.43362389869324136E-02, -2.20514267969936423E-01);
    phi_Val[2][2].Set_Equal_To(1.52712967406405564E+00, -2.20514267969936423E-01);
    phi_Val[2][3].Set_Equal_To(-2.92808131914510694E+00, 3.14850506956868836E-01);
    phi_Val[2][4].Set_Equal_To(3.14850506956868836E-01, -2.92808131914510694E+00);
    phi_Val[2][5].Set_Equal_To(-2.20514267969936423E-01, 1.52712967406405564E+00);
    phi_Val[2][6].Set_Equal_To(9.13898861875817259E-01, 4.09186745943801222E-01);
    phi_Val[2][7].Set_Equal_To(4.09186745943801222E-01, 9.13898861875817259E-01);
    phi_Val[3][0].Set_Equal_To(5.72231807115515089E-03, 5.01418369938958608E-01);
    phi_Val[3][1].Set_Equal_To(-2.86115903557946803E-03, -4.99995930140390232E-01);
    phi_Val[3][2].Set_Equal_To(5.01418369938960495E-01, -4.99995930140390232E-01);
    phi_Val[3][3].Set_Equal_To(5.01418369938958608E-01, -1.42243979856833164E-03);
    phi_Val[3][4].Set_Equal_To(-2.86115903557568243E-03, 5.02857089175965943E-01);
    phi_Val[3][5].Set_Equal_To(5.72231807115515089E-03, -5.07140688010115581E-01);
    phi_Val[3][6].Set_Equal_To(2.99997558084234317E+00, -5.02840809737526873E-01);
    phi_Val[3][7].Set_Equal_To(3.78552774202358384E-15, 1.99429396136728188E+00);
    phi_Val[4][0].Set_Equal_To(-4.99995930140390232E-01, -2.86115903557946803E-03);
    phi_Val[4][1].Set_Equal_To(5.01418369938958608E-01, 5.72231807115515089E-03);
    phi_Val[4][2].Set_Equal_To(-5.07140688010115581E-01, 5.72231807115515089E-03);
    phi_Val[4][3].Set_Equal_To(5.02857089175965943E-01, -2.86115903557568243E-03);
    phi_Val[4][4].Set_Equal_To(-1.42243979856833164E-03, 5.01418369938958608E-01);
    phi_Val[4][5].Set_Equal_To(-4.99995930140390232E-01, 5.01418369938960495E-01);
    phi_Val[4][6].Set_Equal_To(1.99429396136728188E+00, 3.78552774202358384E-15);
    phi_Val[4][7].Set_Equal_To(-5.02840809737526873E-01, 2.99997558084234317E+00);
    phi_Val[5][0].Set_Equal_To(-4.99995930140390232E-01, -1.42243979857021381E-03);
    phi_Val[5][1].Set_Equal_To(-1.42243979857021381E-03, -4.99995930140390232E-01);
    phi_Val[5][2].Set_Equal_To(5.02857089175969718E-01, -4.99995930140390232E-01);
    phi_Val[5][3].Set_Equal_To(-5.07140688010119356E-01, 5.01418369938960495E-01);
    phi_Val[5][4].Set_Equal_To(5.01418369938960495E-01, -5.07140688010119356E-01);
    phi_Val[5][5].Set_Equal_To(-4.99995930140390232E-01, 5.02857089175969718E-01);
    phi_Val[5][6].Set_Equal_To(2.49713477110481064E+00, 5.02840809737530647E-01);
    phi_Val[5][7].Set_Equal_To(5.02840809737530647E-01, 2.49713477110481064E+00);
    phi_Val[6][0].Set_Equal_To(6.95073454616696340E-01, 9.59615983464076172E-01);
    phi_Val[6][1].Set_Equal_To(3.07315887288848122E-01, -4.70860653233716675E-01);
    phi_Val[6][2].Set_Equal_To(4.45201181086138098E-01, -4.70860653233716675E-01);
    phi_Val[6][3].Set_Equal_To(6.78740359025559625E-01, -4.88755330230359442E-01);
    phi_Val[6][4].Set_Equal_To(-1.00238934190554452E+00, 1.19237437070074459E+00);
    phi_Val[6][5].Set_Equal_To(6.95073454616696340E-01, -8.59399011264317836E-01);
    phi_Val[6][6].Set_Equal_To(2.12155219820615315E+00, -1.44837131369443561E+00);
    phi_Val[6][7].Set_Equal_To(-1.30970522919439247E+00, 1.84422002920606731E+00);
    phi_Val[7][0].Set_Equal_To(6.95073454616696340E-01, 1.64325556647621634E-01);
    phi_Val[7][1].Set_Equal_To(-1.00238934190554430E+00, -1.89985028795200184E-01);
    phi_Val[7][2].Set_Equal_To(6.78740359025559625E-01, -1.89985028795200184E-01);
    phi_Val[7][3].Set_Equal_To(4.45201181086138098E-01, 2.56594721475785666E-02);
    phi_Val[7][4].Set_Equal_To(3.07315887288848066E-01, 1.63544765944868581E-01);
    phi_Val[7][5].Set_Equal_To(6.95073454616696340E-01, -1.65468943808077240E+00);
    phi_Val[7][6].Set_Equal_To(3.43125742740054562E+00, -1.38666084500043058E-01);
    phi_Val[7][7].Set_Equal_To(1.30970522919439247E+00, 5.34514800011674729E-01);
    phi_Val[8][0].Set_Equal_To(-4.70860653233716675E-01, 3.07315887288848122E-01);
    phi_Val[8][1].Set_Equal_To(9.59615983464076172E-01, 6.95073454616696340E-01);
    phi_Val[8][2].Set_Equal_To(-8.59399011264317836E-01, 6.95073454616696340E-01);
    phi_Val[8][3].Set_Equal_To(1.19237437070074459E+00, -1.00238934190554452E+00);
    phi_Val[8][4].Set_Equal_To(-4.88755330230359442E-01, 6.78740359025559625E-01);
    phi_Val[8][5].Set_Equal_To(-4.70860653233716675E-01, 4.45201181086138098E-01);
    phi_Val[8][6].Set_Equal_To(1.84422002920606731E+00, -1.30970522919439247E+00);
    phi_Val[8][7].Set_Equal_To(-1.44837131369443561E+00, 2.12155219820615315E+00);
    phi_Val[9][0].Set_Equal_To(-4.70860653233716675E-01, 2.56594721475785700E-02);
    phi_Val[9][1].Set_Equal_To(-4.88755330230359442E-01, -1.89985028795200184E-01);
    phi_Val[9][2].Set_Equal_To(1.19237437070074459E+00, -1.89985028795200184E-01);
    phi_Val[9][3].Set_Equal_To(-8.59399011264317836E-01, 1.64325556647621607E-01);
    phi_Val[9][4].Set_Equal_To(9.59615983464076061E-01, -1.65468943808077240E+00);
    phi_Val[9][5].Set_Equal_To(-4.70860653233716675E-01, 1.63544765944868553E-01);
    phi_Val[9][6].Set_Equal_To(3.29259134290050293E+00, 1.38666084500043058E-01);
    phi_Val[9][7].Set_Equal_To(1.44837131369443561E+00, 6.73180884511717870E-01);
    phi_Val[10][0].Set_Equal_To(-1.89985028795200184E-01, -1.00238934190554430E+00);
    phi_Val[10][1].Set_Equal_To(1.64325556647621634E-01, 6.95073454616696340E-01);
    phi_Val[10][2].Set_Equal_To(-1.65468943808077240E+00, 6.95073454616696340E-01);
    phi_Val[10][3].Set_Equal_To(1.63544765944868581E-01, 3.07315887288848066E-01);
    phi_Val[10][4].Set_Equal_To(2.56594721475785666E-02, 4.45201181086138098E-01);
    phi_Val[10][5].Set_Equal_To(-1.89985028795200184E-01, 6.78740359025559625E-01);
    phi_Val[10][6].Set_Equal_To(5.34514800011674729E-01, 1.30970522919439247E+00);
    phi_Val[10][7].Set_Equal_To(-1.38666084500043058E-01, 3.43125742740054562E+00);
    phi_Val[11][0].Set_Equal_To(-1.89985028795200184E-01, -4.88755330230359442E-01);
    phi_Val[11][1].Set_Equal_To(2.56594721475785700E-02, -4.70860653233716675E-01);
    phi_Val[11][2].Set_Equal_To(1.63544765944868553E-01, -4.70860653233716675E-01);
    phi_Val[11][3].Set_Equal_To(-1.65468943808077240E+00, 9.59615983464076061E-01);
    phi_Val[11][4].Set_Equal_To(1.64325556647621607E-01, -8.59399011264317836E-01);
    phi_Val[11][5].Set_Equal_To(-1.89985028795200184E-01, 1.19237437070074459E+00);
    phi_Val[11][6].Set_Equal_To(6.73180884511717870E-01, 1.44837131369443561E+00);
    phi_Val[11][7].Set_Equal_To(1.38666084500043058E-01, 3.29259134290050293E+00);

    phi_Div[0][0].Set_Equal_To(1.49717273044079047E+01);
    phi_Div[0][1].Set_Equal_To(-4.48586365220395145E+00);
    phi_Div[0][2].Set_Equal_To(-4.48586365220395145E+00);
    phi_Div[0][3].Set_Equal_To(-4.48586365220395145E+00);
    phi_Div[0][4].Set_Equal_To(-4.48586365220395145E+00);
    phi_Div[0][5].Set_Equal_To(1.49717273044079047E+01);
    phi_Div[0][6].Set_Equal_To(-1.94575909566118561E+01);
    phi_Div[0][7].Set_Equal_To(0.00000000000000000E+00);
    phi_Div[1][0].Set_Equal_To(-4.48586365220395145E+00);
    phi_Div[1][1].Set_Equal_To(1.49717273044079047E+01);
    phi_Div[1][2].Set_Equal_To(1.49717273044079047E+01);
    phi_Div[1][3].Set_Equal_To(-4.48586365220395145E+00);
    phi_Div[1][4].Set_Equal_To(-4.48586365220395145E+00);
    phi_Div[1][5].Set_Equal_To(-4.48586365220395145E+00);
    phi_Div[1][6].Set_Equal_To(0.00000000000000000E+00);
    phi_Div[1][7].Set_Equal_To(-1.94575909566118561E+01);
    phi_Div[2][0].Set_Equal_To(-4.48586365220395145E+00);
    phi_Div[2][1].Set_Equal_To(-4.48586365220395145E+00);
    phi_Div[2][2].Set_Equal_To(-4.48586365220395145E+00);
    phi_Div[2][3].Set_Equal_To(1.49717273044079047E+01);
    phi_Div[2][4].Set_Equal_To(1.49717273044079047E+01);
    phi_Div[2][5].Set_Equal_To(-4.48586365220395145E+00);
    phi_Div[2][6].Set_Equal_To(1.94575909566118561E+01);
    phi_Div[2][7].Set_Equal_To(1.94575909566118561E+01);
    phi_Div[3][0].Set_Equal_To(6.03423623179629676E+00);
    phi_Div[3][1].Set_Equal_To(-1.71181158981597026E-02);
    phi_Div[3][2].Set_Equal_To(-1.71181158981597026E-02);
    phi_Div[3][3].Set_Equal_To(-1.71181158981370540E-02);
    phi_Div[3][4].Set_Equal_To(-1.71181158981370540E-02);
    phi_Div[3][5].Set_Equal_To(6.03423623179629676E+00);
    phi_Div[3][6].Set_Equal_To(-6.05135434769443403E+00);
    phi_Div[3][7].Set_Equal_To(2.26485497023531934E-14);
    phi_Div[4][0].Set_Equal_To(-1.71181158981597026E-02);
    phi_Div[4][1].Set_Equal_To(6.03423623179629676E+00);
    phi_Div[4][2].Set_Equal_To(6.03423623179629676E+00);
    phi_Div[4][3].Set_Equal_To(-1.71181158981370540E-02);
    phi_Div[4][4].Set_Equal_To(-1.71181158981370540E-02);
    phi_Div[4][5].Set_Equal_To(-1.71181158981597026E-02);
    phi_Div[4][6].Set_Equal_To(2.26485497023531934E-14);
    phi_Div[4][7].Set_Equal_To(-6.05135434769443403E+00);
    phi_Div[5][0].Set_Equal_To(-1.71181158981597026E-02);
    phi_Div[5][1].Set_Equal_To(-1.71181158981597026E-02);
    phi_Div[5][2].Set_Equal_To(-1.71181158981597026E-02);
    phi_Div[5][3].Set_Equal_To(6.03423623179631896E+00);
    phi_Div[5][4].Set_Equal_To(6.03423623179631896E+00);
    phi_Div[5][5].Set_Equal_To(-1.71181158981597026E-02);
    phi_Div[5][6].Set_Equal_To(6.05135434769447933E+00);
    phi_Div[5][7].Set_Equal_To(6.05135434769447933E+00);
    phi_Div[6][0].Set_Equal_To(9.27605997891357603E+00);
    phi_Div[6][1].Set_Equal_To(1.44845882481084010E+00);
    phi_Div[6][2].Set_Equal_To(1.44845882481084010E+00);
    phi_Div[6][3].Set_Equal_To(-4.72451880372441568E+00);
    phi_Div[6][4].Set_Equal_To(-4.72451880372441568E+00);
    phi_Div[6][5].Set_Equal_To(9.27605997891357603E+00);
    phi_Div[6][6].Set_Equal_To(-1.40005787826379926E+01);
    phi_Div[6][7].Set_Equal_To(-6.17297762853525622E+00);
    phi_Div[7][0].Set_Equal_To(9.27605997891357603E+00);
    phi_Div[7][1].Set_Equal_To(-4.72451880372441568E+00);
    phi_Div[7][2].Set_Equal_To(-4.72451880372441568E+00);
    phi_Div[7][3].Set_Equal_To(1.44845882481083987E+00);
    phi_Div[7][4].Set_Equal_To(1.44845882481083987E+00);
    phi_Div[7][5].Set_Equal_To(9.27605997891357603E+00);
    phi_Div[7][6].Set_Equal_To(-7.82760115410273638E+00);
    phi_Div[7][7].Set_Equal_To(6.17297762853525622E+00);
    phi_Div[8][0].Set_Equal_To(1.44845882481084010E+00);
    phi_Div[8][1].Set_Equal_To(9.27605997891357603E+00);
    phi_Div[8][2].Set_Equal_To(9.27605997891357603E+00);
    phi_Div[8][3].Set_Equal_To(-4.72451880372441568E+00);
    phi_Div[8][4].Set_Equal_To(-4.72451880372441568E+00);
    phi_Div[8][5].Set_Equal_To(1.44845882481084010E+00);
    phi_Div[8][6].Set_Equal_To(-6.17297762853525622E+00);
    phi_Div[8][7].Set_Equal_To(-1.40005787826379926E+01);
    phi_Div[9][0].Set_Equal_To(1.44845882481084010E+00);
    phi_Div[9][1].Set_Equal_To(-4.72451880372441568E+00);
    phi_Div[9][2].Set_Equal_To(-4.72451880372441568E+00);
    phi_Div[9][3].Set_Equal_To(9.27605997891357603E+00);
    phi_Div[9][4].Set_Equal_To(9.27605997891357603E+00);
    phi_Div[9][5].Set_Equal_To(1.44845882481084010E+00);
    phi_Div[9][6].Set_Equal_To(7.82760115410273549E+00);
    phi_Div[9][7].Set_Equal_To(1.40005787826379926E+01);
    phi_Div[10][0].Set_Equal_To(-4.72451880372441568E+00);
    phi_Div[10][1].Set_Equal_To(9.27605997891357603E+00);
    phi_Div[10][2].Set_Equal_To(9.27605997891357603E+00);
    phi_Div[10][3].Set_Equal_To(1.44845882481083987E+00);
    phi_Div[10][4].Set_Equal_To(1.44845882481083987E+00);
    phi_Div[10][5].Set_Equal_To(-4.72451880372441568E+00);
    phi_Div[10][6].Set_Equal_To(6.17297762853525622E+00);
    phi_Div[10][7].Set_Equal_To(-7.82760115410273638E+00);
    phi_Div[11][0].Set_Equal_To(-4.72451880372441568E+00);
    phi_Div[11][1].Set_Equal_To(1.44845882481084010E+00);
    phi_Div[11][2].Set_Equal_To(1.44845882481084010E+00);
    phi_Div[11][3].Set_Equal_To(9.27605997891357603E+00);
    phi_Div[11][4].Set_Equal_To(9.27605997891357603E+00);
    phi_Div[11][5].Set_Equal_To(-4.72451880372441568E+00);
    phi_Div[11][6].Set_Equal_To(1.40005787826379926E+01);
    phi_Div[11][7].Set_Equal_To(7.82760115410273549E+00);


// // set of quadrature points
// static const double Quad_Points[NQ][SUB_TD] = { \
//     {8.73821971016995991E-01, 6.30890144915020046E-02}, \
//     {6.30890144915020046E-02, 8.73821971016995991E-01}, \
//     {6.30890144915020046E-02, 6.30890144915020046E-02}, \
//     {5.01426509658179032E-01, 2.49286745170910012E-01}, \
//     {2.49286745170910012E-01, 5.01426509658179032E-01}, \
//     {2.49286745170910012E-01, 2.49286745170910012E-01}, \
//     {6.36502499121399001E-01, 3.10352451033785004E-01}, \
//     {6.36502499121399001E-01, 5.31450498448160016E-02}, \
//     {3.10352451033785004E-01, 6.36502499121399001E-01}, \
//     {3.10352451033785004E-01, 5.31450498448160016E-02}, \
//     {5.31450498448160016E-02, 6.36502499121399001E-01}, \
//     {5.31450498448160016E-02, 3.10352451033785004E-01}  \
//     };

// // set of quadrature weights
// static const double Quad_Weights[NQ] = { \
//     2.54224531851034996E-02, \
//     2.54224531851034996E-02, \
//     2.54224531851034996E-02, \
//     5.83931378631894968E-02, \
//     5.83931378631894968E-02, \
//     5.83931378631894968E-02, \
//     4.14255378091870005E-02, \
//     4.14255378091870005E-02, \
//     4.14255378091870005E-02, \
//     4.14255378091870005E-02, \
//     4.14255378091870005E-02, \
//     4.14255378091870005E-02  \
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
    // map divergence of basis vectors over (indexing is in the C style)
    // loop through quad points
    for (int qp_i = 0; (qp_i < Num_QP); qp_i++)
        {
        // evaluate for each basis function
        for (int basis_i = 0; (basis_i < Num_Basis); basis_i++)
            {
            // pre-multiply by 1/det(Jac) (and flip sign if necessary)
            Func_vv_Div[basis_i][qp_i].a = Basis_Sign[basis_i] * (phi_Div[qp_i][basis_i].a) * Mesh->Map_Inv_Det_Jac[qp_i].a;
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
