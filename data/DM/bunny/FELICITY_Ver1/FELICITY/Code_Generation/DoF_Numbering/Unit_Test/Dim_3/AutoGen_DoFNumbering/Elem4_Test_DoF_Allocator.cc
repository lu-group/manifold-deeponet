/*
============================================================================================
   This class points to outside MATLAB data which contains info about the
   finite element data.  It will also create a DoF numbering for that element.
   
   NOTE: portions of this code are automatically generated!
   
   Copyright (c) 10-16-2016,  Shawn W. Walker
============================================================================================
*/

/*------------ BEGIN: Auto Generate ------------*/
// set the type of element
#define ELEM_Name        "Elem4_Test"
// set allocator class name
#define EDA               Elem4_Test_DoF_Allocator
// set nodal topology data type name
#define NODAL_TOPOLOGY    Elem4_Test_nodal_top
// set permutation map data type names
#define SINGLE_EDGE_PERM_MAP    Elem4_Test_single_edge_perm_map
#define EDGE_DOF_PERMUTATION    Elem4_Test_edge_dof_perm
#define SINGLE_FACE_PERM_MAP    Elem4_Test_single_face_perm_map
#define FACE_DOF_PERMUTATION    Elem4_Test_face_dof_perm

// set the number of DoF sets
#define Num_Vtx_Sets        0
#define Num_Edge_Sets       0
#define Num_Face_Sets       2
#define Num_Tet_Sets        0

// set the max number (over all sets) of DoFs per entity per set
#define Max_DoF_Per_Vtx     0
#define Max_DoF_Per_Edge    0
#define Max_DoF_Per_Face    3
#define Max_DoF_Per_Tet     0

// set the total number of DoFs per entity
#define Num_DoF_Per_Vtx     0
#define Num_DoF_Per_Edge    0
#define Num_DoF_Per_Face    4
#define Num_DoF_Per_Tet     0

// set the TOTAL number of DoFs per cell (element)
#define Total_DoF_Per_Cell  16
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
// set (intrinsic) dimension and domain
#define Dimension    3
#define Domain_str  "tetrahedron"

// set the number of vertices
#define NUM_VTX    4
// set the number of edges
#define NUM_EDGE   6
// set the number of faces
#define NUM_FACE   4
// set the number of tetrahedrons
#define NUM_TET    1
/*------------   END: Auto Generate ------------*/

/***************************************************************************************/
// data structure containing information on the local element DoF numbering
typedef struct
{
    int  V[Num_Vtx_Sets +1][NUM_VTX +1][Max_DoF_Per_Vtx +1];    // nodes associated with each vertex
    int  E[Num_Edge_Sets+1][NUM_EDGE+1][Max_DoF_Per_Edge+1];    // nodes associated with each edge
    int  F[Num_Face_Sets+1][NUM_FACE+1][Max_DoF_Per_Face+1];    // nodes associated with each face
    int  T[Num_Tet_Sets +1][NUM_TET +1][Max_DoF_Per_Tet +1];    // nodes associated with each tetrahedron
}
NODAL_TOPOLOGY;

/***************************************************************************************/
// data structure containing *one* permutation map
typedef struct
{
    int map[Max_DoF_Per_Edge+1];
}
SINGLE_EDGE_PERM_MAP;

/***************************************************************************************/
// data structure for containing permutation maps of local edge DoFs from
//      one local edge to another.
struct EDGE_DOF_PERMUTATION
{
    EDGE_DOF_PERMUTATION () {} // default constructor

    void Setup_Maps ()
        {
        // init to all zero
        for (int di = 0; (di < Num_Edge_Sets); di++)
        for (int ii = 0; (ii < NUM_EDGE); ii++)
        for (int si = 0; (si < 2); si++)
        for (int jj = 0; (jj < NUM_EDGE); jj++)
        for (int sj = 0; (sj < 2); sj++)
        for (int kk = 0; (kk <= Max_DoF_Per_Edge); kk++)
            perm[di][ii][si][jj][sj].map[kk] = 0;

        // define all edge DoF permutation maps
        // map goes from "current" to "init" local edge
        /*------------ BEGIN: Auto Generate ------------*/
        /*------------   END: Auto Generate ------------*/
        }

    inline const int* Get_Map (const int& dof_set, const int& new_edge, const int& init_edge) const
        {
        // error check
        if ((new_edge==0) || (new_edge < -6) || (new_edge > 6))
            {
            mexPrintf("new_edge must be = +/- 1,2,3,4,5,6.\n");
            mexErrMsgTxt("Invalid local edge index!\n");
            }
        if ((init_edge==0) || (init_edge < -6) || (init_edge > 6))
            {
            mexPrintf("init_edge must be = +/- 1,2,3,4,5,6.\n");
            mexErrMsgTxt("Invalid local edge index!\n");
            }
        if ((dof_set < 1) || (dof_set > Num_Edge_Sets))
            {
            mexPrintf("dof_set must be >= 1 and <= %d.\n",Num_Edge_Sets);
            mexErrMsgTxt("Invalid dof_set index!\n");
            }

        // massage input
        const int ne = abs_index( new_edge) - 1; // C-style indexing
        const int ie = abs_index(init_edge) - 1; // C-style indexing
        int sign_ne = 1; // init to positive
        int sign_ie = 1; // init to positive
        if ( new_edge < 0) sign_ne = 0; // set to negative sign
        if (init_edge < 0) sign_ie = 0; // set to negative sign

        return perm[dof_set-1][ne][sign_ne][ie][sign_ie].map;
        }

private:
    // contains all of the permutation maps
    // input: [dof_set][new_edge_index][sign of new_edge][init_edge_index][sign of init_edge]
    // NOTE: this is not used b/c there are no edge sets!
    SINGLE_EDGE_PERM_MAP perm[1][NUM_EDGE][2][NUM_EDGE][2];
};

/***************************************************************************************/
// data structure containing *one* face-DoF permutation map
typedef struct
{
    int map[Max_DoF_Per_Face+1];
}
SINGLE_FACE_PERM_MAP;

/***************************************************************************************/
// data structure for containing permutation maps of local face DoFs from
//      one local face to another.
struct FACE_DOF_PERMUTATION
{
    FACE_DOF_PERMUTATION () {} // default constructor

    void Setup_Maps ()
        {
        // init to all zero
        for (int di = 0; (di < Num_Face_Sets); di++)
        for (int ii = 0; (ii < NUM_FACE); ii++)
        for (int si = 0; (si < 6); si++)
        for (int jj = 0; (jj < NUM_FACE); jj++)
        for (int sj = 0; (sj < 6); sj++)
        for (int kk = 0; (kk <= Max_DoF_Per_Face); kk++)
            perm[di][ii][si][jj][sj].map[kk] = 0;

        // define all face DoF permutation maps
        // map goes from "current" to "init" local face
        /*------------ BEGIN: Auto Generate ------------*/
        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 1, [1  2  3]
        perm[0][0][0][0][0].map[1] = 1;
        perm[0][0][0][0][0].map[2] = 2;
        perm[0][0][0][0][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 1, [3  1  2]
        perm[0][0][0][0][1].map[1] = 3;
        perm[0][0][0][0][1].map[2] = 1;
        perm[0][0][0][0][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 1, [2  3  1]
        perm[0][0][0][0][2].map[1] = 2;
        perm[0][0][0][0][2].map[2] = 3;
        perm[0][0][0][0][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 1, [3  2  1]
        perm[0][0][0][0][3].map[1] = 3;
        perm[0][0][0][0][3].map[2] = 2;
        perm[0][0][0][0][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 1, [2  1  3]
        perm[0][0][0][0][4].map[1] = 2;
        perm[0][0][0][0][4].map[2] = 1;
        perm[0][0][0][0][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 1, [1  3  2]
        perm[0][0][0][0][5].map[1] = 1;
        perm[0][0][0][0][5].map[2] = 3;
        perm[0][0][0][0][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 2, [1  2  3]
        perm[0][0][0][1][0].map[1] = 1;
        perm[0][0][0][1][0].map[2] = 2;
        perm[0][0][0][1][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 2, [3  1  2]
        perm[0][0][0][1][1].map[1] = 3;
        perm[0][0][0][1][1].map[2] = 1;
        perm[0][0][0][1][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 2, [2  3  1]
        perm[0][0][0][1][2].map[1] = 2;
        perm[0][0][0][1][2].map[2] = 3;
        perm[0][0][0][1][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 2, [3  2  1]
        perm[0][0][0][1][3].map[1] = 3;
        perm[0][0][0][1][3].map[2] = 2;
        perm[0][0][0][1][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 2, [2  1  3]
        perm[0][0][0][1][4].map[1] = 2;
        perm[0][0][0][1][4].map[2] = 1;
        perm[0][0][0][1][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 2, [1  3  2]
        perm[0][0][0][1][5].map[1] = 1;
        perm[0][0][0][1][5].map[2] = 3;
        perm[0][0][0][1][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 3, [1  2  3]
        perm[0][0][0][2][0].map[1] = 1;
        perm[0][0][0][2][0].map[2] = 2;
        perm[0][0][0][2][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 3, [3  1  2]
        perm[0][0][0][2][1].map[1] = 3;
        perm[0][0][0][2][1].map[2] = 1;
        perm[0][0][0][2][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 3, [2  3  1]
        perm[0][0][0][2][2].map[1] = 2;
        perm[0][0][0][2][2].map[2] = 3;
        perm[0][0][0][2][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 3, [3  2  1]
        perm[0][0][0][2][3].map[1] = 3;
        perm[0][0][0][2][3].map[2] = 2;
        perm[0][0][0][2][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 3, [2  1  3]
        perm[0][0][0][2][4].map[1] = 2;
        perm[0][0][0][2][4].map[2] = 1;
        perm[0][0][0][2][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 3, [1  3  2]
        perm[0][0][0][2][5].map[1] = 1;
        perm[0][0][0][2][5].map[2] = 3;
        perm[0][0][0][2][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 4, [1  2  3]
        perm[0][0][0][3][0].map[1] = 1;
        perm[0][0][0][3][0].map[2] = 2;
        perm[0][0][0][3][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 4, [3  1  2]
        perm[0][0][0][3][1].map[1] = 3;
        perm[0][0][0][3][1].map[2] = 1;
        perm[0][0][0][3][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 4, [2  3  1]
        perm[0][0][0][3][2].map[1] = 2;
        perm[0][0][0][3][2].map[2] = 3;
        perm[0][0][0][3][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 4, [3  2  1]
        perm[0][0][0][3][3].map[1] = 3;
        perm[0][0][0][3][3].map[2] = 2;
        perm[0][0][0][3][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 4, [2  1  3]
        perm[0][0][0][3][4].map[1] = 2;
        perm[0][0][0][3][4].map[2] = 1;
        perm[0][0][0][3][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 4, [1  3  2]
        perm[0][0][0][3][5].map[1] = 1;
        perm[0][0][0][3][5].map[2] = 3;
        perm[0][0][0][3][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 1, [1  2  3]
        perm[0][0][1][0][0].map[1] = 2;
        perm[0][0][1][0][0].map[2] = 3;
        perm[0][0][1][0][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 1, [3  1  2]
        perm[0][0][1][0][1].map[1] = 1;
        perm[0][0][1][0][1].map[2] = 2;
        perm[0][0][1][0][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 1, [2  3  1]
        perm[0][0][1][0][2].map[1] = 3;
        perm[0][0][1][0][2].map[2] = 1;
        perm[0][0][1][0][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 1, [3  2  1]
        perm[0][0][1][0][3].map[1] = 2;
        perm[0][0][1][0][3].map[2] = 1;
        perm[0][0][1][0][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 1, [2  1  3]
        perm[0][0][1][0][4].map[1] = 1;
        perm[0][0][1][0][4].map[2] = 3;
        perm[0][0][1][0][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 1, [1  3  2]
        perm[0][0][1][0][5].map[1] = 3;
        perm[0][0][1][0][5].map[2] = 2;
        perm[0][0][1][0][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 2, [1  2  3]
        perm[0][0][1][1][0].map[1] = 2;
        perm[0][0][1][1][0].map[2] = 3;
        perm[0][0][1][1][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 2, [3  1  2]
        perm[0][0][1][1][1].map[1] = 1;
        perm[0][0][1][1][1].map[2] = 2;
        perm[0][0][1][1][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 2, [2  3  1]
        perm[0][0][1][1][2].map[1] = 3;
        perm[0][0][1][1][2].map[2] = 1;
        perm[0][0][1][1][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 2, [3  2  1]
        perm[0][0][1][1][3].map[1] = 2;
        perm[0][0][1][1][3].map[2] = 1;
        perm[0][0][1][1][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 2, [2  1  3]
        perm[0][0][1][1][4].map[1] = 1;
        perm[0][0][1][1][4].map[2] = 3;
        perm[0][0][1][1][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 2, [1  3  2]
        perm[0][0][1][1][5].map[1] = 3;
        perm[0][0][1][1][5].map[2] = 2;
        perm[0][0][1][1][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 3, [1  2  3]
        perm[0][0][1][2][0].map[1] = 2;
        perm[0][0][1][2][0].map[2] = 3;
        perm[0][0][1][2][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 3, [3  1  2]
        perm[0][0][1][2][1].map[1] = 1;
        perm[0][0][1][2][1].map[2] = 2;
        perm[0][0][1][2][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 3, [2  3  1]
        perm[0][0][1][2][2].map[1] = 3;
        perm[0][0][1][2][2].map[2] = 1;
        perm[0][0][1][2][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 3, [3  2  1]
        perm[0][0][1][2][3].map[1] = 2;
        perm[0][0][1][2][3].map[2] = 1;
        perm[0][0][1][2][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 3, [2  1  3]
        perm[0][0][1][2][4].map[1] = 1;
        perm[0][0][1][2][4].map[2] = 3;
        perm[0][0][1][2][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 3, [1  3  2]
        perm[0][0][1][2][5].map[1] = 3;
        perm[0][0][1][2][5].map[2] = 2;
        perm[0][0][1][2][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 4, [1  2  3]
        perm[0][0][1][3][0].map[1] = 2;
        perm[0][0][1][3][0].map[2] = 3;
        perm[0][0][1][3][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 4, [3  1  2]
        perm[0][0][1][3][1].map[1] = 1;
        perm[0][0][1][3][1].map[2] = 2;
        perm[0][0][1][3][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 4, [2  3  1]
        perm[0][0][1][3][2].map[1] = 3;
        perm[0][0][1][3][2].map[2] = 1;
        perm[0][0][1][3][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 4, [3  2  1]
        perm[0][0][1][3][3].map[1] = 2;
        perm[0][0][1][3][3].map[2] = 1;
        perm[0][0][1][3][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 4, [2  1  3]
        perm[0][0][1][3][4].map[1] = 1;
        perm[0][0][1][3][4].map[2] = 3;
        perm[0][0][1][3][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 4, [1  3  2]
        perm[0][0][1][3][5].map[1] = 3;
        perm[0][0][1][3][5].map[2] = 2;
        perm[0][0][1][3][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 1, [1  2  3]
        perm[0][0][2][0][0].map[1] = 3;
        perm[0][0][2][0][0].map[2] = 1;
        perm[0][0][2][0][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 1, [3  1  2]
        perm[0][0][2][0][1].map[1] = 2;
        perm[0][0][2][0][1].map[2] = 3;
        perm[0][0][2][0][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 1, [2  3  1]
        perm[0][0][2][0][2].map[1] = 1;
        perm[0][0][2][0][2].map[2] = 2;
        perm[0][0][2][0][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 1, [3  2  1]
        perm[0][0][2][0][3].map[1] = 1;
        perm[0][0][2][0][3].map[2] = 3;
        perm[0][0][2][0][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 1, [2  1  3]
        perm[0][0][2][0][4].map[1] = 3;
        perm[0][0][2][0][4].map[2] = 2;
        perm[0][0][2][0][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 1, [1  3  2]
        perm[0][0][2][0][5].map[1] = 2;
        perm[0][0][2][0][5].map[2] = 1;
        perm[0][0][2][0][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 2, [1  2  3]
        perm[0][0][2][1][0].map[1] = 3;
        perm[0][0][2][1][0].map[2] = 1;
        perm[0][0][2][1][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 2, [3  1  2]
        perm[0][0][2][1][1].map[1] = 2;
        perm[0][0][2][1][1].map[2] = 3;
        perm[0][0][2][1][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 2, [2  3  1]
        perm[0][0][2][1][2].map[1] = 1;
        perm[0][0][2][1][2].map[2] = 2;
        perm[0][0][2][1][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 2, [3  2  1]
        perm[0][0][2][1][3].map[1] = 1;
        perm[0][0][2][1][3].map[2] = 3;
        perm[0][0][2][1][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 2, [2  1  3]
        perm[0][0][2][1][4].map[1] = 3;
        perm[0][0][2][1][4].map[2] = 2;
        perm[0][0][2][1][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 2, [1  3  2]
        perm[0][0][2][1][5].map[1] = 2;
        perm[0][0][2][1][5].map[2] = 1;
        perm[0][0][2][1][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 3, [1  2  3]
        perm[0][0][2][2][0].map[1] = 3;
        perm[0][0][2][2][0].map[2] = 1;
        perm[0][0][2][2][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 3, [3  1  2]
        perm[0][0][2][2][1].map[1] = 2;
        perm[0][0][2][2][1].map[2] = 3;
        perm[0][0][2][2][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 3, [2  3  1]
        perm[0][0][2][2][2].map[1] = 1;
        perm[0][0][2][2][2].map[2] = 2;
        perm[0][0][2][2][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 3, [3  2  1]
        perm[0][0][2][2][3].map[1] = 1;
        perm[0][0][2][2][3].map[2] = 3;
        perm[0][0][2][2][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 3, [2  1  3]
        perm[0][0][2][2][4].map[1] = 3;
        perm[0][0][2][2][4].map[2] = 2;
        perm[0][0][2][2][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 3, [1  3  2]
        perm[0][0][2][2][5].map[1] = 2;
        perm[0][0][2][2][5].map[2] = 1;
        perm[0][0][2][2][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 4, [1  2  3]
        perm[0][0][2][3][0].map[1] = 3;
        perm[0][0][2][3][0].map[2] = 1;
        perm[0][0][2][3][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 4, [3  1  2]
        perm[0][0][2][3][1].map[1] = 2;
        perm[0][0][2][3][1].map[2] = 3;
        perm[0][0][2][3][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 4, [2  3  1]
        perm[0][0][2][3][2].map[1] = 1;
        perm[0][0][2][3][2].map[2] = 2;
        perm[0][0][2][3][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 4, [3  2  1]
        perm[0][0][2][3][3].map[1] = 1;
        perm[0][0][2][3][3].map[2] = 3;
        perm[0][0][2][3][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 4, [2  1  3]
        perm[0][0][2][3][4].map[1] = 3;
        perm[0][0][2][3][4].map[2] = 2;
        perm[0][0][2][3][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 4, [1  3  2]
        perm[0][0][2][3][5].map[1] = 2;
        perm[0][0][2][3][5].map[2] = 1;
        perm[0][0][2][3][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 1, [1  2  3]
        perm[0][0][3][0][0].map[1] = 3;
        perm[0][0][3][0][0].map[2] = 2;
        perm[0][0][3][0][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 1, [3  1  2]
        perm[0][0][3][0][1].map[1] = 2;
        perm[0][0][3][0][1].map[2] = 1;
        perm[0][0][3][0][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 1, [2  3  1]
        perm[0][0][3][0][2].map[1] = 1;
        perm[0][0][3][0][2].map[2] = 3;
        perm[0][0][3][0][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 1, [3  2  1]
        perm[0][0][3][0][3].map[1] = 1;
        perm[0][0][3][0][3].map[2] = 2;
        perm[0][0][3][0][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 1, [2  1  3]
        perm[0][0][3][0][4].map[1] = 3;
        perm[0][0][3][0][4].map[2] = 1;
        perm[0][0][3][0][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 1, [1  3  2]
        perm[0][0][3][0][5].map[1] = 2;
        perm[0][0][3][0][5].map[2] = 3;
        perm[0][0][3][0][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 2, [1  2  3]
        perm[0][0][3][1][0].map[1] = 3;
        perm[0][0][3][1][0].map[2] = 2;
        perm[0][0][3][1][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 2, [3  1  2]
        perm[0][0][3][1][1].map[1] = 2;
        perm[0][0][3][1][1].map[2] = 1;
        perm[0][0][3][1][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 2, [2  3  1]
        perm[0][0][3][1][2].map[1] = 1;
        perm[0][0][3][1][2].map[2] = 3;
        perm[0][0][3][1][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 2, [3  2  1]
        perm[0][0][3][1][3].map[1] = 1;
        perm[0][0][3][1][3].map[2] = 2;
        perm[0][0][3][1][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 2, [2  1  3]
        perm[0][0][3][1][4].map[1] = 3;
        perm[0][0][3][1][4].map[2] = 1;
        perm[0][0][3][1][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 2, [1  3  2]
        perm[0][0][3][1][5].map[1] = 2;
        perm[0][0][3][1][5].map[2] = 3;
        perm[0][0][3][1][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 3, [1  2  3]
        perm[0][0][3][2][0].map[1] = 3;
        perm[0][0][3][2][0].map[2] = 2;
        perm[0][0][3][2][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 3, [3  1  2]
        perm[0][0][3][2][1].map[1] = 2;
        perm[0][0][3][2][1].map[2] = 1;
        perm[0][0][3][2][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 3, [2  3  1]
        perm[0][0][3][2][2].map[1] = 1;
        perm[0][0][3][2][2].map[2] = 3;
        perm[0][0][3][2][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 3, [3  2  1]
        perm[0][0][3][2][3].map[1] = 1;
        perm[0][0][3][2][3].map[2] = 2;
        perm[0][0][3][2][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 3, [2  1  3]
        perm[0][0][3][2][4].map[1] = 3;
        perm[0][0][3][2][4].map[2] = 1;
        perm[0][0][3][2][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 3, [1  3  2]
        perm[0][0][3][2][5].map[1] = 2;
        perm[0][0][3][2][5].map[2] = 3;
        perm[0][0][3][2][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 4, [1  2  3]
        perm[0][0][3][3][0].map[1] = 3;
        perm[0][0][3][3][0].map[2] = 2;
        perm[0][0][3][3][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 4, [3  1  2]
        perm[0][0][3][3][1].map[1] = 2;
        perm[0][0][3][3][1].map[2] = 1;
        perm[0][0][3][3][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 4, [2  3  1]
        perm[0][0][3][3][2].map[1] = 1;
        perm[0][0][3][3][2].map[2] = 3;
        perm[0][0][3][3][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 4, [3  2  1]
        perm[0][0][3][3][3].map[1] = 1;
        perm[0][0][3][3][3].map[2] = 2;
        perm[0][0][3][3][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 4, [2  1  3]
        perm[0][0][3][3][4].map[1] = 3;
        perm[0][0][3][3][4].map[2] = 1;
        perm[0][0][3][3][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 4, [1  3  2]
        perm[0][0][3][3][5].map[1] = 2;
        perm[0][0][3][3][5].map[2] = 3;
        perm[0][0][3][3][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 1, [1  2  3]
        perm[0][0][4][0][0].map[1] = 2;
        perm[0][0][4][0][0].map[2] = 1;
        perm[0][0][4][0][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 1, [3  1  2]
        perm[0][0][4][0][1].map[1] = 1;
        perm[0][0][4][0][1].map[2] = 3;
        perm[0][0][4][0][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 1, [2  3  1]
        perm[0][0][4][0][2].map[1] = 3;
        perm[0][0][4][0][2].map[2] = 2;
        perm[0][0][4][0][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 1, [3  2  1]
        perm[0][0][4][0][3].map[1] = 2;
        perm[0][0][4][0][3].map[2] = 3;
        perm[0][0][4][0][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 1, [2  1  3]
        perm[0][0][4][0][4].map[1] = 1;
        perm[0][0][4][0][4].map[2] = 2;
        perm[0][0][4][0][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 1, [1  3  2]
        perm[0][0][4][0][5].map[1] = 3;
        perm[0][0][4][0][5].map[2] = 1;
        perm[0][0][4][0][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 2, [1  2  3]
        perm[0][0][4][1][0].map[1] = 2;
        perm[0][0][4][1][0].map[2] = 1;
        perm[0][0][4][1][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 2, [3  1  2]
        perm[0][0][4][1][1].map[1] = 1;
        perm[0][0][4][1][1].map[2] = 3;
        perm[0][0][4][1][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 2, [2  3  1]
        perm[0][0][4][1][2].map[1] = 3;
        perm[0][0][4][1][2].map[2] = 2;
        perm[0][0][4][1][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 2, [3  2  1]
        perm[0][0][4][1][3].map[1] = 2;
        perm[0][0][4][1][3].map[2] = 3;
        perm[0][0][4][1][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 2, [2  1  3]
        perm[0][0][4][1][4].map[1] = 1;
        perm[0][0][4][1][4].map[2] = 2;
        perm[0][0][4][1][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 2, [1  3  2]
        perm[0][0][4][1][5].map[1] = 3;
        perm[0][0][4][1][5].map[2] = 1;
        perm[0][0][4][1][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 3, [1  2  3]
        perm[0][0][4][2][0].map[1] = 2;
        perm[0][0][4][2][0].map[2] = 1;
        perm[0][0][4][2][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 3, [3  1  2]
        perm[0][0][4][2][1].map[1] = 1;
        perm[0][0][4][2][1].map[2] = 3;
        perm[0][0][4][2][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 3, [2  3  1]
        perm[0][0][4][2][2].map[1] = 3;
        perm[0][0][4][2][2].map[2] = 2;
        perm[0][0][4][2][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 3, [3  2  1]
        perm[0][0][4][2][3].map[1] = 2;
        perm[0][0][4][2][3].map[2] = 3;
        perm[0][0][4][2][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 3, [2  1  3]
        perm[0][0][4][2][4].map[1] = 1;
        perm[0][0][4][2][4].map[2] = 2;
        perm[0][0][4][2][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 3, [1  3  2]
        perm[0][0][4][2][5].map[1] = 3;
        perm[0][0][4][2][5].map[2] = 1;
        perm[0][0][4][2][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 4, [1  2  3]
        perm[0][0][4][3][0].map[1] = 2;
        perm[0][0][4][3][0].map[2] = 1;
        perm[0][0][4][3][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 4, [3  1  2]
        perm[0][0][4][3][1].map[1] = 1;
        perm[0][0][4][3][1].map[2] = 3;
        perm[0][0][4][3][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 4, [2  3  1]
        perm[0][0][4][3][2].map[1] = 3;
        perm[0][0][4][3][2].map[2] = 2;
        perm[0][0][4][3][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 4, [3  2  1]
        perm[0][0][4][3][3].map[1] = 2;
        perm[0][0][4][3][3].map[2] = 3;
        perm[0][0][4][3][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 4, [2  1  3]
        perm[0][0][4][3][4].map[1] = 1;
        perm[0][0][4][3][4].map[2] = 2;
        perm[0][0][4][3][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 4, [1  3  2]
        perm[0][0][4][3][5].map[1] = 3;
        perm[0][0][4][3][5].map[2] = 1;
        perm[0][0][4][3][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 1, [1  2  3]
        perm[0][0][5][0][0].map[1] = 1;
        perm[0][0][5][0][0].map[2] = 3;
        perm[0][0][5][0][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 1, [3  1  2]
        perm[0][0][5][0][1].map[1] = 3;
        perm[0][0][5][0][1].map[2] = 2;
        perm[0][0][5][0][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 1, [2  3  1]
        perm[0][0][5][0][2].map[1] = 2;
        perm[0][0][5][0][2].map[2] = 1;
        perm[0][0][5][0][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 1, [3  2  1]
        perm[0][0][5][0][3].map[1] = 3;
        perm[0][0][5][0][3].map[2] = 1;
        perm[0][0][5][0][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 1, [2  1  3]
        perm[0][0][5][0][4].map[1] = 2;
        perm[0][0][5][0][4].map[2] = 3;
        perm[0][0][5][0][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 1, [1  3  2]
        perm[0][0][5][0][5].map[1] = 1;
        perm[0][0][5][0][5].map[2] = 2;
        perm[0][0][5][0][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 2, [1  2  3]
        perm[0][0][5][1][0].map[1] = 1;
        perm[0][0][5][1][0].map[2] = 3;
        perm[0][0][5][1][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 2, [3  1  2]
        perm[0][0][5][1][1].map[1] = 3;
        perm[0][0][5][1][1].map[2] = 2;
        perm[0][0][5][1][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 2, [2  3  1]
        perm[0][0][5][1][2].map[1] = 2;
        perm[0][0][5][1][2].map[2] = 1;
        perm[0][0][5][1][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 2, [3  2  1]
        perm[0][0][5][1][3].map[1] = 3;
        perm[0][0][5][1][3].map[2] = 1;
        perm[0][0][5][1][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 2, [2  1  3]
        perm[0][0][5][1][4].map[1] = 2;
        perm[0][0][5][1][4].map[2] = 3;
        perm[0][0][5][1][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 2, [1  3  2]
        perm[0][0][5][1][5].map[1] = 1;
        perm[0][0][5][1][5].map[2] = 2;
        perm[0][0][5][1][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 3, [1  2  3]
        perm[0][0][5][2][0].map[1] = 1;
        perm[0][0][5][2][0].map[2] = 3;
        perm[0][0][5][2][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 3, [3  1  2]
        perm[0][0][5][2][1].map[1] = 3;
        perm[0][0][5][2][1].map[2] = 2;
        perm[0][0][5][2][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 3, [2  3  1]
        perm[0][0][5][2][2].map[1] = 2;
        perm[0][0][5][2][2].map[2] = 1;
        perm[0][0][5][2][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 3, [3  2  1]
        perm[0][0][5][2][3].map[1] = 3;
        perm[0][0][5][2][3].map[2] = 1;
        perm[0][0][5][2][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 3, [2  1  3]
        perm[0][0][5][2][4].map[1] = 2;
        perm[0][0][5][2][4].map[2] = 3;
        perm[0][0][5][2][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 3, [1  3  2]
        perm[0][0][5][2][5].map[1] = 1;
        perm[0][0][5][2][5].map[2] = 2;
        perm[0][0][5][2][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 4, [1  2  3]
        perm[0][0][5][3][0].map[1] = 1;
        perm[0][0][5][3][0].map[2] = 3;
        perm[0][0][5][3][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 4, [3  1  2]
        perm[0][0][5][3][1].map[1] = 3;
        perm[0][0][5][3][1].map[2] = 2;
        perm[0][0][5][3][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 4, [2  3  1]
        perm[0][0][5][3][2].map[1] = 2;
        perm[0][0][5][3][2].map[2] = 1;
        perm[0][0][5][3][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 4, [3  2  1]
        perm[0][0][5][3][3].map[1] = 3;
        perm[0][0][5][3][3].map[2] = 1;
        perm[0][0][5][3][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 4, [2  1  3]
        perm[0][0][5][3][4].map[1] = 2;
        perm[0][0][5][3][4].map[2] = 3;
        perm[0][0][5][3][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 4, [1  3  2]
        perm[0][0][5][3][5].map[1] = 1;
        perm[0][0][5][3][5].map[2] = 2;
        perm[0][0][5][3][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 1, [1  2  3]
        perm[0][1][0][0][0].map[1] = 1;
        perm[0][1][0][0][0].map[2] = 2;
        perm[0][1][0][0][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 1, [3  1  2]
        perm[0][1][0][0][1].map[1] = 3;
        perm[0][1][0][0][1].map[2] = 1;
        perm[0][1][0][0][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 1, [2  3  1]
        perm[0][1][0][0][2].map[1] = 2;
        perm[0][1][0][0][2].map[2] = 3;
        perm[0][1][0][0][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 1, [3  2  1]
        perm[0][1][0][0][3].map[1] = 3;
        perm[0][1][0][0][3].map[2] = 2;
        perm[0][1][0][0][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 1, [2  1  3]
        perm[0][1][0][0][4].map[1] = 2;
        perm[0][1][0][0][4].map[2] = 1;
        perm[0][1][0][0][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 1, [1  3  2]
        perm[0][1][0][0][5].map[1] = 1;
        perm[0][1][0][0][5].map[2] = 3;
        perm[0][1][0][0][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 2, [1  2  3]
        perm[0][1][0][1][0].map[1] = 1;
        perm[0][1][0][1][0].map[2] = 2;
        perm[0][1][0][1][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 2, [3  1  2]
        perm[0][1][0][1][1].map[1] = 3;
        perm[0][1][0][1][1].map[2] = 1;
        perm[0][1][0][1][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 2, [2  3  1]
        perm[0][1][0][1][2].map[1] = 2;
        perm[0][1][0][1][2].map[2] = 3;
        perm[0][1][0][1][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 2, [3  2  1]
        perm[0][1][0][1][3].map[1] = 3;
        perm[0][1][0][1][3].map[2] = 2;
        perm[0][1][0][1][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 2, [2  1  3]
        perm[0][1][0][1][4].map[1] = 2;
        perm[0][1][0][1][4].map[2] = 1;
        perm[0][1][0][1][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 2, [1  3  2]
        perm[0][1][0][1][5].map[1] = 1;
        perm[0][1][0][1][5].map[2] = 3;
        perm[0][1][0][1][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 3, [1  2  3]
        perm[0][1][0][2][0].map[1] = 1;
        perm[0][1][0][2][0].map[2] = 2;
        perm[0][1][0][2][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 3, [3  1  2]
        perm[0][1][0][2][1].map[1] = 3;
        perm[0][1][0][2][1].map[2] = 1;
        perm[0][1][0][2][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 3, [2  3  1]
        perm[0][1][0][2][2].map[1] = 2;
        perm[0][1][0][2][2].map[2] = 3;
        perm[0][1][0][2][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 3, [3  2  1]
        perm[0][1][0][2][3].map[1] = 3;
        perm[0][1][0][2][3].map[2] = 2;
        perm[0][1][0][2][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 3, [2  1  3]
        perm[0][1][0][2][4].map[1] = 2;
        perm[0][1][0][2][4].map[2] = 1;
        perm[0][1][0][2][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 3, [1  3  2]
        perm[0][1][0][2][5].map[1] = 1;
        perm[0][1][0][2][5].map[2] = 3;
        perm[0][1][0][2][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 4, [1  2  3]
        perm[0][1][0][3][0].map[1] = 1;
        perm[0][1][0][3][0].map[2] = 2;
        perm[0][1][0][3][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 4, [3  1  2]
        perm[0][1][0][3][1].map[1] = 3;
        perm[0][1][0][3][1].map[2] = 1;
        perm[0][1][0][3][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 4, [2  3  1]
        perm[0][1][0][3][2].map[1] = 2;
        perm[0][1][0][3][2].map[2] = 3;
        perm[0][1][0][3][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 4, [3  2  1]
        perm[0][1][0][3][3].map[1] = 3;
        perm[0][1][0][3][3].map[2] = 2;
        perm[0][1][0][3][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 4, [2  1  3]
        perm[0][1][0][3][4].map[1] = 2;
        perm[0][1][0][3][4].map[2] = 1;
        perm[0][1][0][3][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 4, [1  3  2]
        perm[0][1][0][3][5].map[1] = 1;
        perm[0][1][0][3][5].map[2] = 3;
        perm[0][1][0][3][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 1, [1  2  3]
        perm[0][1][1][0][0].map[1] = 2;
        perm[0][1][1][0][0].map[2] = 3;
        perm[0][1][1][0][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 1, [3  1  2]
        perm[0][1][1][0][1].map[1] = 1;
        perm[0][1][1][0][1].map[2] = 2;
        perm[0][1][1][0][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 1, [2  3  1]
        perm[0][1][1][0][2].map[1] = 3;
        perm[0][1][1][0][2].map[2] = 1;
        perm[0][1][1][0][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 1, [3  2  1]
        perm[0][1][1][0][3].map[1] = 2;
        perm[0][1][1][0][3].map[2] = 1;
        perm[0][1][1][0][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 1, [2  1  3]
        perm[0][1][1][0][4].map[1] = 1;
        perm[0][1][1][0][4].map[2] = 3;
        perm[0][1][1][0][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 1, [1  3  2]
        perm[0][1][1][0][5].map[1] = 3;
        perm[0][1][1][0][5].map[2] = 2;
        perm[0][1][1][0][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 2, [1  2  3]
        perm[0][1][1][1][0].map[1] = 2;
        perm[0][1][1][1][0].map[2] = 3;
        perm[0][1][1][1][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 2, [3  1  2]
        perm[0][1][1][1][1].map[1] = 1;
        perm[0][1][1][1][1].map[2] = 2;
        perm[0][1][1][1][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 2, [2  3  1]
        perm[0][1][1][1][2].map[1] = 3;
        perm[0][1][1][1][2].map[2] = 1;
        perm[0][1][1][1][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 2, [3  2  1]
        perm[0][1][1][1][3].map[1] = 2;
        perm[0][1][1][1][3].map[2] = 1;
        perm[0][1][1][1][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 2, [2  1  3]
        perm[0][1][1][1][4].map[1] = 1;
        perm[0][1][1][1][4].map[2] = 3;
        perm[0][1][1][1][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 2, [1  3  2]
        perm[0][1][1][1][5].map[1] = 3;
        perm[0][1][1][1][5].map[2] = 2;
        perm[0][1][1][1][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 3, [1  2  3]
        perm[0][1][1][2][0].map[1] = 2;
        perm[0][1][1][2][0].map[2] = 3;
        perm[0][1][1][2][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 3, [3  1  2]
        perm[0][1][1][2][1].map[1] = 1;
        perm[0][1][1][2][1].map[2] = 2;
        perm[0][1][1][2][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 3, [2  3  1]
        perm[0][1][1][2][2].map[1] = 3;
        perm[0][1][1][2][2].map[2] = 1;
        perm[0][1][1][2][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 3, [3  2  1]
        perm[0][1][1][2][3].map[1] = 2;
        perm[0][1][1][2][3].map[2] = 1;
        perm[0][1][1][2][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 3, [2  1  3]
        perm[0][1][1][2][4].map[1] = 1;
        perm[0][1][1][2][4].map[2] = 3;
        perm[0][1][1][2][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 3, [1  3  2]
        perm[0][1][1][2][5].map[1] = 3;
        perm[0][1][1][2][5].map[2] = 2;
        perm[0][1][1][2][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 4, [1  2  3]
        perm[0][1][1][3][0].map[1] = 2;
        perm[0][1][1][3][0].map[2] = 3;
        perm[0][1][1][3][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 4, [3  1  2]
        perm[0][1][1][3][1].map[1] = 1;
        perm[0][1][1][3][1].map[2] = 2;
        perm[0][1][1][3][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 4, [2  3  1]
        perm[0][1][1][3][2].map[1] = 3;
        perm[0][1][1][3][2].map[2] = 1;
        perm[0][1][1][3][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 4, [3  2  1]
        perm[0][1][1][3][3].map[1] = 2;
        perm[0][1][1][3][3].map[2] = 1;
        perm[0][1][1][3][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 4, [2  1  3]
        perm[0][1][1][3][4].map[1] = 1;
        perm[0][1][1][3][4].map[2] = 3;
        perm[0][1][1][3][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 4, [1  3  2]
        perm[0][1][1][3][5].map[1] = 3;
        perm[0][1][1][3][5].map[2] = 2;
        perm[0][1][1][3][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 1, [1  2  3]
        perm[0][1][2][0][0].map[1] = 3;
        perm[0][1][2][0][0].map[2] = 1;
        perm[0][1][2][0][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 1, [3  1  2]
        perm[0][1][2][0][1].map[1] = 2;
        perm[0][1][2][0][1].map[2] = 3;
        perm[0][1][2][0][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 1, [2  3  1]
        perm[0][1][2][0][2].map[1] = 1;
        perm[0][1][2][0][2].map[2] = 2;
        perm[0][1][2][0][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 1, [3  2  1]
        perm[0][1][2][0][3].map[1] = 1;
        perm[0][1][2][0][3].map[2] = 3;
        perm[0][1][2][0][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 1, [2  1  3]
        perm[0][1][2][0][4].map[1] = 3;
        perm[0][1][2][0][4].map[2] = 2;
        perm[0][1][2][0][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 1, [1  3  2]
        perm[0][1][2][0][5].map[1] = 2;
        perm[0][1][2][0][5].map[2] = 1;
        perm[0][1][2][0][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 2, [1  2  3]
        perm[0][1][2][1][0].map[1] = 3;
        perm[0][1][2][1][0].map[2] = 1;
        perm[0][1][2][1][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 2, [3  1  2]
        perm[0][1][2][1][1].map[1] = 2;
        perm[0][1][2][1][1].map[2] = 3;
        perm[0][1][2][1][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 2, [2  3  1]
        perm[0][1][2][1][2].map[1] = 1;
        perm[0][1][2][1][2].map[2] = 2;
        perm[0][1][2][1][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 2, [3  2  1]
        perm[0][1][2][1][3].map[1] = 1;
        perm[0][1][2][1][3].map[2] = 3;
        perm[0][1][2][1][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 2, [2  1  3]
        perm[0][1][2][1][4].map[1] = 3;
        perm[0][1][2][1][4].map[2] = 2;
        perm[0][1][2][1][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 2, [1  3  2]
        perm[0][1][2][1][5].map[1] = 2;
        perm[0][1][2][1][5].map[2] = 1;
        perm[0][1][2][1][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 3, [1  2  3]
        perm[0][1][2][2][0].map[1] = 3;
        perm[0][1][2][2][0].map[2] = 1;
        perm[0][1][2][2][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 3, [3  1  2]
        perm[0][1][2][2][1].map[1] = 2;
        perm[0][1][2][2][1].map[2] = 3;
        perm[0][1][2][2][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 3, [2  3  1]
        perm[0][1][2][2][2].map[1] = 1;
        perm[0][1][2][2][2].map[2] = 2;
        perm[0][1][2][2][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 3, [3  2  1]
        perm[0][1][2][2][3].map[1] = 1;
        perm[0][1][2][2][3].map[2] = 3;
        perm[0][1][2][2][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 3, [2  1  3]
        perm[0][1][2][2][4].map[1] = 3;
        perm[0][1][2][2][4].map[2] = 2;
        perm[0][1][2][2][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 3, [1  3  2]
        perm[0][1][2][2][5].map[1] = 2;
        perm[0][1][2][2][5].map[2] = 1;
        perm[0][1][2][2][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 4, [1  2  3]
        perm[0][1][2][3][0].map[1] = 3;
        perm[0][1][2][3][0].map[2] = 1;
        perm[0][1][2][3][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 4, [3  1  2]
        perm[0][1][2][3][1].map[1] = 2;
        perm[0][1][2][3][1].map[2] = 3;
        perm[0][1][2][3][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 4, [2  3  1]
        perm[0][1][2][3][2].map[1] = 1;
        perm[0][1][2][3][2].map[2] = 2;
        perm[0][1][2][3][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 4, [3  2  1]
        perm[0][1][2][3][3].map[1] = 1;
        perm[0][1][2][3][3].map[2] = 3;
        perm[0][1][2][3][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 4, [2  1  3]
        perm[0][1][2][3][4].map[1] = 3;
        perm[0][1][2][3][4].map[2] = 2;
        perm[0][1][2][3][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 4, [1  3  2]
        perm[0][1][2][3][5].map[1] = 2;
        perm[0][1][2][3][5].map[2] = 1;
        perm[0][1][2][3][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 1, [1  2  3]
        perm[0][1][3][0][0].map[1] = 3;
        perm[0][1][3][0][0].map[2] = 2;
        perm[0][1][3][0][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 1, [3  1  2]
        perm[0][1][3][0][1].map[1] = 2;
        perm[0][1][3][0][1].map[2] = 1;
        perm[0][1][3][0][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 1, [2  3  1]
        perm[0][1][3][0][2].map[1] = 1;
        perm[0][1][3][0][2].map[2] = 3;
        perm[0][1][3][0][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 1, [3  2  1]
        perm[0][1][3][0][3].map[1] = 1;
        perm[0][1][3][0][3].map[2] = 2;
        perm[0][1][3][0][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 1, [2  1  3]
        perm[0][1][3][0][4].map[1] = 3;
        perm[0][1][3][0][4].map[2] = 1;
        perm[0][1][3][0][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 1, [1  3  2]
        perm[0][1][3][0][5].map[1] = 2;
        perm[0][1][3][0][5].map[2] = 3;
        perm[0][1][3][0][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 2, [1  2  3]
        perm[0][1][3][1][0].map[1] = 3;
        perm[0][1][3][1][0].map[2] = 2;
        perm[0][1][3][1][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 2, [3  1  2]
        perm[0][1][3][1][1].map[1] = 2;
        perm[0][1][3][1][1].map[2] = 1;
        perm[0][1][3][1][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 2, [2  3  1]
        perm[0][1][3][1][2].map[1] = 1;
        perm[0][1][3][1][2].map[2] = 3;
        perm[0][1][3][1][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 2, [3  2  1]
        perm[0][1][3][1][3].map[1] = 1;
        perm[0][1][3][1][3].map[2] = 2;
        perm[0][1][3][1][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 2, [2  1  3]
        perm[0][1][3][1][4].map[1] = 3;
        perm[0][1][3][1][4].map[2] = 1;
        perm[0][1][3][1][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 2, [1  3  2]
        perm[0][1][3][1][5].map[1] = 2;
        perm[0][1][3][1][5].map[2] = 3;
        perm[0][1][3][1][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 3, [1  2  3]
        perm[0][1][3][2][0].map[1] = 3;
        perm[0][1][3][2][0].map[2] = 2;
        perm[0][1][3][2][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 3, [3  1  2]
        perm[0][1][3][2][1].map[1] = 2;
        perm[0][1][3][2][1].map[2] = 1;
        perm[0][1][3][2][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 3, [2  3  1]
        perm[0][1][3][2][2].map[1] = 1;
        perm[0][1][3][2][2].map[2] = 3;
        perm[0][1][3][2][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 3, [3  2  1]
        perm[0][1][3][2][3].map[1] = 1;
        perm[0][1][3][2][3].map[2] = 2;
        perm[0][1][3][2][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 3, [2  1  3]
        perm[0][1][3][2][4].map[1] = 3;
        perm[0][1][3][2][4].map[2] = 1;
        perm[0][1][3][2][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 3, [1  3  2]
        perm[0][1][3][2][5].map[1] = 2;
        perm[0][1][3][2][5].map[2] = 3;
        perm[0][1][3][2][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 4, [1  2  3]
        perm[0][1][3][3][0].map[1] = 3;
        perm[0][1][3][3][0].map[2] = 2;
        perm[0][1][3][3][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 4, [3  1  2]
        perm[0][1][3][3][1].map[1] = 2;
        perm[0][1][3][3][1].map[2] = 1;
        perm[0][1][3][3][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 4, [2  3  1]
        perm[0][1][3][3][2].map[1] = 1;
        perm[0][1][3][3][2].map[2] = 3;
        perm[0][1][3][3][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 4, [3  2  1]
        perm[0][1][3][3][3].map[1] = 1;
        perm[0][1][3][3][3].map[2] = 2;
        perm[0][1][3][3][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 4, [2  1  3]
        perm[0][1][3][3][4].map[1] = 3;
        perm[0][1][3][3][4].map[2] = 1;
        perm[0][1][3][3][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 4, [1  3  2]
        perm[0][1][3][3][5].map[1] = 2;
        perm[0][1][3][3][5].map[2] = 3;
        perm[0][1][3][3][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 1, [1  2  3]
        perm[0][1][4][0][0].map[1] = 2;
        perm[0][1][4][0][0].map[2] = 1;
        perm[0][1][4][0][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 1, [3  1  2]
        perm[0][1][4][0][1].map[1] = 1;
        perm[0][1][4][0][1].map[2] = 3;
        perm[0][1][4][0][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 1, [2  3  1]
        perm[0][1][4][0][2].map[1] = 3;
        perm[0][1][4][0][2].map[2] = 2;
        perm[0][1][4][0][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 1, [3  2  1]
        perm[0][1][4][0][3].map[1] = 2;
        perm[0][1][4][0][3].map[2] = 3;
        perm[0][1][4][0][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 1, [2  1  3]
        perm[0][1][4][0][4].map[1] = 1;
        perm[0][1][4][0][4].map[2] = 2;
        perm[0][1][4][0][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 1, [1  3  2]
        perm[0][1][4][0][5].map[1] = 3;
        perm[0][1][4][0][5].map[2] = 1;
        perm[0][1][4][0][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 2, [1  2  3]
        perm[0][1][4][1][0].map[1] = 2;
        perm[0][1][4][1][0].map[2] = 1;
        perm[0][1][4][1][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 2, [3  1  2]
        perm[0][1][4][1][1].map[1] = 1;
        perm[0][1][4][1][1].map[2] = 3;
        perm[0][1][4][1][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 2, [2  3  1]
        perm[0][1][4][1][2].map[1] = 3;
        perm[0][1][4][1][2].map[2] = 2;
        perm[0][1][4][1][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 2, [3  2  1]
        perm[0][1][4][1][3].map[1] = 2;
        perm[0][1][4][1][3].map[2] = 3;
        perm[0][1][4][1][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 2, [2  1  3]
        perm[0][1][4][1][4].map[1] = 1;
        perm[0][1][4][1][4].map[2] = 2;
        perm[0][1][4][1][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 2, [1  3  2]
        perm[0][1][4][1][5].map[1] = 3;
        perm[0][1][4][1][5].map[2] = 1;
        perm[0][1][4][1][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 3, [1  2  3]
        perm[0][1][4][2][0].map[1] = 2;
        perm[0][1][4][2][0].map[2] = 1;
        perm[0][1][4][2][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 3, [3  1  2]
        perm[0][1][4][2][1].map[1] = 1;
        perm[0][1][4][2][1].map[2] = 3;
        perm[0][1][4][2][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 3, [2  3  1]
        perm[0][1][4][2][2].map[1] = 3;
        perm[0][1][4][2][2].map[2] = 2;
        perm[0][1][4][2][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 3, [3  2  1]
        perm[0][1][4][2][3].map[1] = 2;
        perm[0][1][4][2][3].map[2] = 3;
        perm[0][1][4][2][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 3, [2  1  3]
        perm[0][1][4][2][4].map[1] = 1;
        perm[0][1][4][2][4].map[2] = 2;
        perm[0][1][4][2][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 3, [1  3  2]
        perm[0][1][4][2][5].map[1] = 3;
        perm[0][1][4][2][5].map[2] = 1;
        perm[0][1][4][2][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 4, [1  2  3]
        perm[0][1][4][3][0].map[1] = 2;
        perm[0][1][4][3][0].map[2] = 1;
        perm[0][1][4][3][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 4, [3  1  2]
        perm[0][1][4][3][1].map[1] = 1;
        perm[0][1][4][3][1].map[2] = 3;
        perm[0][1][4][3][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 4, [2  3  1]
        perm[0][1][4][3][2].map[1] = 3;
        perm[0][1][4][3][2].map[2] = 2;
        perm[0][1][4][3][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 4, [3  2  1]
        perm[0][1][4][3][3].map[1] = 2;
        perm[0][1][4][3][3].map[2] = 3;
        perm[0][1][4][3][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 4, [2  1  3]
        perm[0][1][4][3][4].map[1] = 1;
        perm[0][1][4][3][4].map[2] = 2;
        perm[0][1][4][3][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 4, [1  3  2]
        perm[0][1][4][3][5].map[1] = 3;
        perm[0][1][4][3][5].map[2] = 1;
        perm[0][1][4][3][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 1, [1  2  3]
        perm[0][1][5][0][0].map[1] = 1;
        perm[0][1][5][0][0].map[2] = 3;
        perm[0][1][5][0][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 1, [3  1  2]
        perm[0][1][5][0][1].map[1] = 3;
        perm[0][1][5][0][1].map[2] = 2;
        perm[0][1][5][0][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 1, [2  3  1]
        perm[0][1][5][0][2].map[1] = 2;
        perm[0][1][5][0][2].map[2] = 1;
        perm[0][1][5][0][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 1, [3  2  1]
        perm[0][1][5][0][3].map[1] = 3;
        perm[0][1][5][0][3].map[2] = 1;
        perm[0][1][5][0][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 1, [2  1  3]
        perm[0][1][5][0][4].map[1] = 2;
        perm[0][1][5][0][4].map[2] = 3;
        perm[0][1][5][0][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 1, [1  3  2]
        perm[0][1][5][0][5].map[1] = 1;
        perm[0][1][5][0][5].map[2] = 2;
        perm[0][1][5][0][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 2, [1  2  3]
        perm[0][1][5][1][0].map[1] = 1;
        perm[0][1][5][1][0].map[2] = 3;
        perm[0][1][5][1][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 2, [3  1  2]
        perm[0][1][5][1][1].map[1] = 3;
        perm[0][1][5][1][1].map[2] = 2;
        perm[0][1][5][1][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 2, [2  3  1]
        perm[0][1][5][1][2].map[1] = 2;
        perm[0][1][5][1][2].map[2] = 1;
        perm[0][1][5][1][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 2, [3  2  1]
        perm[0][1][5][1][3].map[1] = 3;
        perm[0][1][5][1][3].map[2] = 1;
        perm[0][1][5][1][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 2, [2  1  3]
        perm[0][1][5][1][4].map[1] = 2;
        perm[0][1][5][1][4].map[2] = 3;
        perm[0][1][5][1][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 2, [1  3  2]
        perm[0][1][5][1][5].map[1] = 1;
        perm[0][1][5][1][5].map[2] = 2;
        perm[0][1][5][1][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 3, [1  2  3]
        perm[0][1][5][2][0].map[1] = 1;
        perm[0][1][5][2][0].map[2] = 3;
        perm[0][1][5][2][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 3, [3  1  2]
        perm[0][1][5][2][1].map[1] = 3;
        perm[0][1][5][2][1].map[2] = 2;
        perm[0][1][5][2][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 3, [2  3  1]
        perm[0][1][5][2][2].map[1] = 2;
        perm[0][1][5][2][2].map[2] = 1;
        perm[0][1][5][2][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 3, [3  2  1]
        perm[0][1][5][2][3].map[1] = 3;
        perm[0][1][5][2][3].map[2] = 1;
        perm[0][1][5][2][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 3, [2  1  3]
        perm[0][1][5][2][4].map[1] = 2;
        perm[0][1][5][2][4].map[2] = 3;
        perm[0][1][5][2][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 3, [1  3  2]
        perm[0][1][5][2][5].map[1] = 1;
        perm[0][1][5][2][5].map[2] = 2;
        perm[0][1][5][2][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 4, [1  2  3]
        perm[0][1][5][3][0].map[1] = 1;
        perm[0][1][5][3][0].map[2] = 3;
        perm[0][1][5][3][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 4, [3  1  2]
        perm[0][1][5][3][1].map[1] = 3;
        perm[0][1][5][3][1].map[2] = 2;
        perm[0][1][5][3][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 4, [2  3  1]
        perm[0][1][5][3][2].map[1] = 2;
        perm[0][1][5][3][2].map[2] = 1;
        perm[0][1][5][3][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 4, [3  2  1]
        perm[0][1][5][3][3].map[1] = 3;
        perm[0][1][5][3][3].map[2] = 1;
        perm[0][1][5][3][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 4, [2  1  3]
        perm[0][1][5][3][4].map[1] = 2;
        perm[0][1][5][3][4].map[2] = 3;
        perm[0][1][5][3][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 4, [1  3  2]
        perm[0][1][5][3][5].map[1] = 1;
        perm[0][1][5][3][5].map[2] = 2;
        perm[0][1][5][3][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 1, [1  2  3]
        perm[0][2][0][0][0].map[1] = 1;
        perm[0][2][0][0][0].map[2] = 2;
        perm[0][2][0][0][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 1, [3  1  2]
        perm[0][2][0][0][1].map[1] = 3;
        perm[0][2][0][0][1].map[2] = 1;
        perm[0][2][0][0][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 1, [2  3  1]
        perm[0][2][0][0][2].map[1] = 2;
        perm[0][2][0][0][2].map[2] = 3;
        perm[0][2][0][0][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 1, [3  2  1]
        perm[0][2][0][0][3].map[1] = 3;
        perm[0][2][0][0][3].map[2] = 2;
        perm[0][2][0][0][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 1, [2  1  3]
        perm[0][2][0][0][4].map[1] = 2;
        perm[0][2][0][0][4].map[2] = 1;
        perm[0][2][0][0][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 1, [1  3  2]
        perm[0][2][0][0][5].map[1] = 1;
        perm[0][2][0][0][5].map[2] = 3;
        perm[0][2][0][0][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 2, [1  2  3]
        perm[0][2][0][1][0].map[1] = 1;
        perm[0][2][0][1][0].map[2] = 2;
        perm[0][2][0][1][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 2, [3  1  2]
        perm[0][2][0][1][1].map[1] = 3;
        perm[0][2][0][1][1].map[2] = 1;
        perm[0][2][0][1][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 2, [2  3  1]
        perm[0][2][0][1][2].map[1] = 2;
        perm[0][2][0][1][2].map[2] = 3;
        perm[0][2][0][1][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 2, [3  2  1]
        perm[0][2][0][1][3].map[1] = 3;
        perm[0][2][0][1][3].map[2] = 2;
        perm[0][2][0][1][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 2, [2  1  3]
        perm[0][2][0][1][4].map[1] = 2;
        perm[0][2][0][1][4].map[2] = 1;
        perm[0][2][0][1][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 2, [1  3  2]
        perm[0][2][0][1][5].map[1] = 1;
        perm[0][2][0][1][5].map[2] = 3;
        perm[0][2][0][1][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 3, [1  2  3]
        perm[0][2][0][2][0].map[1] = 1;
        perm[0][2][0][2][0].map[2] = 2;
        perm[0][2][0][2][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 3, [3  1  2]
        perm[0][2][0][2][1].map[1] = 3;
        perm[0][2][0][2][1].map[2] = 1;
        perm[0][2][0][2][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 3, [2  3  1]
        perm[0][2][0][2][2].map[1] = 2;
        perm[0][2][0][2][2].map[2] = 3;
        perm[0][2][0][2][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 3, [3  2  1]
        perm[0][2][0][2][3].map[1] = 3;
        perm[0][2][0][2][3].map[2] = 2;
        perm[0][2][0][2][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 3, [2  1  3]
        perm[0][2][0][2][4].map[1] = 2;
        perm[0][2][0][2][4].map[2] = 1;
        perm[0][2][0][2][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 3, [1  3  2]
        perm[0][2][0][2][5].map[1] = 1;
        perm[0][2][0][2][5].map[2] = 3;
        perm[0][2][0][2][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 4, [1  2  3]
        perm[0][2][0][3][0].map[1] = 1;
        perm[0][2][0][3][0].map[2] = 2;
        perm[0][2][0][3][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 4, [3  1  2]
        perm[0][2][0][3][1].map[1] = 3;
        perm[0][2][0][3][1].map[2] = 1;
        perm[0][2][0][3][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 4, [2  3  1]
        perm[0][2][0][3][2].map[1] = 2;
        perm[0][2][0][3][2].map[2] = 3;
        perm[0][2][0][3][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 4, [3  2  1]
        perm[0][2][0][3][3].map[1] = 3;
        perm[0][2][0][3][3].map[2] = 2;
        perm[0][2][0][3][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 4, [2  1  3]
        perm[0][2][0][3][4].map[1] = 2;
        perm[0][2][0][3][4].map[2] = 1;
        perm[0][2][0][3][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 4, [1  3  2]
        perm[0][2][0][3][5].map[1] = 1;
        perm[0][2][0][3][5].map[2] = 3;
        perm[0][2][0][3][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 1, [1  2  3]
        perm[0][2][1][0][0].map[1] = 2;
        perm[0][2][1][0][0].map[2] = 3;
        perm[0][2][1][0][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 1, [3  1  2]
        perm[0][2][1][0][1].map[1] = 1;
        perm[0][2][1][0][1].map[2] = 2;
        perm[0][2][1][0][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 1, [2  3  1]
        perm[0][2][1][0][2].map[1] = 3;
        perm[0][2][1][0][2].map[2] = 1;
        perm[0][2][1][0][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 1, [3  2  1]
        perm[0][2][1][0][3].map[1] = 2;
        perm[0][2][1][0][3].map[2] = 1;
        perm[0][2][1][0][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 1, [2  1  3]
        perm[0][2][1][0][4].map[1] = 1;
        perm[0][2][1][0][4].map[2] = 3;
        perm[0][2][1][0][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 1, [1  3  2]
        perm[0][2][1][0][5].map[1] = 3;
        perm[0][2][1][0][5].map[2] = 2;
        perm[0][2][1][0][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 2, [1  2  3]
        perm[0][2][1][1][0].map[1] = 2;
        perm[0][2][1][1][0].map[2] = 3;
        perm[0][2][1][1][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 2, [3  1  2]
        perm[0][2][1][1][1].map[1] = 1;
        perm[0][2][1][1][1].map[2] = 2;
        perm[0][2][1][1][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 2, [2  3  1]
        perm[0][2][1][1][2].map[1] = 3;
        perm[0][2][1][1][2].map[2] = 1;
        perm[0][2][1][1][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 2, [3  2  1]
        perm[0][2][1][1][3].map[1] = 2;
        perm[0][2][1][1][3].map[2] = 1;
        perm[0][2][1][1][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 2, [2  1  3]
        perm[0][2][1][1][4].map[1] = 1;
        perm[0][2][1][1][4].map[2] = 3;
        perm[0][2][1][1][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 2, [1  3  2]
        perm[0][2][1][1][5].map[1] = 3;
        perm[0][2][1][1][5].map[2] = 2;
        perm[0][2][1][1][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 3, [1  2  3]
        perm[0][2][1][2][0].map[1] = 2;
        perm[0][2][1][2][0].map[2] = 3;
        perm[0][2][1][2][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 3, [3  1  2]
        perm[0][2][1][2][1].map[1] = 1;
        perm[0][2][1][2][1].map[2] = 2;
        perm[0][2][1][2][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 3, [2  3  1]
        perm[0][2][1][2][2].map[1] = 3;
        perm[0][2][1][2][2].map[2] = 1;
        perm[0][2][1][2][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 3, [3  2  1]
        perm[0][2][1][2][3].map[1] = 2;
        perm[0][2][1][2][3].map[2] = 1;
        perm[0][2][1][2][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 3, [2  1  3]
        perm[0][2][1][2][4].map[1] = 1;
        perm[0][2][1][2][4].map[2] = 3;
        perm[0][2][1][2][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 3, [1  3  2]
        perm[0][2][1][2][5].map[1] = 3;
        perm[0][2][1][2][5].map[2] = 2;
        perm[0][2][1][2][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 4, [1  2  3]
        perm[0][2][1][3][0].map[1] = 2;
        perm[0][2][1][3][0].map[2] = 3;
        perm[0][2][1][3][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 4, [3  1  2]
        perm[0][2][1][3][1].map[1] = 1;
        perm[0][2][1][3][1].map[2] = 2;
        perm[0][2][1][3][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 4, [2  3  1]
        perm[0][2][1][3][2].map[1] = 3;
        perm[0][2][1][3][2].map[2] = 1;
        perm[0][2][1][3][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 4, [3  2  1]
        perm[0][2][1][3][3].map[1] = 2;
        perm[0][2][1][3][3].map[2] = 1;
        perm[0][2][1][3][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 4, [2  1  3]
        perm[0][2][1][3][4].map[1] = 1;
        perm[0][2][1][3][4].map[2] = 3;
        perm[0][2][1][3][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 4, [1  3  2]
        perm[0][2][1][3][5].map[1] = 3;
        perm[0][2][1][3][5].map[2] = 2;
        perm[0][2][1][3][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 1, [1  2  3]
        perm[0][2][2][0][0].map[1] = 3;
        perm[0][2][2][0][0].map[2] = 1;
        perm[0][2][2][0][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 1, [3  1  2]
        perm[0][2][2][0][1].map[1] = 2;
        perm[0][2][2][0][1].map[2] = 3;
        perm[0][2][2][0][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 1, [2  3  1]
        perm[0][2][2][0][2].map[1] = 1;
        perm[0][2][2][0][2].map[2] = 2;
        perm[0][2][2][0][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 1, [3  2  1]
        perm[0][2][2][0][3].map[1] = 1;
        perm[0][2][2][0][3].map[2] = 3;
        perm[0][2][2][0][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 1, [2  1  3]
        perm[0][2][2][0][4].map[1] = 3;
        perm[0][2][2][0][4].map[2] = 2;
        perm[0][2][2][0][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 1, [1  3  2]
        perm[0][2][2][0][5].map[1] = 2;
        perm[0][2][2][0][5].map[2] = 1;
        perm[0][2][2][0][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 2, [1  2  3]
        perm[0][2][2][1][0].map[1] = 3;
        perm[0][2][2][1][0].map[2] = 1;
        perm[0][2][2][1][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 2, [3  1  2]
        perm[0][2][2][1][1].map[1] = 2;
        perm[0][2][2][1][1].map[2] = 3;
        perm[0][2][2][1][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 2, [2  3  1]
        perm[0][2][2][1][2].map[1] = 1;
        perm[0][2][2][1][2].map[2] = 2;
        perm[0][2][2][1][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 2, [3  2  1]
        perm[0][2][2][1][3].map[1] = 1;
        perm[0][2][2][1][3].map[2] = 3;
        perm[0][2][2][1][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 2, [2  1  3]
        perm[0][2][2][1][4].map[1] = 3;
        perm[0][2][2][1][4].map[2] = 2;
        perm[0][2][2][1][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 2, [1  3  2]
        perm[0][2][2][1][5].map[1] = 2;
        perm[0][2][2][1][5].map[2] = 1;
        perm[0][2][2][1][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 3, [1  2  3]
        perm[0][2][2][2][0].map[1] = 3;
        perm[0][2][2][2][0].map[2] = 1;
        perm[0][2][2][2][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 3, [3  1  2]
        perm[0][2][2][2][1].map[1] = 2;
        perm[0][2][2][2][1].map[2] = 3;
        perm[0][2][2][2][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 3, [2  3  1]
        perm[0][2][2][2][2].map[1] = 1;
        perm[0][2][2][2][2].map[2] = 2;
        perm[0][2][2][2][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 3, [3  2  1]
        perm[0][2][2][2][3].map[1] = 1;
        perm[0][2][2][2][3].map[2] = 3;
        perm[0][2][2][2][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 3, [2  1  3]
        perm[0][2][2][2][4].map[1] = 3;
        perm[0][2][2][2][4].map[2] = 2;
        perm[0][2][2][2][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 3, [1  3  2]
        perm[0][2][2][2][5].map[1] = 2;
        perm[0][2][2][2][5].map[2] = 1;
        perm[0][2][2][2][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 4, [1  2  3]
        perm[0][2][2][3][0].map[1] = 3;
        perm[0][2][2][3][0].map[2] = 1;
        perm[0][2][2][3][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 4, [3  1  2]
        perm[0][2][2][3][1].map[1] = 2;
        perm[0][2][2][3][1].map[2] = 3;
        perm[0][2][2][3][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 4, [2  3  1]
        perm[0][2][2][3][2].map[1] = 1;
        perm[0][2][2][3][2].map[2] = 2;
        perm[0][2][2][3][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 4, [3  2  1]
        perm[0][2][2][3][3].map[1] = 1;
        perm[0][2][2][3][3].map[2] = 3;
        perm[0][2][2][3][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 4, [2  1  3]
        perm[0][2][2][3][4].map[1] = 3;
        perm[0][2][2][3][4].map[2] = 2;
        perm[0][2][2][3][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 4, [1  3  2]
        perm[0][2][2][3][5].map[1] = 2;
        perm[0][2][2][3][5].map[2] = 1;
        perm[0][2][2][3][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 1, [1  2  3]
        perm[0][2][3][0][0].map[1] = 3;
        perm[0][2][3][0][0].map[2] = 2;
        perm[0][2][3][0][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 1, [3  1  2]
        perm[0][2][3][0][1].map[1] = 2;
        perm[0][2][3][0][1].map[2] = 1;
        perm[0][2][3][0][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 1, [2  3  1]
        perm[0][2][3][0][2].map[1] = 1;
        perm[0][2][3][0][2].map[2] = 3;
        perm[0][2][3][0][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 1, [3  2  1]
        perm[0][2][3][0][3].map[1] = 1;
        perm[0][2][3][0][3].map[2] = 2;
        perm[0][2][3][0][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 1, [2  1  3]
        perm[0][2][3][0][4].map[1] = 3;
        perm[0][2][3][0][4].map[2] = 1;
        perm[0][2][3][0][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 1, [1  3  2]
        perm[0][2][3][0][5].map[1] = 2;
        perm[0][2][3][0][5].map[2] = 3;
        perm[0][2][3][0][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 2, [1  2  3]
        perm[0][2][3][1][0].map[1] = 3;
        perm[0][2][3][1][0].map[2] = 2;
        perm[0][2][3][1][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 2, [3  1  2]
        perm[0][2][3][1][1].map[1] = 2;
        perm[0][2][3][1][1].map[2] = 1;
        perm[0][2][3][1][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 2, [2  3  1]
        perm[0][2][3][1][2].map[1] = 1;
        perm[0][2][3][1][2].map[2] = 3;
        perm[0][2][3][1][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 2, [3  2  1]
        perm[0][2][3][1][3].map[1] = 1;
        perm[0][2][3][1][3].map[2] = 2;
        perm[0][2][3][1][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 2, [2  1  3]
        perm[0][2][3][1][4].map[1] = 3;
        perm[0][2][3][1][4].map[2] = 1;
        perm[0][2][3][1][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 2, [1  3  2]
        perm[0][2][3][1][5].map[1] = 2;
        perm[0][2][3][1][5].map[2] = 3;
        perm[0][2][3][1][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 3, [1  2  3]
        perm[0][2][3][2][0].map[1] = 3;
        perm[0][2][3][2][0].map[2] = 2;
        perm[0][2][3][2][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 3, [3  1  2]
        perm[0][2][3][2][1].map[1] = 2;
        perm[0][2][3][2][1].map[2] = 1;
        perm[0][2][3][2][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 3, [2  3  1]
        perm[0][2][3][2][2].map[1] = 1;
        perm[0][2][3][2][2].map[2] = 3;
        perm[0][2][3][2][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 3, [3  2  1]
        perm[0][2][3][2][3].map[1] = 1;
        perm[0][2][3][2][3].map[2] = 2;
        perm[0][2][3][2][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 3, [2  1  3]
        perm[0][2][3][2][4].map[1] = 3;
        perm[0][2][3][2][4].map[2] = 1;
        perm[0][2][3][2][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 3, [1  3  2]
        perm[0][2][3][2][5].map[1] = 2;
        perm[0][2][3][2][5].map[2] = 3;
        perm[0][2][3][2][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 4, [1  2  3]
        perm[0][2][3][3][0].map[1] = 3;
        perm[0][2][3][3][0].map[2] = 2;
        perm[0][2][3][3][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 4, [3  1  2]
        perm[0][2][3][3][1].map[1] = 2;
        perm[0][2][3][3][1].map[2] = 1;
        perm[0][2][3][3][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 4, [2  3  1]
        perm[0][2][3][3][2].map[1] = 1;
        perm[0][2][3][3][2].map[2] = 3;
        perm[0][2][3][3][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 4, [3  2  1]
        perm[0][2][3][3][3].map[1] = 1;
        perm[0][2][3][3][3].map[2] = 2;
        perm[0][2][3][3][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 4, [2  1  3]
        perm[0][2][3][3][4].map[1] = 3;
        perm[0][2][3][3][4].map[2] = 1;
        perm[0][2][3][3][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 4, [1  3  2]
        perm[0][2][3][3][5].map[1] = 2;
        perm[0][2][3][3][5].map[2] = 3;
        perm[0][2][3][3][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 1, [1  2  3]
        perm[0][2][4][0][0].map[1] = 2;
        perm[0][2][4][0][0].map[2] = 1;
        perm[0][2][4][0][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 1, [3  1  2]
        perm[0][2][4][0][1].map[1] = 1;
        perm[0][2][4][0][1].map[2] = 3;
        perm[0][2][4][0][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 1, [2  3  1]
        perm[0][2][4][0][2].map[1] = 3;
        perm[0][2][4][0][2].map[2] = 2;
        perm[0][2][4][0][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 1, [3  2  1]
        perm[0][2][4][0][3].map[1] = 2;
        perm[0][2][4][0][3].map[2] = 3;
        perm[0][2][4][0][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 1, [2  1  3]
        perm[0][2][4][0][4].map[1] = 1;
        perm[0][2][4][0][4].map[2] = 2;
        perm[0][2][4][0][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 1, [1  3  2]
        perm[0][2][4][0][5].map[1] = 3;
        perm[0][2][4][0][5].map[2] = 1;
        perm[0][2][4][0][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 2, [1  2  3]
        perm[0][2][4][1][0].map[1] = 2;
        perm[0][2][4][1][0].map[2] = 1;
        perm[0][2][4][1][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 2, [3  1  2]
        perm[0][2][4][1][1].map[1] = 1;
        perm[0][2][4][1][1].map[2] = 3;
        perm[0][2][4][1][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 2, [2  3  1]
        perm[0][2][4][1][2].map[1] = 3;
        perm[0][2][4][1][2].map[2] = 2;
        perm[0][2][4][1][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 2, [3  2  1]
        perm[0][2][4][1][3].map[1] = 2;
        perm[0][2][4][1][3].map[2] = 3;
        perm[0][2][4][1][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 2, [2  1  3]
        perm[0][2][4][1][4].map[1] = 1;
        perm[0][2][4][1][4].map[2] = 2;
        perm[0][2][4][1][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 2, [1  3  2]
        perm[0][2][4][1][5].map[1] = 3;
        perm[0][2][4][1][5].map[2] = 1;
        perm[0][2][4][1][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 3, [1  2  3]
        perm[0][2][4][2][0].map[1] = 2;
        perm[0][2][4][2][0].map[2] = 1;
        perm[0][2][4][2][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 3, [3  1  2]
        perm[0][2][4][2][1].map[1] = 1;
        perm[0][2][4][2][1].map[2] = 3;
        perm[0][2][4][2][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 3, [2  3  1]
        perm[0][2][4][2][2].map[1] = 3;
        perm[0][2][4][2][2].map[2] = 2;
        perm[0][2][4][2][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 3, [3  2  1]
        perm[0][2][4][2][3].map[1] = 2;
        perm[0][2][4][2][3].map[2] = 3;
        perm[0][2][4][2][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 3, [2  1  3]
        perm[0][2][4][2][4].map[1] = 1;
        perm[0][2][4][2][4].map[2] = 2;
        perm[0][2][4][2][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 3, [1  3  2]
        perm[0][2][4][2][5].map[1] = 3;
        perm[0][2][4][2][5].map[2] = 1;
        perm[0][2][4][2][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 4, [1  2  3]
        perm[0][2][4][3][0].map[1] = 2;
        perm[0][2][4][3][0].map[2] = 1;
        perm[0][2][4][3][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 4, [3  1  2]
        perm[0][2][4][3][1].map[1] = 1;
        perm[0][2][4][3][1].map[2] = 3;
        perm[0][2][4][3][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 4, [2  3  1]
        perm[0][2][4][3][2].map[1] = 3;
        perm[0][2][4][3][2].map[2] = 2;
        perm[0][2][4][3][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 4, [3  2  1]
        perm[0][2][4][3][3].map[1] = 2;
        perm[0][2][4][3][3].map[2] = 3;
        perm[0][2][4][3][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 4, [2  1  3]
        perm[0][2][4][3][4].map[1] = 1;
        perm[0][2][4][3][4].map[2] = 2;
        perm[0][2][4][3][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 4, [1  3  2]
        perm[0][2][4][3][5].map[1] = 3;
        perm[0][2][4][3][5].map[2] = 1;
        perm[0][2][4][3][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 1, [1  2  3]
        perm[0][2][5][0][0].map[1] = 1;
        perm[0][2][5][0][0].map[2] = 3;
        perm[0][2][5][0][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 1, [3  1  2]
        perm[0][2][5][0][1].map[1] = 3;
        perm[0][2][5][0][1].map[2] = 2;
        perm[0][2][5][0][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 1, [2  3  1]
        perm[0][2][5][0][2].map[1] = 2;
        perm[0][2][5][0][2].map[2] = 1;
        perm[0][2][5][0][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 1, [3  2  1]
        perm[0][2][5][0][3].map[1] = 3;
        perm[0][2][5][0][3].map[2] = 1;
        perm[0][2][5][0][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 1, [2  1  3]
        perm[0][2][5][0][4].map[1] = 2;
        perm[0][2][5][0][4].map[2] = 3;
        perm[0][2][5][0][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 1, [1  3  2]
        perm[0][2][5][0][5].map[1] = 1;
        perm[0][2][5][0][5].map[2] = 2;
        perm[0][2][5][0][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 2, [1  2  3]
        perm[0][2][5][1][0].map[1] = 1;
        perm[0][2][5][1][0].map[2] = 3;
        perm[0][2][5][1][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 2, [3  1  2]
        perm[0][2][5][1][1].map[1] = 3;
        perm[0][2][5][1][1].map[2] = 2;
        perm[0][2][5][1][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 2, [2  3  1]
        perm[0][2][5][1][2].map[1] = 2;
        perm[0][2][5][1][2].map[2] = 1;
        perm[0][2][5][1][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 2, [3  2  1]
        perm[0][2][5][1][3].map[1] = 3;
        perm[0][2][5][1][3].map[2] = 1;
        perm[0][2][5][1][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 2, [2  1  3]
        perm[0][2][5][1][4].map[1] = 2;
        perm[0][2][5][1][4].map[2] = 3;
        perm[0][2][5][1][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 2, [1  3  2]
        perm[0][2][5][1][5].map[1] = 1;
        perm[0][2][5][1][5].map[2] = 2;
        perm[0][2][5][1][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 3, [1  2  3]
        perm[0][2][5][2][0].map[1] = 1;
        perm[0][2][5][2][0].map[2] = 3;
        perm[0][2][5][2][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 3, [3  1  2]
        perm[0][2][5][2][1].map[1] = 3;
        perm[0][2][5][2][1].map[2] = 2;
        perm[0][2][5][2][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 3, [2  3  1]
        perm[0][2][5][2][2].map[1] = 2;
        perm[0][2][5][2][2].map[2] = 1;
        perm[0][2][5][2][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 3, [3  2  1]
        perm[0][2][5][2][3].map[1] = 3;
        perm[0][2][5][2][3].map[2] = 1;
        perm[0][2][5][2][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 3, [2  1  3]
        perm[0][2][5][2][4].map[1] = 2;
        perm[0][2][5][2][4].map[2] = 3;
        perm[0][2][5][2][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 3, [1  3  2]
        perm[0][2][5][2][5].map[1] = 1;
        perm[0][2][5][2][5].map[2] = 2;
        perm[0][2][5][2][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 4, [1  2  3]
        perm[0][2][5][3][0].map[1] = 1;
        perm[0][2][5][3][0].map[2] = 3;
        perm[0][2][5][3][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 4, [3  1  2]
        perm[0][2][5][3][1].map[1] = 3;
        perm[0][2][5][3][1].map[2] = 2;
        perm[0][2][5][3][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 4, [2  3  1]
        perm[0][2][5][3][2].map[1] = 2;
        perm[0][2][5][3][2].map[2] = 1;
        perm[0][2][5][3][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 4, [3  2  1]
        perm[0][2][5][3][3].map[1] = 3;
        perm[0][2][5][3][3].map[2] = 1;
        perm[0][2][5][3][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 4, [2  1  3]
        perm[0][2][5][3][4].map[1] = 2;
        perm[0][2][5][3][4].map[2] = 3;
        perm[0][2][5][3][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 4, [1  3  2]
        perm[0][2][5][3][5].map[1] = 1;
        perm[0][2][5][3][5].map[2] = 2;
        perm[0][2][5][3][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 1, [1  2  3]
        perm[0][3][0][0][0].map[1] = 1;
        perm[0][3][0][0][0].map[2] = 2;
        perm[0][3][0][0][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 1, [3  1  2]
        perm[0][3][0][0][1].map[1] = 3;
        perm[0][3][0][0][1].map[2] = 1;
        perm[0][3][0][0][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 1, [2  3  1]
        perm[0][3][0][0][2].map[1] = 2;
        perm[0][3][0][0][2].map[2] = 3;
        perm[0][3][0][0][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 1, [3  2  1]
        perm[0][3][0][0][3].map[1] = 3;
        perm[0][3][0][0][3].map[2] = 2;
        perm[0][3][0][0][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 1, [2  1  3]
        perm[0][3][0][0][4].map[1] = 2;
        perm[0][3][0][0][4].map[2] = 1;
        perm[0][3][0][0][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 1, [1  3  2]
        perm[0][3][0][0][5].map[1] = 1;
        perm[0][3][0][0][5].map[2] = 3;
        perm[0][3][0][0][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 2, [1  2  3]
        perm[0][3][0][1][0].map[1] = 1;
        perm[0][3][0][1][0].map[2] = 2;
        perm[0][3][0][1][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 2, [3  1  2]
        perm[0][3][0][1][1].map[1] = 3;
        perm[0][3][0][1][1].map[2] = 1;
        perm[0][3][0][1][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 2, [2  3  1]
        perm[0][3][0][1][2].map[1] = 2;
        perm[0][3][0][1][2].map[2] = 3;
        perm[0][3][0][1][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 2, [3  2  1]
        perm[0][3][0][1][3].map[1] = 3;
        perm[0][3][0][1][3].map[2] = 2;
        perm[0][3][0][1][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 2, [2  1  3]
        perm[0][3][0][1][4].map[1] = 2;
        perm[0][3][0][1][4].map[2] = 1;
        perm[0][3][0][1][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 2, [1  3  2]
        perm[0][3][0][1][5].map[1] = 1;
        perm[0][3][0][1][5].map[2] = 3;
        perm[0][3][0][1][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 3, [1  2  3]
        perm[0][3][0][2][0].map[1] = 1;
        perm[0][3][0][2][0].map[2] = 2;
        perm[0][3][0][2][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 3, [3  1  2]
        perm[0][3][0][2][1].map[1] = 3;
        perm[0][3][0][2][1].map[2] = 1;
        perm[0][3][0][2][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 3, [2  3  1]
        perm[0][3][0][2][2].map[1] = 2;
        perm[0][3][0][2][2].map[2] = 3;
        perm[0][3][0][2][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 3, [3  2  1]
        perm[0][3][0][2][3].map[1] = 3;
        perm[0][3][0][2][3].map[2] = 2;
        perm[0][3][0][2][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 3, [2  1  3]
        perm[0][3][0][2][4].map[1] = 2;
        perm[0][3][0][2][4].map[2] = 1;
        perm[0][3][0][2][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 3, [1  3  2]
        perm[0][3][0][2][5].map[1] = 1;
        perm[0][3][0][2][5].map[2] = 3;
        perm[0][3][0][2][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 4, [1  2  3]
        perm[0][3][0][3][0].map[1] = 1;
        perm[0][3][0][3][0].map[2] = 2;
        perm[0][3][0][3][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 4, [3  1  2]
        perm[0][3][0][3][1].map[1] = 3;
        perm[0][3][0][3][1].map[2] = 1;
        perm[0][3][0][3][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 4, [2  3  1]
        perm[0][3][0][3][2].map[1] = 2;
        perm[0][3][0][3][2].map[2] = 3;
        perm[0][3][0][3][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 4, [3  2  1]
        perm[0][3][0][3][3].map[1] = 3;
        perm[0][3][0][3][3].map[2] = 2;
        perm[0][3][0][3][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 4, [2  1  3]
        perm[0][3][0][3][4].map[1] = 2;
        perm[0][3][0][3][4].map[2] = 1;
        perm[0][3][0][3][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 4, [1  3  2]
        perm[0][3][0][3][5].map[1] = 1;
        perm[0][3][0][3][5].map[2] = 3;
        perm[0][3][0][3][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 1, [1  2  3]
        perm[0][3][1][0][0].map[1] = 2;
        perm[0][3][1][0][0].map[2] = 3;
        perm[0][3][1][0][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 1, [3  1  2]
        perm[0][3][1][0][1].map[1] = 1;
        perm[0][3][1][0][1].map[2] = 2;
        perm[0][3][1][0][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 1, [2  3  1]
        perm[0][3][1][0][2].map[1] = 3;
        perm[0][3][1][0][2].map[2] = 1;
        perm[0][3][1][0][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 1, [3  2  1]
        perm[0][3][1][0][3].map[1] = 2;
        perm[0][3][1][0][3].map[2] = 1;
        perm[0][3][1][0][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 1, [2  1  3]
        perm[0][3][1][0][4].map[1] = 1;
        perm[0][3][1][0][4].map[2] = 3;
        perm[0][3][1][0][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 1, [1  3  2]
        perm[0][3][1][0][5].map[1] = 3;
        perm[0][3][1][0][5].map[2] = 2;
        perm[0][3][1][0][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 2, [1  2  3]
        perm[0][3][1][1][0].map[1] = 2;
        perm[0][3][1][1][0].map[2] = 3;
        perm[0][3][1][1][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 2, [3  1  2]
        perm[0][3][1][1][1].map[1] = 1;
        perm[0][3][1][1][1].map[2] = 2;
        perm[0][3][1][1][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 2, [2  3  1]
        perm[0][3][1][1][2].map[1] = 3;
        perm[0][3][1][1][2].map[2] = 1;
        perm[0][3][1][1][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 2, [3  2  1]
        perm[0][3][1][1][3].map[1] = 2;
        perm[0][3][1][1][3].map[2] = 1;
        perm[0][3][1][1][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 2, [2  1  3]
        perm[0][3][1][1][4].map[1] = 1;
        perm[0][3][1][1][4].map[2] = 3;
        perm[0][3][1][1][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 2, [1  3  2]
        perm[0][3][1][1][5].map[1] = 3;
        perm[0][3][1][1][5].map[2] = 2;
        perm[0][3][1][1][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 3, [1  2  3]
        perm[0][3][1][2][0].map[1] = 2;
        perm[0][3][1][2][0].map[2] = 3;
        perm[0][3][1][2][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 3, [3  1  2]
        perm[0][3][1][2][1].map[1] = 1;
        perm[0][3][1][2][1].map[2] = 2;
        perm[0][3][1][2][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 3, [2  3  1]
        perm[0][3][1][2][2].map[1] = 3;
        perm[0][3][1][2][2].map[2] = 1;
        perm[0][3][1][2][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 3, [3  2  1]
        perm[0][3][1][2][3].map[1] = 2;
        perm[0][3][1][2][3].map[2] = 1;
        perm[0][3][1][2][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 3, [2  1  3]
        perm[0][3][1][2][4].map[1] = 1;
        perm[0][3][1][2][4].map[2] = 3;
        perm[0][3][1][2][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 3, [1  3  2]
        perm[0][3][1][2][5].map[1] = 3;
        perm[0][3][1][2][5].map[2] = 2;
        perm[0][3][1][2][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 4, [1  2  3]
        perm[0][3][1][3][0].map[1] = 2;
        perm[0][3][1][3][0].map[2] = 3;
        perm[0][3][1][3][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 4, [3  1  2]
        perm[0][3][1][3][1].map[1] = 1;
        perm[0][3][1][3][1].map[2] = 2;
        perm[0][3][1][3][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 4, [2  3  1]
        perm[0][3][1][3][2].map[1] = 3;
        perm[0][3][1][3][2].map[2] = 1;
        perm[0][3][1][3][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 4, [3  2  1]
        perm[0][3][1][3][3].map[1] = 2;
        perm[0][3][1][3][3].map[2] = 1;
        perm[0][3][1][3][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 4, [2  1  3]
        perm[0][3][1][3][4].map[1] = 1;
        perm[0][3][1][3][4].map[2] = 3;
        perm[0][3][1][3][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 4, [1  3  2]
        perm[0][3][1][3][5].map[1] = 3;
        perm[0][3][1][3][5].map[2] = 2;
        perm[0][3][1][3][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 1, [1  2  3]
        perm[0][3][2][0][0].map[1] = 3;
        perm[0][3][2][0][0].map[2] = 1;
        perm[0][3][2][0][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 1, [3  1  2]
        perm[0][3][2][0][1].map[1] = 2;
        perm[0][3][2][0][1].map[2] = 3;
        perm[0][3][2][0][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 1, [2  3  1]
        perm[0][3][2][0][2].map[1] = 1;
        perm[0][3][2][0][2].map[2] = 2;
        perm[0][3][2][0][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 1, [3  2  1]
        perm[0][3][2][0][3].map[1] = 1;
        perm[0][3][2][0][3].map[2] = 3;
        perm[0][3][2][0][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 1, [2  1  3]
        perm[0][3][2][0][4].map[1] = 3;
        perm[0][3][2][0][4].map[2] = 2;
        perm[0][3][2][0][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 1, [1  3  2]
        perm[0][3][2][0][5].map[1] = 2;
        perm[0][3][2][0][5].map[2] = 1;
        perm[0][3][2][0][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 2, [1  2  3]
        perm[0][3][2][1][0].map[1] = 3;
        perm[0][3][2][1][0].map[2] = 1;
        perm[0][3][2][1][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 2, [3  1  2]
        perm[0][3][2][1][1].map[1] = 2;
        perm[0][3][2][1][1].map[2] = 3;
        perm[0][3][2][1][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 2, [2  3  1]
        perm[0][3][2][1][2].map[1] = 1;
        perm[0][3][2][1][2].map[2] = 2;
        perm[0][3][2][1][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 2, [3  2  1]
        perm[0][3][2][1][3].map[1] = 1;
        perm[0][3][2][1][3].map[2] = 3;
        perm[0][3][2][1][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 2, [2  1  3]
        perm[0][3][2][1][4].map[1] = 3;
        perm[0][3][2][1][4].map[2] = 2;
        perm[0][3][2][1][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 2, [1  3  2]
        perm[0][3][2][1][5].map[1] = 2;
        perm[0][3][2][1][5].map[2] = 1;
        perm[0][3][2][1][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 3, [1  2  3]
        perm[0][3][2][2][0].map[1] = 3;
        perm[0][3][2][2][0].map[2] = 1;
        perm[0][3][2][2][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 3, [3  1  2]
        perm[0][3][2][2][1].map[1] = 2;
        perm[0][3][2][2][1].map[2] = 3;
        perm[0][3][2][2][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 3, [2  3  1]
        perm[0][3][2][2][2].map[1] = 1;
        perm[0][3][2][2][2].map[2] = 2;
        perm[0][3][2][2][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 3, [3  2  1]
        perm[0][3][2][2][3].map[1] = 1;
        perm[0][3][2][2][3].map[2] = 3;
        perm[0][3][2][2][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 3, [2  1  3]
        perm[0][3][2][2][4].map[1] = 3;
        perm[0][3][2][2][4].map[2] = 2;
        perm[0][3][2][2][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 3, [1  3  2]
        perm[0][3][2][2][5].map[1] = 2;
        perm[0][3][2][2][5].map[2] = 1;
        perm[0][3][2][2][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 4, [1  2  3]
        perm[0][3][2][3][0].map[1] = 3;
        perm[0][3][2][3][0].map[2] = 1;
        perm[0][3][2][3][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 4, [3  1  2]
        perm[0][3][2][3][1].map[1] = 2;
        perm[0][3][2][3][1].map[2] = 3;
        perm[0][3][2][3][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 4, [2  3  1]
        perm[0][3][2][3][2].map[1] = 1;
        perm[0][3][2][3][2].map[2] = 2;
        perm[0][3][2][3][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 4, [3  2  1]
        perm[0][3][2][3][3].map[1] = 1;
        perm[0][3][2][3][3].map[2] = 3;
        perm[0][3][2][3][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 4, [2  1  3]
        perm[0][3][2][3][4].map[1] = 3;
        perm[0][3][2][3][4].map[2] = 2;
        perm[0][3][2][3][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 4, [1  3  2]
        perm[0][3][2][3][5].map[1] = 2;
        perm[0][3][2][3][5].map[2] = 1;
        perm[0][3][2][3][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 1, [1  2  3]
        perm[0][3][3][0][0].map[1] = 3;
        perm[0][3][3][0][0].map[2] = 2;
        perm[0][3][3][0][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 1, [3  1  2]
        perm[0][3][3][0][1].map[1] = 2;
        perm[0][3][3][0][1].map[2] = 1;
        perm[0][3][3][0][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 1, [2  3  1]
        perm[0][3][3][0][2].map[1] = 1;
        perm[0][3][3][0][2].map[2] = 3;
        perm[0][3][3][0][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 1, [3  2  1]
        perm[0][3][3][0][3].map[1] = 1;
        perm[0][3][3][0][3].map[2] = 2;
        perm[0][3][3][0][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 1, [2  1  3]
        perm[0][3][3][0][4].map[1] = 3;
        perm[0][3][3][0][4].map[2] = 1;
        perm[0][3][3][0][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 1, [1  3  2]
        perm[0][3][3][0][5].map[1] = 2;
        perm[0][3][3][0][5].map[2] = 3;
        perm[0][3][3][0][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 2, [1  2  3]
        perm[0][3][3][1][0].map[1] = 3;
        perm[0][3][3][1][0].map[2] = 2;
        perm[0][3][3][1][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 2, [3  1  2]
        perm[0][3][3][1][1].map[1] = 2;
        perm[0][3][3][1][1].map[2] = 1;
        perm[0][3][3][1][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 2, [2  3  1]
        perm[0][3][3][1][2].map[1] = 1;
        perm[0][3][3][1][2].map[2] = 3;
        perm[0][3][3][1][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 2, [3  2  1]
        perm[0][3][3][1][3].map[1] = 1;
        perm[0][3][3][1][3].map[2] = 2;
        perm[0][3][3][1][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 2, [2  1  3]
        perm[0][3][3][1][4].map[1] = 3;
        perm[0][3][3][1][4].map[2] = 1;
        perm[0][3][3][1][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 2, [1  3  2]
        perm[0][3][3][1][5].map[1] = 2;
        perm[0][3][3][1][5].map[2] = 3;
        perm[0][3][3][1][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 3, [1  2  3]
        perm[0][3][3][2][0].map[1] = 3;
        perm[0][3][3][2][0].map[2] = 2;
        perm[0][3][3][2][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 3, [3  1  2]
        perm[0][3][3][2][1].map[1] = 2;
        perm[0][3][3][2][1].map[2] = 1;
        perm[0][3][3][2][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 3, [2  3  1]
        perm[0][3][3][2][2].map[1] = 1;
        perm[0][3][3][2][2].map[2] = 3;
        perm[0][3][3][2][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 3, [3  2  1]
        perm[0][3][3][2][3].map[1] = 1;
        perm[0][3][3][2][3].map[2] = 2;
        perm[0][3][3][2][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 3, [2  1  3]
        perm[0][3][3][2][4].map[1] = 3;
        perm[0][3][3][2][4].map[2] = 1;
        perm[0][3][3][2][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 3, [1  3  2]
        perm[0][3][3][2][5].map[1] = 2;
        perm[0][3][3][2][5].map[2] = 3;
        perm[0][3][3][2][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 4, [1  2  3]
        perm[0][3][3][3][0].map[1] = 3;
        perm[0][3][3][3][0].map[2] = 2;
        perm[0][3][3][3][0].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 4, [3  1  2]
        perm[0][3][3][3][1].map[1] = 2;
        perm[0][3][3][3][1].map[2] = 1;
        perm[0][3][3][3][1].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 4, [2  3  1]
        perm[0][3][3][3][2].map[1] = 1;
        perm[0][3][3][3][2].map[2] = 3;
        perm[0][3][3][3][2].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 4, [3  2  1]
        perm[0][3][3][3][3].map[1] = 1;
        perm[0][3][3][3][3].map[2] = 2;
        perm[0][3][3][3][3].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 4, [2  1  3]
        perm[0][3][3][3][4].map[1] = 3;
        perm[0][3][3][3][4].map[2] = 1;
        perm[0][3][3][3][4].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 4, [1  3  2]
        perm[0][3][3][3][5].map[1] = 2;
        perm[0][3][3][3][5].map[2] = 3;
        perm[0][3][3][3][5].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 1, [1  2  3]
        perm[0][3][4][0][0].map[1] = 2;
        perm[0][3][4][0][0].map[2] = 1;
        perm[0][3][4][0][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 1, [3  1  2]
        perm[0][3][4][0][1].map[1] = 1;
        perm[0][3][4][0][1].map[2] = 3;
        perm[0][3][4][0][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 1, [2  3  1]
        perm[0][3][4][0][2].map[1] = 3;
        perm[0][3][4][0][2].map[2] = 2;
        perm[0][3][4][0][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 1, [3  2  1]
        perm[0][3][4][0][3].map[1] = 2;
        perm[0][3][4][0][3].map[2] = 3;
        perm[0][3][4][0][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 1, [2  1  3]
        perm[0][3][4][0][4].map[1] = 1;
        perm[0][3][4][0][4].map[2] = 2;
        perm[0][3][4][0][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 1, [1  3  2]
        perm[0][3][4][0][5].map[1] = 3;
        perm[0][3][4][0][5].map[2] = 1;
        perm[0][3][4][0][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 2, [1  2  3]
        perm[0][3][4][1][0].map[1] = 2;
        perm[0][3][4][1][0].map[2] = 1;
        perm[0][3][4][1][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 2, [3  1  2]
        perm[0][3][4][1][1].map[1] = 1;
        perm[0][3][4][1][1].map[2] = 3;
        perm[0][3][4][1][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 2, [2  3  1]
        perm[0][3][4][1][2].map[1] = 3;
        perm[0][3][4][1][2].map[2] = 2;
        perm[0][3][4][1][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 2, [3  2  1]
        perm[0][3][4][1][3].map[1] = 2;
        perm[0][3][4][1][3].map[2] = 3;
        perm[0][3][4][1][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 2, [2  1  3]
        perm[0][3][4][1][4].map[1] = 1;
        perm[0][3][4][1][4].map[2] = 2;
        perm[0][3][4][1][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 2, [1  3  2]
        perm[0][3][4][1][5].map[1] = 3;
        perm[0][3][4][1][5].map[2] = 1;
        perm[0][3][4][1][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 3, [1  2  3]
        perm[0][3][4][2][0].map[1] = 2;
        perm[0][3][4][2][0].map[2] = 1;
        perm[0][3][4][2][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 3, [3  1  2]
        perm[0][3][4][2][1].map[1] = 1;
        perm[0][3][4][2][1].map[2] = 3;
        perm[0][3][4][2][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 3, [2  3  1]
        perm[0][3][4][2][2].map[1] = 3;
        perm[0][3][4][2][2].map[2] = 2;
        perm[0][3][4][2][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 3, [3  2  1]
        perm[0][3][4][2][3].map[1] = 2;
        perm[0][3][4][2][3].map[2] = 3;
        perm[0][3][4][2][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 3, [2  1  3]
        perm[0][3][4][2][4].map[1] = 1;
        perm[0][3][4][2][4].map[2] = 2;
        perm[0][3][4][2][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 3, [1  3  2]
        perm[0][3][4][2][5].map[1] = 3;
        perm[0][3][4][2][5].map[2] = 1;
        perm[0][3][4][2][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 4, [1  2  3]
        perm[0][3][4][3][0].map[1] = 2;
        perm[0][3][4][3][0].map[2] = 1;
        perm[0][3][4][3][0].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 4, [3  1  2]
        perm[0][3][4][3][1].map[1] = 1;
        perm[0][3][4][3][1].map[2] = 3;
        perm[0][3][4][3][1].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 4, [2  3  1]
        perm[0][3][4][3][2].map[1] = 3;
        perm[0][3][4][3][2].map[2] = 2;
        perm[0][3][4][3][2].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 4, [3  2  1]
        perm[0][3][4][3][3].map[1] = 2;
        perm[0][3][4][3][3].map[2] = 3;
        perm[0][3][4][3][3].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 4, [2  1  3]
        perm[0][3][4][3][4].map[1] = 1;
        perm[0][3][4][3][4].map[2] = 2;
        perm[0][3][4][3][4].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 4, [1  3  2]
        perm[0][3][4][3][5].map[1] = 3;
        perm[0][3][4][3][5].map[2] = 1;
        perm[0][3][4][3][5].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 1, [1  2  3]
        perm[0][3][5][0][0].map[1] = 1;
        perm[0][3][5][0][0].map[2] = 3;
        perm[0][3][5][0][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 1, [3  1  2]
        perm[0][3][5][0][1].map[1] = 3;
        perm[0][3][5][0][1].map[2] = 2;
        perm[0][3][5][0][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 1, [2  3  1]
        perm[0][3][5][0][2].map[1] = 2;
        perm[0][3][5][0][2].map[2] = 1;
        perm[0][3][5][0][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 1, [3  2  1]
        perm[0][3][5][0][3].map[1] = 3;
        perm[0][3][5][0][3].map[2] = 1;
        perm[0][3][5][0][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 1, [2  1  3]
        perm[0][3][5][0][4].map[1] = 2;
        perm[0][3][5][0][4].map[2] = 3;
        perm[0][3][5][0][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 1, [1  3  2]
        perm[0][3][5][0][5].map[1] = 1;
        perm[0][3][5][0][5].map[2] = 2;
        perm[0][3][5][0][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 2, [1  2  3]
        perm[0][3][5][1][0].map[1] = 1;
        perm[0][3][5][1][0].map[2] = 3;
        perm[0][3][5][1][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 2, [3  1  2]
        perm[0][3][5][1][1].map[1] = 3;
        perm[0][3][5][1][1].map[2] = 2;
        perm[0][3][5][1][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 2, [2  3  1]
        perm[0][3][5][1][2].map[1] = 2;
        perm[0][3][5][1][2].map[2] = 1;
        perm[0][3][5][1][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 2, [3  2  1]
        perm[0][3][5][1][3].map[1] = 3;
        perm[0][3][5][1][3].map[2] = 1;
        perm[0][3][5][1][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 2, [2  1  3]
        perm[0][3][5][1][4].map[1] = 2;
        perm[0][3][5][1][4].map[2] = 3;
        perm[0][3][5][1][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 2, [1  3  2]
        perm[0][3][5][1][5].map[1] = 1;
        perm[0][3][5][1][5].map[2] = 2;
        perm[0][3][5][1][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 3, [1  2  3]
        perm[0][3][5][2][0].map[1] = 1;
        perm[0][3][5][2][0].map[2] = 3;
        perm[0][3][5][2][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 3, [3  1  2]
        perm[0][3][5][2][1].map[1] = 3;
        perm[0][3][5][2][1].map[2] = 2;
        perm[0][3][5][2][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 3, [2  3  1]
        perm[0][3][5][2][2].map[1] = 2;
        perm[0][3][5][2][2].map[2] = 1;
        perm[0][3][5][2][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 3, [3  2  1]
        perm[0][3][5][2][3].map[1] = 3;
        perm[0][3][5][2][3].map[2] = 1;
        perm[0][3][5][2][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 3, [2  1  3]
        perm[0][3][5][2][4].map[1] = 2;
        perm[0][3][5][2][4].map[2] = 3;
        perm[0][3][5][2][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 3, [1  3  2]
        perm[0][3][5][2][5].map[1] = 1;
        perm[0][3][5][2][5].map[2] = 2;
        perm[0][3][5][2][5].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 4, [1  2  3]
        perm[0][3][5][3][0].map[1] = 1;
        perm[0][3][5][3][0].map[2] = 3;
        perm[0][3][5][3][0].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 4, [3  1  2]
        perm[0][3][5][3][1].map[1] = 3;
        perm[0][3][5][3][1].map[2] = 2;
        perm[0][3][5][3][1].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 4, [2  3  1]
        perm[0][3][5][3][2].map[1] = 2;
        perm[0][3][5][3][2].map[2] = 1;
        perm[0][3][5][3][2].map[3] = 3;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 4, [3  2  1]
        perm[0][3][5][3][3].map[1] = 3;
        perm[0][3][5][3][3].map[2] = 1;
        perm[0][3][5][3][3].map[3] = 2;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 4, [2  1  3]
        perm[0][3][5][3][4].map[1] = 2;
        perm[0][3][5][3][4].map[2] = 3;
        perm[0][3][5][3][4].map[3] = 1;

        // DoF set #1, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 4, [1  3  2]
        perm[0][3][5][3][5].map[1] = 1;
        perm[0][3][5][3][5].map[2] = 2;
        perm[0][3][5][3][5].map[3] = 3;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 1, [1  2  3]
        perm[1][0][0][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 1, [3  1  2]
        perm[1][0][0][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 1, [2  3  1]
        perm[1][0][0][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 1, [3  2  1]
        perm[1][0][0][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 1, [2  1  3]
        perm[1][0][0][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 1, [1  3  2]
        perm[1][0][0][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 2, [1  2  3]
        perm[1][0][0][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 2, [3  1  2]
        perm[1][0][0][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 2, [2  3  1]
        perm[1][0][0][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 2, [3  2  1]
        perm[1][0][0][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 2, [2  1  3]
        perm[1][0][0][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 2, [1  3  2]
        perm[1][0][0][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 3, [1  2  3]
        perm[1][0][0][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 3, [3  1  2]
        perm[1][0][0][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 3, [2  3  1]
        perm[1][0][0][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 3, [3  2  1]
        perm[1][0][0][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 3, [2  1  3]
        perm[1][0][0][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 3, [1  3  2]
        perm[1][0][0][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 4, [1  2  3]
        perm[1][0][0][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 4, [3  1  2]
        perm[1][0][0][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 4, [2  3  1]
        perm[1][0][0][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 4, [3  2  1]
        perm[1][0][0][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 4, [2  1  3]
        perm[1][0][0][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  2  3] ---> (init: face, perm) 4, [1  3  2]
        perm[1][0][0][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 1, [1  2  3]
        perm[1][0][1][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 1, [3  1  2]
        perm[1][0][1][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 1, [2  3  1]
        perm[1][0][1][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 1, [3  2  1]
        perm[1][0][1][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 1, [2  1  3]
        perm[1][0][1][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 1, [1  3  2]
        perm[1][0][1][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 2, [1  2  3]
        perm[1][0][1][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 2, [3  1  2]
        perm[1][0][1][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 2, [2  3  1]
        perm[1][0][1][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 2, [3  2  1]
        perm[1][0][1][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 2, [2  1  3]
        perm[1][0][1][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 2, [1  3  2]
        perm[1][0][1][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 3, [1  2  3]
        perm[1][0][1][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 3, [3  1  2]
        perm[1][0][1][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 3, [2  3  1]
        perm[1][0][1][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 3, [3  2  1]
        perm[1][0][1][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 3, [2  1  3]
        perm[1][0][1][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 3, [1  3  2]
        perm[1][0][1][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 4, [1  2  3]
        perm[1][0][1][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 4, [3  1  2]
        perm[1][0][1][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 4, [2  3  1]
        perm[1][0][1][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 4, [3  2  1]
        perm[1][0][1][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 4, [2  1  3]
        perm[1][0][1][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  1  2] ---> (init: face, perm) 4, [1  3  2]
        perm[1][0][1][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 1, [1  2  3]
        perm[1][0][2][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 1, [3  1  2]
        perm[1][0][2][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 1, [2  3  1]
        perm[1][0][2][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 1, [3  2  1]
        perm[1][0][2][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 1, [2  1  3]
        perm[1][0][2][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 1, [1  3  2]
        perm[1][0][2][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 2, [1  2  3]
        perm[1][0][2][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 2, [3  1  2]
        perm[1][0][2][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 2, [2  3  1]
        perm[1][0][2][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 2, [3  2  1]
        perm[1][0][2][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 2, [2  1  3]
        perm[1][0][2][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 2, [1  3  2]
        perm[1][0][2][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 3, [1  2  3]
        perm[1][0][2][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 3, [3  1  2]
        perm[1][0][2][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 3, [2  3  1]
        perm[1][0][2][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 3, [3  2  1]
        perm[1][0][2][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 3, [2  1  3]
        perm[1][0][2][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 3, [1  3  2]
        perm[1][0][2][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 4, [1  2  3]
        perm[1][0][2][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 4, [3  1  2]
        perm[1][0][2][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 4, [2  3  1]
        perm[1][0][2][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 4, [3  2  1]
        perm[1][0][2][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 4, [2  1  3]
        perm[1][0][2][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  3  1] ---> (init: face, perm) 4, [1  3  2]
        perm[1][0][2][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 1, [1  2  3]
        perm[1][0][3][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 1, [3  1  2]
        perm[1][0][3][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 1, [2  3  1]
        perm[1][0][3][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 1, [3  2  1]
        perm[1][0][3][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 1, [2  1  3]
        perm[1][0][3][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 1, [1  3  2]
        perm[1][0][3][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 2, [1  2  3]
        perm[1][0][3][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 2, [3  1  2]
        perm[1][0][3][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 2, [2  3  1]
        perm[1][0][3][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 2, [3  2  1]
        perm[1][0][3][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 2, [2  1  3]
        perm[1][0][3][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 2, [1  3  2]
        perm[1][0][3][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 3, [1  2  3]
        perm[1][0][3][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 3, [3  1  2]
        perm[1][0][3][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 3, [2  3  1]
        perm[1][0][3][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 3, [3  2  1]
        perm[1][0][3][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 3, [2  1  3]
        perm[1][0][3][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 3, [1  3  2]
        perm[1][0][3][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 4, [1  2  3]
        perm[1][0][3][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 4, [3  1  2]
        perm[1][0][3][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 4, [2  3  1]
        perm[1][0][3][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 4, [3  2  1]
        perm[1][0][3][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 4, [2  1  3]
        perm[1][0][3][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [3  2  1] ---> (init: face, perm) 4, [1  3  2]
        perm[1][0][3][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 1, [1  2  3]
        perm[1][0][4][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 1, [3  1  2]
        perm[1][0][4][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 1, [2  3  1]
        perm[1][0][4][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 1, [3  2  1]
        perm[1][0][4][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 1, [2  1  3]
        perm[1][0][4][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 1, [1  3  2]
        perm[1][0][4][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 2, [1  2  3]
        perm[1][0][4][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 2, [3  1  2]
        perm[1][0][4][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 2, [2  3  1]
        perm[1][0][4][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 2, [3  2  1]
        perm[1][0][4][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 2, [2  1  3]
        perm[1][0][4][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 2, [1  3  2]
        perm[1][0][4][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 3, [1  2  3]
        perm[1][0][4][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 3, [3  1  2]
        perm[1][0][4][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 3, [2  3  1]
        perm[1][0][4][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 3, [3  2  1]
        perm[1][0][4][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 3, [2  1  3]
        perm[1][0][4][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 3, [1  3  2]
        perm[1][0][4][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 4, [1  2  3]
        perm[1][0][4][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 4, [3  1  2]
        perm[1][0][4][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 4, [2  3  1]
        perm[1][0][4][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 4, [3  2  1]
        perm[1][0][4][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 4, [2  1  3]
        perm[1][0][4][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [2  1  3] ---> (init: face, perm) 4, [1  3  2]
        perm[1][0][4][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 1, [1  2  3]
        perm[1][0][5][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 1, [3  1  2]
        perm[1][0][5][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 1, [2  3  1]
        perm[1][0][5][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 1, [3  2  1]
        perm[1][0][5][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 1, [2  1  3]
        perm[1][0][5][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 1, [1  3  2]
        perm[1][0][5][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 2, [1  2  3]
        perm[1][0][5][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 2, [3  1  2]
        perm[1][0][5][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 2, [2  3  1]
        perm[1][0][5][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 2, [3  2  1]
        perm[1][0][5][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 2, [2  1  3]
        perm[1][0][5][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 2, [1  3  2]
        perm[1][0][5][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 3, [1  2  3]
        perm[1][0][5][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 3, [3  1  2]
        perm[1][0][5][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 3, [2  3  1]
        perm[1][0][5][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 3, [3  2  1]
        perm[1][0][5][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 3, [2  1  3]
        perm[1][0][5][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 3, [1  3  2]
        perm[1][0][5][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 4, [1  2  3]
        perm[1][0][5][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 4, [3  1  2]
        perm[1][0][5][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 4, [2  3  1]
        perm[1][0][5][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 4, [3  2  1]
        perm[1][0][5][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 4, [2  1  3]
        perm[1][0][5][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 1, [1  3  2] ---> (init: face, perm) 4, [1  3  2]
        perm[1][0][5][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 1, [1  2  3]
        perm[1][1][0][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 1, [3  1  2]
        perm[1][1][0][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 1, [2  3  1]
        perm[1][1][0][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 1, [3  2  1]
        perm[1][1][0][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 1, [2  1  3]
        perm[1][1][0][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 1, [1  3  2]
        perm[1][1][0][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 2, [1  2  3]
        perm[1][1][0][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 2, [3  1  2]
        perm[1][1][0][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 2, [2  3  1]
        perm[1][1][0][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 2, [3  2  1]
        perm[1][1][0][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 2, [2  1  3]
        perm[1][1][0][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 2, [1  3  2]
        perm[1][1][0][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 3, [1  2  3]
        perm[1][1][0][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 3, [3  1  2]
        perm[1][1][0][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 3, [2  3  1]
        perm[1][1][0][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 3, [3  2  1]
        perm[1][1][0][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 3, [2  1  3]
        perm[1][1][0][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 3, [1  3  2]
        perm[1][1][0][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 4, [1  2  3]
        perm[1][1][0][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 4, [3  1  2]
        perm[1][1][0][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 4, [2  3  1]
        perm[1][1][0][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 4, [3  2  1]
        perm[1][1][0][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 4, [2  1  3]
        perm[1][1][0][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  2  3] ---> (init: face, perm) 4, [1  3  2]
        perm[1][1][0][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 1, [1  2  3]
        perm[1][1][1][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 1, [3  1  2]
        perm[1][1][1][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 1, [2  3  1]
        perm[1][1][1][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 1, [3  2  1]
        perm[1][1][1][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 1, [2  1  3]
        perm[1][1][1][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 1, [1  3  2]
        perm[1][1][1][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 2, [1  2  3]
        perm[1][1][1][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 2, [3  1  2]
        perm[1][1][1][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 2, [2  3  1]
        perm[1][1][1][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 2, [3  2  1]
        perm[1][1][1][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 2, [2  1  3]
        perm[1][1][1][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 2, [1  3  2]
        perm[1][1][1][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 3, [1  2  3]
        perm[1][1][1][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 3, [3  1  2]
        perm[1][1][1][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 3, [2  3  1]
        perm[1][1][1][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 3, [3  2  1]
        perm[1][1][1][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 3, [2  1  3]
        perm[1][1][1][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 3, [1  3  2]
        perm[1][1][1][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 4, [1  2  3]
        perm[1][1][1][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 4, [3  1  2]
        perm[1][1][1][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 4, [2  3  1]
        perm[1][1][1][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 4, [3  2  1]
        perm[1][1][1][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 4, [2  1  3]
        perm[1][1][1][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  1  2] ---> (init: face, perm) 4, [1  3  2]
        perm[1][1][1][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 1, [1  2  3]
        perm[1][1][2][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 1, [3  1  2]
        perm[1][1][2][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 1, [2  3  1]
        perm[1][1][2][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 1, [3  2  1]
        perm[1][1][2][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 1, [2  1  3]
        perm[1][1][2][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 1, [1  3  2]
        perm[1][1][2][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 2, [1  2  3]
        perm[1][1][2][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 2, [3  1  2]
        perm[1][1][2][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 2, [2  3  1]
        perm[1][1][2][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 2, [3  2  1]
        perm[1][1][2][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 2, [2  1  3]
        perm[1][1][2][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 2, [1  3  2]
        perm[1][1][2][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 3, [1  2  3]
        perm[1][1][2][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 3, [3  1  2]
        perm[1][1][2][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 3, [2  3  1]
        perm[1][1][2][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 3, [3  2  1]
        perm[1][1][2][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 3, [2  1  3]
        perm[1][1][2][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 3, [1  3  2]
        perm[1][1][2][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 4, [1  2  3]
        perm[1][1][2][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 4, [3  1  2]
        perm[1][1][2][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 4, [2  3  1]
        perm[1][1][2][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 4, [3  2  1]
        perm[1][1][2][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 4, [2  1  3]
        perm[1][1][2][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  3  1] ---> (init: face, perm) 4, [1  3  2]
        perm[1][1][2][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 1, [1  2  3]
        perm[1][1][3][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 1, [3  1  2]
        perm[1][1][3][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 1, [2  3  1]
        perm[1][1][3][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 1, [3  2  1]
        perm[1][1][3][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 1, [2  1  3]
        perm[1][1][3][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 1, [1  3  2]
        perm[1][1][3][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 2, [1  2  3]
        perm[1][1][3][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 2, [3  1  2]
        perm[1][1][3][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 2, [2  3  1]
        perm[1][1][3][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 2, [3  2  1]
        perm[1][1][3][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 2, [2  1  3]
        perm[1][1][3][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 2, [1  3  2]
        perm[1][1][3][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 3, [1  2  3]
        perm[1][1][3][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 3, [3  1  2]
        perm[1][1][3][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 3, [2  3  1]
        perm[1][1][3][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 3, [3  2  1]
        perm[1][1][3][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 3, [2  1  3]
        perm[1][1][3][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 3, [1  3  2]
        perm[1][1][3][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 4, [1  2  3]
        perm[1][1][3][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 4, [3  1  2]
        perm[1][1][3][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 4, [2  3  1]
        perm[1][1][3][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 4, [3  2  1]
        perm[1][1][3][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 4, [2  1  3]
        perm[1][1][3][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [3  2  1] ---> (init: face, perm) 4, [1  3  2]
        perm[1][1][3][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 1, [1  2  3]
        perm[1][1][4][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 1, [3  1  2]
        perm[1][1][4][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 1, [2  3  1]
        perm[1][1][4][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 1, [3  2  1]
        perm[1][1][4][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 1, [2  1  3]
        perm[1][1][4][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 1, [1  3  2]
        perm[1][1][4][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 2, [1  2  3]
        perm[1][1][4][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 2, [3  1  2]
        perm[1][1][4][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 2, [2  3  1]
        perm[1][1][4][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 2, [3  2  1]
        perm[1][1][4][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 2, [2  1  3]
        perm[1][1][4][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 2, [1  3  2]
        perm[1][1][4][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 3, [1  2  3]
        perm[1][1][4][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 3, [3  1  2]
        perm[1][1][4][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 3, [2  3  1]
        perm[1][1][4][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 3, [3  2  1]
        perm[1][1][4][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 3, [2  1  3]
        perm[1][1][4][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 3, [1  3  2]
        perm[1][1][4][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 4, [1  2  3]
        perm[1][1][4][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 4, [3  1  2]
        perm[1][1][4][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 4, [2  3  1]
        perm[1][1][4][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 4, [3  2  1]
        perm[1][1][4][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 4, [2  1  3]
        perm[1][1][4][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [2  1  3] ---> (init: face, perm) 4, [1  3  2]
        perm[1][1][4][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 1, [1  2  3]
        perm[1][1][5][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 1, [3  1  2]
        perm[1][1][5][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 1, [2  3  1]
        perm[1][1][5][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 1, [3  2  1]
        perm[1][1][5][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 1, [2  1  3]
        perm[1][1][5][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 1, [1  3  2]
        perm[1][1][5][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 2, [1  2  3]
        perm[1][1][5][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 2, [3  1  2]
        perm[1][1][5][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 2, [2  3  1]
        perm[1][1][5][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 2, [3  2  1]
        perm[1][1][5][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 2, [2  1  3]
        perm[1][1][5][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 2, [1  3  2]
        perm[1][1][5][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 3, [1  2  3]
        perm[1][1][5][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 3, [3  1  2]
        perm[1][1][5][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 3, [2  3  1]
        perm[1][1][5][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 3, [3  2  1]
        perm[1][1][5][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 3, [2  1  3]
        perm[1][1][5][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 3, [1  3  2]
        perm[1][1][5][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 4, [1  2  3]
        perm[1][1][5][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 4, [3  1  2]
        perm[1][1][5][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 4, [2  3  1]
        perm[1][1][5][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 4, [3  2  1]
        perm[1][1][5][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 4, [2  1  3]
        perm[1][1][5][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 2, [1  3  2] ---> (init: face, perm) 4, [1  3  2]
        perm[1][1][5][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 1, [1  2  3]
        perm[1][2][0][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 1, [3  1  2]
        perm[1][2][0][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 1, [2  3  1]
        perm[1][2][0][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 1, [3  2  1]
        perm[1][2][0][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 1, [2  1  3]
        perm[1][2][0][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 1, [1  3  2]
        perm[1][2][0][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 2, [1  2  3]
        perm[1][2][0][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 2, [3  1  2]
        perm[1][2][0][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 2, [2  3  1]
        perm[1][2][0][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 2, [3  2  1]
        perm[1][2][0][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 2, [2  1  3]
        perm[1][2][0][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 2, [1  3  2]
        perm[1][2][0][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 3, [1  2  3]
        perm[1][2][0][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 3, [3  1  2]
        perm[1][2][0][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 3, [2  3  1]
        perm[1][2][0][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 3, [3  2  1]
        perm[1][2][0][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 3, [2  1  3]
        perm[1][2][0][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 3, [1  3  2]
        perm[1][2][0][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 4, [1  2  3]
        perm[1][2][0][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 4, [3  1  2]
        perm[1][2][0][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 4, [2  3  1]
        perm[1][2][0][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 4, [3  2  1]
        perm[1][2][0][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 4, [2  1  3]
        perm[1][2][0][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  2  3] ---> (init: face, perm) 4, [1  3  2]
        perm[1][2][0][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 1, [1  2  3]
        perm[1][2][1][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 1, [3  1  2]
        perm[1][2][1][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 1, [2  3  1]
        perm[1][2][1][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 1, [3  2  1]
        perm[1][2][1][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 1, [2  1  3]
        perm[1][2][1][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 1, [1  3  2]
        perm[1][2][1][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 2, [1  2  3]
        perm[1][2][1][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 2, [3  1  2]
        perm[1][2][1][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 2, [2  3  1]
        perm[1][2][1][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 2, [3  2  1]
        perm[1][2][1][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 2, [2  1  3]
        perm[1][2][1][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 2, [1  3  2]
        perm[1][2][1][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 3, [1  2  3]
        perm[1][2][1][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 3, [3  1  2]
        perm[1][2][1][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 3, [2  3  1]
        perm[1][2][1][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 3, [3  2  1]
        perm[1][2][1][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 3, [2  1  3]
        perm[1][2][1][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 3, [1  3  2]
        perm[1][2][1][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 4, [1  2  3]
        perm[1][2][1][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 4, [3  1  2]
        perm[1][2][1][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 4, [2  3  1]
        perm[1][2][1][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 4, [3  2  1]
        perm[1][2][1][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 4, [2  1  3]
        perm[1][2][1][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  1  2] ---> (init: face, perm) 4, [1  3  2]
        perm[1][2][1][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 1, [1  2  3]
        perm[1][2][2][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 1, [3  1  2]
        perm[1][2][2][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 1, [2  3  1]
        perm[1][2][2][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 1, [3  2  1]
        perm[1][2][2][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 1, [2  1  3]
        perm[1][2][2][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 1, [1  3  2]
        perm[1][2][2][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 2, [1  2  3]
        perm[1][2][2][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 2, [3  1  2]
        perm[1][2][2][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 2, [2  3  1]
        perm[1][2][2][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 2, [3  2  1]
        perm[1][2][2][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 2, [2  1  3]
        perm[1][2][2][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 2, [1  3  2]
        perm[1][2][2][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 3, [1  2  3]
        perm[1][2][2][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 3, [3  1  2]
        perm[1][2][2][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 3, [2  3  1]
        perm[1][2][2][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 3, [3  2  1]
        perm[1][2][2][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 3, [2  1  3]
        perm[1][2][2][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 3, [1  3  2]
        perm[1][2][2][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 4, [1  2  3]
        perm[1][2][2][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 4, [3  1  2]
        perm[1][2][2][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 4, [2  3  1]
        perm[1][2][2][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 4, [3  2  1]
        perm[1][2][2][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 4, [2  1  3]
        perm[1][2][2][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  3  1] ---> (init: face, perm) 4, [1  3  2]
        perm[1][2][2][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 1, [1  2  3]
        perm[1][2][3][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 1, [3  1  2]
        perm[1][2][3][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 1, [2  3  1]
        perm[1][2][3][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 1, [3  2  1]
        perm[1][2][3][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 1, [2  1  3]
        perm[1][2][3][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 1, [1  3  2]
        perm[1][2][3][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 2, [1  2  3]
        perm[1][2][3][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 2, [3  1  2]
        perm[1][2][3][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 2, [2  3  1]
        perm[1][2][3][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 2, [3  2  1]
        perm[1][2][3][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 2, [2  1  3]
        perm[1][2][3][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 2, [1  3  2]
        perm[1][2][3][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 3, [1  2  3]
        perm[1][2][3][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 3, [3  1  2]
        perm[1][2][3][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 3, [2  3  1]
        perm[1][2][3][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 3, [3  2  1]
        perm[1][2][3][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 3, [2  1  3]
        perm[1][2][3][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 3, [1  3  2]
        perm[1][2][3][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 4, [1  2  3]
        perm[1][2][3][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 4, [3  1  2]
        perm[1][2][3][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 4, [2  3  1]
        perm[1][2][3][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 4, [3  2  1]
        perm[1][2][3][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 4, [2  1  3]
        perm[1][2][3][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [3  2  1] ---> (init: face, perm) 4, [1  3  2]
        perm[1][2][3][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 1, [1  2  3]
        perm[1][2][4][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 1, [3  1  2]
        perm[1][2][4][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 1, [2  3  1]
        perm[1][2][4][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 1, [3  2  1]
        perm[1][2][4][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 1, [2  1  3]
        perm[1][2][4][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 1, [1  3  2]
        perm[1][2][4][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 2, [1  2  3]
        perm[1][2][4][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 2, [3  1  2]
        perm[1][2][4][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 2, [2  3  1]
        perm[1][2][4][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 2, [3  2  1]
        perm[1][2][4][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 2, [2  1  3]
        perm[1][2][4][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 2, [1  3  2]
        perm[1][2][4][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 3, [1  2  3]
        perm[1][2][4][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 3, [3  1  2]
        perm[1][2][4][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 3, [2  3  1]
        perm[1][2][4][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 3, [3  2  1]
        perm[1][2][4][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 3, [2  1  3]
        perm[1][2][4][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 3, [1  3  2]
        perm[1][2][4][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 4, [1  2  3]
        perm[1][2][4][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 4, [3  1  2]
        perm[1][2][4][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 4, [2  3  1]
        perm[1][2][4][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 4, [3  2  1]
        perm[1][2][4][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 4, [2  1  3]
        perm[1][2][4][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [2  1  3] ---> (init: face, perm) 4, [1  3  2]
        perm[1][2][4][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 1, [1  2  3]
        perm[1][2][5][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 1, [3  1  2]
        perm[1][2][5][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 1, [2  3  1]
        perm[1][2][5][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 1, [3  2  1]
        perm[1][2][5][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 1, [2  1  3]
        perm[1][2][5][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 1, [1  3  2]
        perm[1][2][5][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 2, [1  2  3]
        perm[1][2][5][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 2, [3  1  2]
        perm[1][2][5][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 2, [2  3  1]
        perm[1][2][5][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 2, [3  2  1]
        perm[1][2][5][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 2, [2  1  3]
        perm[1][2][5][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 2, [1  3  2]
        perm[1][2][5][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 3, [1  2  3]
        perm[1][2][5][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 3, [3  1  2]
        perm[1][2][5][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 3, [2  3  1]
        perm[1][2][5][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 3, [3  2  1]
        perm[1][2][5][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 3, [2  1  3]
        perm[1][2][5][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 3, [1  3  2]
        perm[1][2][5][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 4, [1  2  3]
        perm[1][2][5][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 4, [3  1  2]
        perm[1][2][5][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 4, [2  3  1]
        perm[1][2][5][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 4, [3  2  1]
        perm[1][2][5][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 4, [2  1  3]
        perm[1][2][5][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 3, [1  3  2] ---> (init: face, perm) 4, [1  3  2]
        perm[1][2][5][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 1, [1  2  3]
        perm[1][3][0][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 1, [3  1  2]
        perm[1][3][0][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 1, [2  3  1]
        perm[1][3][0][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 1, [3  2  1]
        perm[1][3][0][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 1, [2  1  3]
        perm[1][3][0][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 1, [1  3  2]
        perm[1][3][0][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 2, [1  2  3]
        perm[1][3][0][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 2, [3  1  2]
        perm[1][3][0][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 2, [2  3  1]
        perm[1][3][0][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 2, [3  2  1]
        perm[1][3][0][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 2, [2  1  3]
        perm[1][3][0][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 2, [1  3  2]
        perm[1][3][0][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 3, [1  2  3]
        perm[1][3][0][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 3, [3  1  2]
        perm[1][3][0][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 3, [2  3  1]
        perm[1][3][0][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 3, [3  2  1]
        perm[1][3][0][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 3, [2  1  3]
        perm[1][3][0][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 3, [1  3  2]
        perm[1][3][0][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 4, [1  2  3]
        perm[1][3][0][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 4, [3  1  2]
        perm[1][3][0][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 4, [2  3  1]
        perm[1][3][0][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 4, [3  2  1]
        perm[1][3][0][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 4, [2  1  3]
        perm[1][3][0][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  2  3] ---> (init: face, perm) 4, [1  3  2]
        perm[1][3][0][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 1, [1  2  3]
        perm[1][3][1][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 1, [3  1  2]
        perm[1][3][1][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 1, [2  3  1]
        perm[1][3][1][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 1, [3  2  1]
        perm[1][3][1][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 1, [2  1  3]
        perm[1][3][1][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 1, [1  3  2]
        perm[1][3][1][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 2, [1  2  3]
        perm[1][3][1][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 2, [3  1  2]
        perm[1][3][1][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 2, [2  3  1]
        perm[1][3][1][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 2, [3  2  1]
        perm[1][3][1][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 2, [2  1  3]
        perm[1][3][1][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 2, [1  3  2]
        perm[1][3][1][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 3, [1  2  3]
        perm[1][3][1][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 3, [3  1  2]
        perm[1][3][1][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 3, [2  3  1]
        perm[1][3][1][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 3, [3  2  1]
        perm[1][3][1][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 3, [2  1  3]
        perm[1][3][1][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 3, [1  3  2]
        perm[1][3][1][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 4, [1  2  3]
        perm[1][3][1][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 4, [3  1  2]
        perm[1][3][1][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 4, [2  3  1]
        perm[1][3][1][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 4, [3  2  1]
        perm[1][3][1][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 4, [2  1  3]
        perm[1][3][1][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  1  2] ---> (init: face, perm) 4, [1  3  2]
        perm[1][3][1][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 1, [1  2  3]
        perm[1][3][2][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 1, [3  1  2]
        perm[1][3][2][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 1, [2  3  1]
        perm[1][3][2][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 1, [3  2  1]
        perm[1][3][2][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 1, [2  1  3]
        perm[1][3][2][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 1, [1  3  2]
        perm[1][3][2][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 2, [1  2  3]
        perm[1][3][2][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 2, [3  1  2]
        perm[1][3][2][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 2, [2  3  1]
        perm[1][3][2][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 2, [3  2  1]
        perm[1][3][2][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 2, [2  1  3]
        perm[1][3][2][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 2, [1  3  2]
        perm[1][3][2][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 3, [1  2  3]
        perm[1][3][2][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 3, [3  1  2]
        perm[1][3][2][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 3, [2  3  1]
        perm[1][3][2][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 3, [3  2  1]
        perm[1][3][2][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 3, [2  1  3]
        perm[1][3][2][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 3, [1  3  2]
        perm[1][3][2][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 4, [1  2  3]
        perm[1][3][2][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 4, [3  1  2]
        perm[1][3][2][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 4, [2  3  1]
        perm[1][3][2][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 4, [3  2  1]
        perm[1][3][2][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 4, [2  1  3]
        perm[1][3][2][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  3  1] ---> (init: face, perm) 4, [1  3  2]
        perm[1][3][2][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 1, [1  2  3]
        perm[1][3][3][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 1, [3  1  2]
        perm[1][3][3][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 1, [2  3  1]
        perm[1][3][3][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 1, [3  2  1]
        perm[1][3][3][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 1, [2  1  3]
        perm[1][3][3][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 1, [1  3  2]
        perm[1][3][3][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 2, [1  2  3]
        perm[1][3][3][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 2, [3  1  2]
        perm[1][3][3][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 2, [2  3  1]
        perm[1][3][3][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 2, [3  2  1]
        perm[1][3][3][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 2, [2  1  3]
        perm[1][3][3][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 2, [1  3  2]
        perm[1][3][3][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 3, [1  2  3]
        perm[1][3][3][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 3, [3  1  2]
        perm[1][3][3][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 3, [2  3  1]
        perm[1][3][3][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 3, [3  2  1]
        perm[1][3][3][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 3, [2  1  3]
        perm[1][3][3][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 3, [1  3  2]
        perm[1][3][3][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 4, [1  2  3]
        perm[1][3][3][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 4, [3  1  2]
        perm[1][3][3][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 4, [2  3  1]
        perm[1][3][3][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 4, [3  2  1]
        perm[1][3][3][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 4, [2  1  3]
        perm[1][3][3][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [3  2  1] ---> (init: face, perm) 4, [1  3  2]
        perm[1][3][3][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 1, [1  2  3]
        perm[1][3][4][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 1, [3  1  2]
        perm[1][3][4][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 1, [2  3  1]
        perm[1][3][4][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 1, [3  2  1]
        perm[1][3][4][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 1, [2  1  3]
        perm[1][3][4][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 1, [1  3  2]
        perm[1][3][4][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 2, [1  2  3]
        perm[1][3][4][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 2, [3  1  2]
        perm[1][3][4][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 2, [2  3  1]
        perm[1][3][4][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 2, [3  2  1]
        perm[1][3][4][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 2, [2  1  3]
        perm[1][3][4][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 2, [1  3  2]
        perm[1][3][4][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 3, [1  2  3]
        perm[1][3][4][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 3, [3  1  2]
        perm[1][3][4][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 3, [2  3  1]
        perm[1][3][4][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 3, [3  2  1]
        perm[1][3][4][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 3, [2  1  3]
        perm[1][3][4][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 3, [1  3  2]
        perm[1][3][4][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 4, [1  2  3]
        perm[1][3][4][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 4, [3  1  2]
        perm[1][3][4][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 4, [2  3  1]
        perm[1][3][4][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 4, [3  2  1]
        perm[1][3][4][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 4, [2  1  3]
        perm[1][3][4][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [2  1  3] ---> (init: face, perm) 4, [1  3  2]
        perm[1][3][4][3][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 1, [1  2  3]
        perm[1][3][5][0][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 1, [3  1  2]
        perm[1][3][5][0][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 1, [2  3  1]
        perm[1][3][5][0][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 1, [3  2  1]
        perm[1][3][5][0][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 1, [2  1  3]
        perm[1][3][5][0][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 1, [1  3  2]
        perm[1][3][5][0][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 2, [1  2  3]
        perm[1][3][5][1][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 2, [3  1  2]
        perm[1][3][5][1][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 2, [2  3  1]
        perm[1][3][5][1][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 2, [3  2  1]
        perm[1][3][5][1][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 2, [2  1  3]
        perm[1][3][5][1][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 2, [1  3  2]
        perm[1][3][5][1][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 3, [1  2  3]
        perm[1][3][5][2][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 3, [3  1  2]
        perm[1][3][5][2][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 3, [2  3  1]
        perm[1][3][5][2][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 3, [3  2  1]
        perm[1][3][5][2][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 3, [2  1  3]
        perm[1][3][5][2][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 3, [1  3  2]
        perm[1][3][5][2][5].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 4, [1  2  3]
        perm[1][3][5][3][0].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 4, [3  1  2]
        perm[1][3][5][3][1].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 4, [2  3  1]
        perm[1][3][5][3][2].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 4, [3  2  1]
        perm[1][3][5][3][3].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 4, [2  1  3]
        perm[1][3][5][3][4].map[1] = 1;

        // DoF set #2, (current: face, perm) 4, [1  3  2] ---> (init: face, perm) 4, [1  3  2]
        perm[1][3][5][3][5].map[1] = 1;

        /*------------   END: Auto Generate ------------*/
        }

    inline const int* Get_Map (const int& dof_set, const int& new_face,  const int& perm_new,
                                                   const int& init_face, const int& perm_init) const
        {
        // error check
        if ((new_face < 1) || (new_face > 4))
            {
            mexPrintf("new_face must be = 1,2,3,4.\n");
            mexErrMsgTxt("Invalid local face index!\n");
            }
        if ((perm_new < 1) || (perm_new > 6))
            {
            mexPrintf("perm_new must be = 1,2,3,4,5,6.\n");
            mexErrMsgTxt("Invalid permutation index!\n");
            }
        if ((init_face < 1) || (init_face > 4))
            {
            mexPrintf("init_face must be = 1,2,3,4.\n");
            mexErrMsgTxt("Invalid local face index!\n");
            }
        if ((perm_init < 1) || (perm_init > 6))
            {
            mexPrintf("perm_init must be = 1,2,3,4,5,6.\n");
            mexErrMsgTxt("Invalid permutation index!\n");
            }
        if ((dof_set < 1) || (dof_set > Num_Face_Sets))
            {
            mexPrintf("dof_set must be >= 1 and <= %d.\n",Num_Face_Sets);
            mexErrMsgTxt("Invalid dof_set index!\n");
            }

        return perm[dof_set-1][new_face-1][perm_new-1][init_face-1][perm_init-1].map;
        }

private:
    // contains all of the permutation maps
    // input: [dof_set][new_face_index][perm of new_face][init_face_index][perm of init_face]
    SINGLE_FACE_PERM_MAP perm[Num_Face_Sets][NUM_FACE][6][NUM_FACE][6];
};

/***************************************************************************************/
class EDA
{
public:
    char*    Name;              // name of finite element space
    int      Dim;               // intrinsic dimension
    char*    Domain;            // type of domain: "interval", "triangle", "tetrahedron"

    NODAL_TOPOLOGY   Node;      // Nodal DoF arrangment
    EDGE_DOF_PERMUTATION   Edge_DoF_Perm; // permutation maps for local edge DoFs:
                                          // this is for allocating DoFs on edges
                                          //      consistently between elements.
    FACE_DOF_PERMUTATION   Face_DoF_Perm; // permutation maps for local face DoFs:
                                          // this is for allocating DoFs on faces
                                          //      consistently between elements.

    EDA ();  // constructor
    ~EDA (); // DE-structor
    mxArray* Init_DoF_Map(int);
    void  Fill_DoF_Map  (TETRAHEDRON_DATA*);
    int  Assign_Vtx_DoF (TETRAHEDRON_DATA*, const int);
    int  Assign_Edge_DoF(TETRAHEDRON_DATA*, const int);
    int  Assign_Face_DoF(TETRAHEDRON_DATA*, const int);
    int  Assign_Tet_DoF (TETRAHEDRON_DATA*, const int);
    void  Error_Check_DoF_Map(const int&);

private:
    int*  cell_dof[Total_DoF_Per_Cell+1];
};

/***************************************************************************************/
/* constructor */
EDA::EDA ()
{
    // setup
    Name              = (char *) ELEM_Name;
    Dim               = Dimension;
    Domain            = (char *) Domain_str;
    Edge_DoF_Perm.Setup_Maps(); // setup edge DoF permutation maps
    Face_DoF_Perm.Setup_Maps(); // setup face DoF permutation maps

/*------------ BEGIN: Auto Generate ------------*/
    // nodal DoF arrangement


    Node.F[1][1][1] = 1; // Set1, F1
    Node.F[1][1][2] = 2; // Set1, F1
    Node.F[1][1][3] = 3; // Set1, F1
    Node.F[1][2][1] = 4; // Set1, F2
    Node.F[1][2][2] = 5; // Set1, F2
    Node.F[1][2][3] = 6; // Set1, F2
    Node.F[1][3][1] = 7; // Set1, F3
    Node.F[1][3][2] = 8; // Set1, F3
    Node.F[1][3][3] = 9; // Set1, F3
    Node.F[1][4][1] = 10; // Set1, F4
    Node.F[1][4][2] = 11; // Set1, F4
    Node.F[1][4][3] = 12; // Set1, F4

    Node.F[2][1][1] = 13; // Set2, F1
    Node.F[2][2][1] = 14; // Set2, F2
    Node.F[2][3][1] = 15; // Set2, F3
    Node.F[2][4][1] = 16; // Set2, F4


/*------------   END: Auto Generate ------------*/
}
/***************************************************************************************/


/***************************************************************************************/
/* destructor */
EDA::~EDA ()
{
}
/***************************************************************************************/


/***************************************************************************************/
/* initialize FEM DoF map to all zeros */
mxArray* EDA::Init_DoF_Map(int  Num_Cell)     // output to MATLAB
{
    mxArray* DoF_Map;

    // BEGIN: allocate and access output data
    DoF_Map = mxCreateNumericMatrix((mwSize)Num_Cell, (mwSize)Total_DoF_Per_Cell, mxUINT32_CLASS, mxREAL);
    // access the data
    // split up the columns of the data
    cell_dof[0] = NULL; // not used!
    cell_dof[1] = (int *) mxGetPr(DoF_Map);
    for (int i = 2; (i <= Total_DoF_Per_Cell); i++) // note: off by one because of C-style indexing!
        cell_dof[i] = cell_dof[i-1] + Num_Cell;
    // END: allocate and access output data

    return DoF_Map;
}
/***************************************************************************************/


/***************************************************************************************/
/* assign DoFs */
void EDA::Fill_DoF_Map(TETRAHEDRON_DATA*  Tet_Data)
{
    // output what we are doing
    printf("Generating DoFs on ");
    printf(Domain);
    printf("s for the Finite Element: ");
    printf(Name);
    printf(".\n");

    // write the local to global DoF mapping
    int DoF_Offset = 0;
    DoF_Offset = Assign_Vtx_DoF (Tet_Data, DoF_Offset);
    DoF_Offset = Assign_Edge_DoF(Tet_Data, DoF_Offset);
    DoF_Offset = Assign_Face_DoF(Tet_Data, DoF_Offset);
    DoF_Offset = Assign_Tet_DoF (Tet_Data, DoF_Offset);
    Error_Check_DoF_Map(Tet_Data->Num_Tet);
}
/***************************************************************************************/


/***************************************************************************************/
/* allocate the vertex DoFs */
int EDA::Assign_Vtx_DoF(TETRAHEDRON_DATA*      Tet_Data,     // inputs (list of tets)
                        const int              DoF_Offset)
{
    // for every vertex, store the first tet (with local vertex index) that allocated DoFs for it
    vector<int> Vtx_to_Tet[2];
    Vtx_to_Tet[0].resize((unsigned int)(Tet_Data->Max_Vtx_Index+1));
    Vtx_to_Tet[1].resize((unsigned int)(Tet_Data->Max_Vtx_Index+1));

    int Vtx_DoF_Counter = DoF_Offset;
    // allocate DoFs for the global vertices that are actually present
    if (Num_DoF_Per_Vtx > 0)
        {
        int VTX_COUNT = 0; // keep track of the number of unique vertices
        for (int ti=0; ti < Tet_Data->Num_Tet; ti++)
            {
            // loop thru all vertices of each tet
            for (int vtxi=0; vtxi < NUM_VTX; vtxi++)
                {
                const unsigned int Vertex = (unsigned int) Tet_Data->Tet_DoF[vtxi][ti];
                const int tet_ind         =   ti+1; // put into MATLAB-style indexing
                const int vtx_ind         = vtxi+1; // put into MATLAB-style indexing
                // if this vertex has NOT been visited yet
                if (Vtx_to_Tet[0][Vertex]==0)
                    {
                    VTX_COUNT++;

                    // remember which tet is associated with this vertex
                    Vtx_to_Tet[0][Vertex] = tet_ind;
                    // remember the local vertex
                    Vtx_to_Tet[1][Vertex] = vtx_ind;

                    /*------------ BEGIN: Auto Generate ------------*/
                    // Set 1: add more DoFs
                    /*------------   END: Auto Generate ------------*/
                    }
                else
                    {
                    const int old_ti      = Vtx_to_Tet[0][Vertex]-1; // put into C-style indexing
                    const int old_vtx_ind = Vtx_to_Tet[1][Vertex];

                    /*------------ BEGIN: Auto Generate ------------*/
                    // Set 1: copy over DoFs
                    /*------------   END: Auto Generate ------------*/
                    }
                }
            }
        // store the number of unique vertices
        Tet_Data->Num_Unique_Vertices = VTX_COUNT;
        }
    return Vtx_DoF_Counter;
}
/***************************************************************************************/


/***************************************************************************************/
/* allocate the edge DoFs */
int EDA::Assign_Edge_DoF(TETRAHEDRON_DATA*      Tet_Data,    // inputs (list of edges in triangulation)
                         const int              DoF_Offset)
{
    int Edge_DoF_Counter = DoF_Offset;
    // allocate DoFs for the edges in the triangulation
    if (Num_DoF_Per_Edge > 0)
        {
        // temp storage for DoFs on edge
        int SE_DoF[Num_Edge_Sets+1][Max_DoF_Per_Edge+1];
        // initialize to NULL value
        for (int ind1=0; ind1 < Num_Edge_Sets+1; ind1++)
        for (int ind2=0; ind2 < Max_DoF_Per_Edge+1; ind2++)
            SE_DoF[ind1][ind2] = -1;

        // initialize previous edge to NULL
        EDGE_TO_CELL Prev_EI;
        Prev_EI.edge_info[0] = -1;
        Prev_EI.edge_info[1] = -1;
        Prev_EI.edge_info[2] = -1;
        Prev_EI.edge_info[3] = 0;

        std::vector<EDGE_TO_CELL>::iterator ei; // need iterator
        int initial_local_edge_index = -100; // init to a bogus value
        int EDGE_COUNT = 0;
        for (ei=Tet_Data->Edge_List.begin(); ei!=Tet_Data->Edge_List.end(); ++ei)
            {
            const EDGE_TO_CELL Current_EI = *ei;
            // determine if this edge is different from the previous edge
            const bool NEW_EDGE = (Prev_EI.edge_info[0]!=Current_EI.edge_info[0]) || (Prev_EI.edge_info[1]!=Current_EI.edge_info[1]);
            // get the current cell index, and current local edge index
            const unsigned int current_cell_index = (unsigned int) Current_EI.edge_info[2];
            const int current_local_edge_index = Current_EI.edge_info[3];

            // if this edge has NOT been visited yet
            if (NEW_EDGE)
                {
                EDGE_COUNT++;
                Prev_EI = Current_EI; // update previous

                // store the local edge index that these DoFs will be allocated on
                initial_local_edge_index = current_local_edge_index;

                // store new DoF's in a temporary structure
                /*------------ BEGIN: Auto Generate ------------*/
                // Set 1: add more DoFs
                /*------------   END: Auto Generate ------------*/
                }
            else // we are still on the same global edge
                {
                // if the previous edge and the current edge are contained in the same cell (tetrahedron)
                if (Prev_EI.edge_info[2]==current_cell_index)
                    {
                    // then, this should not happen!
                    mexPrintf("An edge appears more than once, and referenced to the same cell,\n");
                    mexPrintf("        in an internal list of this sub-routine!\n");
                    mexPrintf("This should never happen!\n");
                    mexPrintf("Please report this bug!\n");
                    mexErrMsgTxt("STOP!\n");
                    }
                }

            const int* edge_perm_map = NULL; // this holds the permutation to apply
            const int cei = abs_index(current_local_edge_index); // get the positive index

            /*------------ BEGIN: Auto Generate ------------*/
            // Set 1:
            edge_perm_map = Edge_DoF_Perm.Get_Map(1, current_local_edge_index, initial_local_edge_index);
            // do *nothing*; there are no edge DoFs here
            /*------------   END: Auto Generate ------------*/
            }
        // store the number of unique edges
        Tet_Data->Num_Unique_Edges = EDGE_COUNT;
        }
    return Edge_DoF_Counter;
}
/***************************************************************************************/


/***************************************************************************************/
/* allocate the face DoFs */
int EDA::Assign_Face_DoF(TETRAHEDRON_DATA*      Tet_Data,    // inputs (list of faces in tetrahedral mesh)
                         const int              DoF_Offset)
{
    int Face_DoF_Counter = DoF_Offset;
    // allocate DoFs for the faces in the triangulation
    if (Num_DoF_Per_Face > 0)
        {
        // temp storage for DoFs on face
        int SF_DoF[Num_Face_Sets+1][Max_DoF_Per_Face+1];
        // initialize to NULL value
        for (int ind1=0; ind1 < Num_Face_Sets+1; ind1++)
        for (int ind2=0; ind2 < Max_DoF_Per_Face+1; ind2++)
            SF_DoF[ind1][ind2] = -1;

        // initialize previous face to NULL
        FACE_TO_CELL Prev_FI;
        Prev_FI.face_info[0] = 0;
        Prev_FI.face_info[1] = 0;
        Prev_FI.face_info[2] = 0;
        Prev_FI.face_info[3] = 0;
        Prev_FI.face_info[4] = 0;
        Prev_FI.face_info[5] = 0;

        std::vector<FACE_TO_CELL>::iterator fi; // need iterator
        unsigned int initial_local_face_index = 100; // init to a bogus value
        unsigned int initial_perm             = 100; // init to a bogus value
        int FACE_COUNT = 0;
        for (fi=Tet_Data->Face_List.begin(); fi!=Tet_Data->Face_List.end(); ++fi)
            {
            const FACE_TO_CELL Current_FI = *fi;
            // determine if this face is different from the previous face
            const bool NEW_FACE = (Prev_FI.face_info[0]!=Current_FI.face_info[0]) ||
                                  (Prev_FI.face_info[1]!=Current_FI.face_info[1]) ||
                                  (Prev_FI.face_info[2]!=Current_FI.face_info[2]);
            // get the current cell index, local face index, and permutation
            const unsigned int current_cell_index       = Current_FI.face_info[3];
            const unsigned int current_local_face_index = Current_FI.face_info[4];
            const unsigned int current_perm             = Current_FI.face_info[5];

            // if this face has NOT been visited yet
            if (NEW_FACE)
                {
                FACE_COUNT++;
                Prev_FI = Current_FI; // update previous

                // store the local face index that these DoFs will be allocated on
                initial_local_face_index = current_local_face_index;
                initial_perm             = current_perm;

                // store new DoF's in a temporary structure
                /*------------ BEGIN: Auto Generate ------------*/
                // Set 1: add more DoFs
                Face_DoF_Counter++;
                SF_DoF[1][1] = Face_DoF_Counter;
                Face_DoF_Counter++;
                SF_DoF[1][2] = Face_DoF_Counter;
                Face_DoF_Counter++;
                SF_DoF[1][3] = Face_DoF_Counter;
                // Set 2: add more DoFs
                Face_DoF_Counter++;
                SF_DoF[2][1] = Face_DoF_Counter;
                /*------------   END: Auto Generate ------------*/
                }
            else // we are still on the same global face
                {
                // if the previous face and the current face are contained in the same cell (tetrahedron)
                if (Prev_FI.face_info[3]==current_cell_index)
                    {
                    // then, this should not happen!
                    mexPrintf("A face appears more than once, and referenced to the same cell,\n");
                    mexPrintf("        in an internal list of this sub-routine!\n");
                    mexPrintf("This should never happen!\n");
                    mexPrintf("Please report this bug!\n");
                    mexErrMsgTxt("STOP!\n");
                    }
                }

            const int* face_perm_map = NULL; // this holds the permutation to apply
            const int cfi = (int) current_local_face_index;

            /*------------ BEGIN: Auto Generate ------------*/
            // Set 1:
            face_perm_map = Face_DoF_Perm.Get_Map(1, current_local_face_index, current_perm, initial_local_face_index, initial_perm);
            cell_dof[ Node.F[1][cfi][1] ][current_cell_index] = SF_DoF[1][ face_perm_map[1] ];
            cell_dof[ Node.F[1][cfi][2] ][current_cell_index] = SF_DoF[1][ face_perm_map[2] ];
            cell_dof[ Node.F[1][cfi][3] ][current_cell_index] = SF_DoF[1][ face_perm_map[3] ];
            // Set 2:
            face_perm_map = Face_DoF_Perm.Get_Map(2, current_local_face_index, current_perm, initial_local_face_index, initial_perm);
            cell_dof[ Node.F[2][cfi][1] ][current_cell_index] = SF_DoF[2][ face_perm_map[1] ];
            /*------------   END: Auto Generate ------------*/
            }
        // store the number of unique faces
        Tet_Data->Num_Unique_Faces = FACE_COUNT;
        }
    return Face_DoF_Counter;
}
/***************************************************************************************/


/***************************************************************************************/
/* allocate the tet DoFs */
int EDA::Assign_Tet_DoF(TETRAHEDRON_DATA*       Tet_Data,    // inputs (list of tets)
                        const int               DoF_Offset)
{
    int Tet_DoF_Counter = DoF_Offset;
    // special case: there are 1-to-N tets (no gaps in numbering)
    if (Num_DoF_Per_Tet > 0)
        {
        // allocate DoFs for the global tets that are actually present
        for (int ti=0; ti < Tet_Data->Num_Tet; ti++)
            {
            const int tet_ind = 1; // there is only 1 tet in 3D

            /*------------ BEGIN: Auto Generate ------------*/
            // Set 1: add more DoFs
            /*------------   END: Auto Generate ------------*/
            }
        }
    return Tet_DoF_Counter;
}
/***************************************************************************************/


/***************************************************************************************/
/* perform error checks on Finite Element Degree-of-Freedom (DoF) map */
void EDA::Error_Check_DoF_Map(const int&  Num_Cell)
{
    // BEGIN: error check
    const unsigned int LENGTH = Total_DoF_Per_Cell*Num_Cell;
    const unsigned int MIN_DoF = *min_element(cell_dof[1], cell_dof[1]+LENGTH);
    if (MIN_DoF < 1)
        {
        mexPrintf("ERROR: DoFmap references DoF with index 0!\n");
        mexPrintf("ERROR: Some DoFs were never allocated!\n");
        mexErrMsgTxt("ERROR: Make sure your mesh describes a domain that is a manifold!");
        }
    // END: error check
}
/***************************************************************************************/


#undef ELEM_Name
#undef EDA
#undef NODAL_TOPOLOGY
#undef SINGLE_EDGE_PERM_MAP
#undef EDGE_DOF_PERMUTATION
#undef SINGLE_FACE_PERM_MAP
#undef FACE_DOF_PERMUTATION
#undef Dimension
#undef Domain_str

#undef NUM_VTX
#undef NUM_EDGE
#undef NUM_FACE
#undef NUM_TET

#undef Num_Vtx_Sets
#undef Num_Edge_Sets
#undef Num_Face_Sets
#undef Num_Tet_Sets

#undef Max_DoF_Per_Vtx
#undef Max_DoF_Per_Edge
#undef Max_DoF_Per_Face
#undef Max_DoF_Per_Tet

#undef Num_DoF_Per_Vtx
#undef Num_DoF_Per_Edge
#undef Num_DoF_Per_Face
#undef Num_DoF_Per_Tet

#undef Total_DoF_Per_Cell

/***/
