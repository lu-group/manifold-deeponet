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
#define ELEM_Name        "Elem1_Test"
// set allocator class name
#define EDA               Elem1_Test_DoF_Allocator
// set nodal topology data type name
#define NODAL_TOPOLOGY    Elem1_Test_nodal_top
// set permutation map data type names
#define SINGLE_EDGE_PERM_MAP    Elem1_Test_single_edge_perm_map
#define EDGE_DOF_PERMUTATION    Elem1_Test_edge_dof_perm
#define SINGLE_FACE_PERM_MAP    Elem1_Test_single_face_perm_map
#define FACE_DOF_PERMUTATION    Elem1_Test_face_dof_perm

// set the number of DoF sets
#define Num_Vtx_Sets        1
#define Num_Edge_Sets       0
#define Num_Face_Sets       0
#define Num_Tet_Sets        0

// set the max number (over all sets) of DoFs per entity per set
#define Max_DoF_Per_Vtx     1
#define Max_DoF_Per_Edge    0
#define Max_DoF_Per_Face    0
#define Max_DoF_Per_Tet     0

// set the total number of DoFs per entity
#define Num_DoF_Per_Vtx     1
#define Num_DoF_Per_Edge    0
#define Num_DoF_Per_Face    0
#define Num_DoF_Per_Tet     0

// set the TOTAL number of DoFs per cell (element)
#define Total_DoF_Per_Cell  3
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
// set (intrinsic) dimension and domain
#define Dimension    2
#define Domain_str  "triangle"

// set the number of vertices
#define NUM_VTX    3
// set the number of edges
#define NUM_EDGE   3
// set the number of faces
#define NUM_FACE   1
// set the number of tetrahedrons
#define NUM_TET    0
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
        if ((new_edge==0) || (new_edge < -3) || (new_edge > 3))
            {
            mexPrintf("new_edge must be = +/- 1,2,3.\n");
            mexErrMsgTxt("Invalid local edge index!\n");
            }
        if ((init_edge==0) || (init_edge < -3) || (init_edge > 3))
            {
            mexPrintf("init_edge must be = +/- 1,2,3.\n");
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
class EDA
{
public:
    char*    Name;              // name of finite element space
    int      Dim;               // intrinsic dimension
    char*    Domain;            // type of domain: "interval", "triangle", "tetrahedron"

    NODAL_TOPOLOGY         Node;          // Nodal DoF arrangment
    EDGE_DOF_PERMUTATION   Edge_DoF_Perm; // permutation maps for local edge DoFs:
                                          // this is for allocating DoFs on edges
                                          //      consistently between elements.

    EDA ();  // constructor
    ~EDA (); // DE-structor
    mxArray* Init_DoF_Map(int);
    void  Fill_DoF_Map  (TRIANGLE_EDGE_SEARCH*);
    int  Assign_Vtx_DoF (TRIANGLE_EDGE_SEARCH*, const int);
    int  Assign_Edge_DoF(TRIANGLE_EDGE_SEARCH*, const int);
    int  Assign_Face_DoF(TRIANGLE_EDGE_SEARCH*, const int);
    int  Assign_Tet_DoF (TRIANGLE_EDGE_SEARCH*, const int);
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

/*------------ BEGIN: Auto Generate ------------*/
    // nodal DoF arrangement
    Node.V[1][1][1] = 1; // Set1, V1
    Node.V[1][2][1] = 2; // Set1, V2
    Node.V[1][3][1] = 3; // Set1, V3




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
void EDA::Fill_DoF_Map(TRIANGLE_EDGE_SEARCH*  Tri_Edge_Search)
{
    // output what we are doing
    printf("Generating DoFs on ");
    printf(Domain);
    printf("s for the Finite Element: ");
    printf(Name);
    printf(".\n");

    // write the local to global DoF mapping
    int DoF_Offset = 0;
    DoF_Offset = Assign_Vtx_DoF (Tri_Edge_Search, DoF_Offset);
    DoF_Offset = Assign_Edge_DoF(Tri_Edge_Search, DoF_Offset);
    DoF_Offset = Assign_Face_DoF(Tri_Edge_Search, DoF_Offset);
    DoF_Offset = Assign_Tet_DoF (Tri_Edge_Search, DoF_Offset);
    Error_Check_DoF_Map(Tri_Edge_Search->Num_Tri);
}
/***************************************************************************************/


/***************************************************************************************/
/* allocate the vertex DoFs */
int EDA::Assign_Vtx_DoF(TRIANGLE_EDGE_SEARCH*  Tri_Edge_Search,   // inputs (list of triangles)
                        const int              DoF_Offset)
{
    // for every vertex, store the first triangle (with local vertex index) that allocated DoFs for it
    vector<int> Vtx_to_Tri[2];
    Vtx_to_Tri[0].resize((unsigned int)(Tri_Edge_Search->Max_Vtx_Index+1),-1); // init to NULL value
    Vtx_to_Tri[1].resize((unsigned int)(Tri_Edge_Search->Max_Vtx_Index+1),-1); // init to NULL value

    int Vtx_DoF_Counter = DoF_Offset;
    // allocate DoFs for the global vertices that are actually present
    if (Num_DoF_Per_Vtx > 0)
        {
        int VTX_COUNT = 0; // keep track of the number of unique vertices
        for (int ti=0; ti < Tri_Edge_Search->Num_Tri; ti++)
            {
            // loop thru all vertices of each triangle
            for (int vtxi=0; vtxi < NUM_VTX; vtxi++)
                {
                const unsigned int Vertex = (unsigned int) Tri_Edge_Search->Tri_DoF[vtxi][ti];
                const int tri_ind         =     ti;
                const int vtx_ind         = vtxi+1; // put into MATLAB-style indexing
                // if this vertex has NOT been visited yet
                if (Vtx_to_Tri[0][Vertex]==-1)
                    {
                    VTX_COUNT++;

                    // remember which triangle is associated with this vertex
                    Vtx_to_Tri[0][Vertex] = tri_ind;
                    // remember the local vertex
                    Vtx_to_Tri[1][Vertex] = vtx_ind;

                    /*------------ BEGIN: Auto Generate ------------*/
                    // Set 1: add more DoFs
                    Vtx_DoF_Counter++;
                    cell_dof[ Node.V[1][vtx_ind][1] ][ti] = Vtx_DoF_Counter;
                    /*------------   END: Auto Generate ------------*/
                    }
                else
                    {
                    const int old_ti      = Vtx_to_Tri[0][Vertex];
                    const int old_vtx_ind = Vtx_to_Tri[1][Vertex];

                    /*------------ BEGIN: Auto Generate ------------*/
                    // Set 1: copy over DoFs
                    cell_dof[ Node.V[1][vtx_ind][1] ][ti] = cell_dof[ Node.V[1][old_vtx_ind][1] ][old_ti];
                    /*------------   END: Auto Generate ------------*/
                    }
                }
            }
        // store the number of unique vertices
        Tri_Edge_Search->Num_Unique_Vertices = VTX_COUNT;
        }
    return Vtx_DoF_Counter;
}
/***************************************************************************************/


/***************************************************************************************/
/* allocate the edge DoFs */
int EDA::Assign_Edge_DoF(TRIANGLE_EDGE_SEARCH*  Tri_Edge_Search,   // inputs (list of edges in triangulation)
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
        for (ei=Tri_Edge_Search->Edge_List.begin(); ei!=Tri_Edge_Search->Edge_List.end(); ++ei)
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
                // if the previous edge and the current edge are contained in the same cell (triangle)
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
        Tri_Edge_Search->Num_Unique_Edges = EDGE_COUNT;
        }
    return Edge_DoF_Counter;
}
/***************************************************************************************/


/***************************************************************************************/
/* allocate the face DoFs */
int EDA::Assign_Face_DoF(TRIANGLE_EDGE_SEARCH*  Tri_Edge_Search,   // inputs (list of triangles)
                         const int              DoF_Offset)
{
    int Face_DoF_Counter = DoF_Offset;
    // special case: there are 1-to-N triangles (no gaps in numbering)
    if (Num_DoF_Per_Face > 0)
        {
        // allocate DoFs for the faces (triangles) in the mesh
        for (int ti=0; ti < Tri_Edge_Search->Num_Tri; ti++)
            {
            const int face_ind = 1; // there is only 1 face in 2D
            /*------------ BEGIN: Auto Generate ------------*/
            // Set 1: add more DoFs
            /*------------   END: Auto Generate ------------*/
            }
        }
    return Face_DoF_Counter;
}
/***************************************************************************************/


/***************************************************************************************/
/* allocate the tet DoFs */
int EDA::Assign_Tet_DoF(TRIANGLE_EDGE_SEARCH*  Tri_Edge_Search,   // inputs (list of triangles)
                        const int              DoF_Offset)
{
    int Tet_DoF_Counter = DoF_Offset;
    // special case: there are no tets in 2D
    if (Num_DoF_Per_Tet > 0)
        {
        // this should not run!
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
