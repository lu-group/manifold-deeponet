/*
============================================================================================
   Methods for a C++ Class that does generic point seearching in meshes.

   NOTE: portions of this code are automatically generated!

   Copyright (c) 06-16-2014,  Shawn W. Walker
============================================================================================
*/

#define GPS Generic_Point_Search

/***************************************************************************************/
/* constructor */
GPS::GPS (const mxArray *prhs[])
{
    /*------------ BEGIN: Auto Generate ------------*/
    // setup inputs
    const mxArray *PRHS_EMPTY = NULL;
    Domain_Omega_embedded_in_Omega_restricted_to_Omega.Setup_Data(prhs[PRHS_EMPTY_2], prhs[PRHS_Omega_Mesh_DoFmap], PRHS_EMPTY);

    Omega_Search_Data.Setup("Omega", 2, prhs[PRHS_Omega_Search_Data]);

    Omega_Found_Points.Setup(&Omega_Search_Data, 2);

    geom_Omega_embedded_in_Omega_restricted_to_Omega.Setup_Mesh_Geometry(prhs[PRHS_Omega_Mesh_Vertices], prhs[PRHS_Omega_Mesh_DoFmap], prhs[PRHS_EMPTY_1]);
    geom_Omega_embedded_in_Omega_restricted_to_Omega.Domain = &Domain_Omega_embedded_in_Omega_restricted_to_Omega;

    Setup_Data(prhs);
    /*------------   END: Auto Generate ------------*/
}
/***************************************************************************************/


/***************************************************************************************/
/* destructor */
GPS::~GPS ()
{
    // clear it
    /*------------ BEGIN: Auto Generate ------------*/
    delete(Omega_Search_Obj);
    /*------------   END: Auto Generate ------------*/
}
/***************************************************************************************/


/***************************************************************************************/
/* setup point search objects and pass domain and geometry pointers around */
void GPS::Setup_Data(const mxArray *prhs[]) // input from MATLAB
{
    /*------------ BEGIN: Auto Generate ------------*/
    // setup the desired domain point searches
    Omega_Search_Obj = new CLASS_Search_Omega(&Omega_Search_Data, &Omega_Found_Points);
    /*------------   END: Auto Generate ------------*/

    /*------------ BEGIN: Auto Generate ------------*/
    // pass pointers around
    Omega_Search_Obj->Domain = &Domain_Omega_embedded_in_Omega_restricted_to_Omega;
    Omega_Search_Obj->GeomFunc = &geom_Omega_embedded_in_Omega_restricted_to_Omega;
    Omega_Search_Obj->Consistency_Check();

    /*------------   END: Auto Generate ------------*/
}
/***************************************************************************************/


/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* here is the main calling routine for point searching */
void GPS::Find_Points ()
{
    // find points in the sub-Domain: Omega
    Omega_Search_Obj->Find_Points();

}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/

/*------------ BEGIN: Auto Generate ------------*/
/***************************************************************************************/
/* this outputs all point searching data as MATLAB matrices (in cell arrays) */
void GPS::Output_Points (mxArray* plhs[])
{
    Output_Single_Point_Data(0, Omega_Found_Points.Get_mxLocal_Points_Ptr(), mxCreateString(Omega_Found_Points.Domain_Name), plhs[0]);

}
/***************************************************************************************/
/*------------   END: Auto Generate ------------*/


/***************************************************************************************/
/* setup POINTS data to be output to MATLAB */
#define NUM_Points_Fieldnames (sizeof(Points_Fieldnames)/sizeof(*Points_Fieldnames))
void GPS::Init_Output_Data (mxArray* plhs[])               // output
{
    // declare constant arrays (see 'Generic_Point_Search.h')
    const char *Points_Fieldnames[] = {OUT_DATA_str, OUT_NAME_str};

    // declare parameters for outputing structures to MATLAB
    mwSize Points_dims[2] = {1, 1}; // just initialize to a 1x1 struct

    // set the number of MATLAB structs to create
    Points_dims[1] = NUM_PT_SEARCH;
    /*** setup LHS argument of MATLAB calling function (i.e. the output structs) ***/
    //                            2, 1xN struct,    X sub-fields,  field names
    plhs[0] = mxCreateStructArray(2, Points_dims, NUM_Points_Fieldnames, Points_Fieldnames);
}
/***************************************************************************************/


/***************************************************************************************/
/* output single domain point search data to MATLAB */
void GPS::Output_Single_Point_Data(mwIndex index, mxArray* Points_ptr, mxArray* Points_Name,   // input
                                   mxArray* mxOUT) // output
{
	// point the output MATLAB structure fields to the correct data blocks!
	mxSetFieldByNumber(mxOUT,index,mxGetFieldNumber(mxOUT,OUT_DATA_str), Points_ptr);
	mxSetFieldByNumber(mxOUT,index,mxGetFieldNumber(mxOUT,OUT_NAME_str), Points_Name);
}
/***************************************************************************************/

#undef GPS

/***/
