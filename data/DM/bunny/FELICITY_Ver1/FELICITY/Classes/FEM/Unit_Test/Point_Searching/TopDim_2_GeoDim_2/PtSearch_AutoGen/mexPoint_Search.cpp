/*
============================================================================================
   This is the main "gateway" routine that interfaces MATLAB with a custom point searching
   code (useful for finding interpolation points).

   See other accompanying files for more details.

   NOTE: this code was generated from the FELICITY package:
         Finite ELement Implementation and Computational Interface Tool for You

   OUTPUTS
   -------
   plhs[0] = POINTS(1).DATA{Cell_Indices, Local_Ref_Coord}, POINTS(1).Name,
             POINTS(2).DATA{Cell_Indices, Local_Ref_Coord}, POINTS(2).Name, etc...

   NOTE: portions of this code are automatically generated!

   WARNING!: Make sure all inputs to this mex function are NOT sparse matrices!!!
             Only FULL matrices are allowed as inputs!!!

   Copyright (c) 06-16-2014,  Shawn W. Walker
============================================================================================
*/

// default libraries you need
#include <algorithm>
#include <cstring>
#include <math.h>
#include <mex.h> // <-- This one is required

/*------------ BEGIN: Auto Generate ------------*/
// define input indices
#define PRHS_Omega_Mesh_Vertices                                               0
#define PRHS_Omega_Mesh_DoFmap                                                 1
#define PRHS_EMPTY_1                                                           2
#define PRHS_EMPTY_2                                                           3
#define PRHS_Omega_Search_Data                                                 4
/*------------   END: Auto Generate ------------*/

/* include classes and other sub-routines */
#include "Misc_Stuff.h"
#include "Generic_Point_Search.h"
#include "Generic_Point_Search.cc"

// note: 'prhs' represents the Right-Hand-Side arguments from MATLAB (inputs)
//       'plhs' represents the  Left-Hand-Side arguments from MATLAB (outputs)


/***************************************************************************************/
// define the "gateway" function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /*------------ BEGIN: Auto Generate ------------*/
    /* BEGIN: Error Checking */
    if (nrhs!=5)
        {
        mexPrintf("ERROR: 5 inputs required!\n");
        mexPrintf("\n");
        mexPrintf("      INPUTS                                                          ORDER \n");
        mexPrintf("      -------------------                                             ----- \n");
        mexPrintf("      Omega_Mesh_Vertices                                               0 \n");
        mexPrintf("      Omega_Mesh_DoFmap                                                 1 \n");
        mexPrintf("      EMPTY_1                                                           2 \n");
        mexPrintf("      EMPTY_2                                                           3 \n");
        mexPrintf("      The following inputs are 1x3 cell arrays: {Cell_Indices, Global_Coord, Neighbors} \n");
        mexPrintf("         Cell_Indices = initial guess for the enclosing cells of Global_Coord. \n");
        mexPrintf("                        (this can be an empty matrix) \n");
        mexPrintf("         Global_Coord = global coordinates of points to search for in the sub-Domain. \n");
        mexPrintf("         Neighbors    = neighbor data structure for the sub-Domain. \n");
        mexPrintf("      Omega_Search_Data                                                 4 \n");
        mexPrintf("\n");
        mexPrintf("      -------------------------------------------------\n");
        mexPrintf("      Output is an array of structs:\n");
        mexPrintf("      POINTS(:).DATA = {Cell_Indices, Local_Ref_Coord}\n");
        mexPrintf("      POINTS(:).Name = 'name of sub-Domain where points were found'\n");
        mexPrintf("\n");
        mexPrintf("      OUTPUTS For sub-Domains (in consecutive order) \n");
        mexPrintf("      -------------------------------------------------\n");
        mexPrintf("      Omega \n");
        mexPrintf("\n");
        mexErrMsgTxt("Check the number of input arguments!");
        }
    if (nlhs!=1) mexErrMsgTxt("1 output is required!");
    /* END: Error Checking */
    /*------------   END: Auto Generate ------------*/


    // declare the point search object
    Generic_Point_Search*   Pt_Search_obj;
    Pt_Search_obj = new Generic_Point_Search(prhs);

    /*** search for points! ***/
    Pt_Search_obj->Find_Points();

    // output found points back to MATLAB
    Pt_Search_obj->Init_Output_Data(plhs);
    Pt_Search_obj->Output_Points(plhs);

    delete(Pt_Search_obj);
}

/***/
