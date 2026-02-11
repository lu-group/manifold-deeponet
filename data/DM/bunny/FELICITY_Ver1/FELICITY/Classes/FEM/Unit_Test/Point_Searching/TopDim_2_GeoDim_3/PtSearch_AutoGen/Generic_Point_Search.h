/*
============================================================================================
   Header file for a C++ Class that contains methods for generic point searching
   of meshes.

   NOTE: portions of this code are automatically generated!

   Copyright (c) 06-16-2014,  Shawn W. Walker
============================================================================================
*/

/*------------ BEGIN: Auto Generate ------------*/
// set the number of point searches to perform
#define NUM_PT_SEARCH    1

// In MATLAB, the output (POINTS) point data should look like:
//            POINTS.DATA
//                  .Name
//
// Here, we define the strings that makes these variable names
#define OUT_DATA_str    "DATA"
#define OUT_NAME_str    "Name"
/*------------   END: Auto Generate ------------*/

/***************************************************************************************/
/*** C++ class ***/
class Generic_Point_Search
{
public:
    //Generic_Point_Search (); // constructor
    Generic_Point_Search (const mxArray *[]); // constructor
    ~Generic_Point_Search (); // DE-structor


    void Setup_Data (const mxArray*[]);
    void Find_Points ();
    void Output_Points (mxArray*[]);
    void Init_Output_Data (mxArray*[]);
    void Output_Single_Point_Data (mwIndex, mxArray*, mxArray*, mxArray*);

private:
    // these variables are defined from inputs coming from MATLAB

    /*------------ BEGIN: Auto Generate ------------*/
    // classes for (sub)domain(s) and topological entities
    CLASS_Domain_Gamma_embedded_in_Gamma_restricted_to_Gamma    Domain_Gamma_embedded_in_Gamma_restricted_to_Gamma;

    // mesh geometry classes (can be higher order)
    CLASS_geom_Gamma_embedded_in_Gamma_restricted_to_Gamma   geom_Gamma_embedded_in_Gamma_restricted_to_Gamma;

    // classes for search data and found points on subdomains
    Subdomain_Search_Data_Class    Gamma_Search_Data;
    Unstructured_Local_Points_Class    Gamma_Found_Points;

    // pointers to a search object for each domain to be searched
    CLASS_Search_Gamma*    Gamma_Search_Obj;
    /*------------   END: Auto Generate ------------*/
};

/***/
