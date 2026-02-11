/***************************************************************************************/
/* Tabulate the tensor for the local element contribution */
void SpecificFEM::Tabulate_Tensor(const CLASS_geom_Omega_embedded_in_Omega_restricted_to_Omega& Mesh)
{
    const unsigned int NQ = 6;

    // Compute element tensor using quadrature

    // Loop quadrature points for integral
    for (unsigned int i = 0; i < ROW_NB; i++)
        {
        double  A0_value = 0.0; // initialize
        for (unsigned int qp = 0; qp < NQ; qp++)
            {
            const double  integrand_0 = (*P0_phi_restricted_to_Omega->Func_f_Value)[i][qp].a*sin(geom_Omega_embedded_in_Omega_restricted_to_Omega->Map_PHI[qp].v[0]*3.141592653589793)*sin(geom_Omega_embedded_in_Omega_restricted_to_Omega->Map_PHI[qp].v[1]*3.141592653589793)*1.973920880217872E1;
            A0_value += integrand_0 * Mesh.Map_Det_Jac_w_Weight[qp].a;
            }
        FE_Tensor_0[i] = A0_value;
        }
}
/***************************************************************************************/
