/***************************************************************************************/
/* Tabulate the tensor for the local element contribution */
void SpecificFEM::Tabulate_Tensor(const CLASS_geom_Gamma_embedded_in_Gamma_restricted_to_Gamma& Mesh)
{
    const unsigned int NQ = 19;

    // Compute element tensor using quadrature

    // Loop quadrature points for integral
    for (unsigned int i = 0; i < ROW_NB; i++)
        {
        double  A0_value = 0.0; // initialize
        for (unsigned int qp = 0; qp < NQ; qp++)
            {
            const double  integrand_0 = (*Scalar_P2_phi_restricted_to_Gamma->Func_f_Value)[i][qp].a*cos(geom_Gamma_embedded_in_Gamma_restricted_to_Gamma->Map_PHI[qp].v[1])*sin(geom_Gamma_embedded_in_Gamma_restricted_to_Gamma->Map_PHI[qp].v[0])*sin(geom_Gamma_embedded_in_Gamma_restricted_to_Gamma->Map_PHI[qp].v[2]);
            A0_value += integrand_0 * Mesh.Map_Det_Jac_w_Weight[qp].a;
            }
        FE_Tensor_0[i] = A0_value;
        }
}
/***************************************************************************************/
