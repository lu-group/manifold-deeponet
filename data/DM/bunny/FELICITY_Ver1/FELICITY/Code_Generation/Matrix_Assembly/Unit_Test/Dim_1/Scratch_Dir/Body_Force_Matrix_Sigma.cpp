/***************************************************************************************/
/* Tabulate the tensor for the local element contribution */
void SpecificFEM::Tabulate_Tensor(const CLASS_geom_Sigma_embedded_in_Sigma_restricted_to_Sigma& Mesh)
{
    const unsigned int NQ = 8;

    // Compute element tensor using quadrature

    // Loop quadrature points for integral
    for (unsigned int i = 0; i < ROW_NB; i++)
        {
        double  A0_value = 0.0; // initialize
        for (unsigned int qp = 0; qp < NQ; qp++)
            {
            const double  integrand_0 = (*Scalar_P2_phi_restricted_to_Sigma->Func_f_Value)[i][qp].a*foo(geom_Sigma_embedded_in_Sigma_restricted_to_Sigma->Map_PHI[qp].v[2]*3.141592653589793);
            A0_value += integrand_0 * Mesh.Map_Det_Jac_w_Weight[qp].a;
            }
        FE_Tensor_0[i] = A0_value;
        }
}
/***************************************************************************************/
