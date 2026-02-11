/***************************************************************************************/
/* Tabulate the tensor for the local element contribution */
void SpecificFEM::Tabulate_Tensor(const CLASS_geom_Gamma_embedded_in_Omega_restricted_to_Gamma& Mesh)
{
    const unsigned int NQ = 5;

    // Compute element tensor using quadrature

    // Loop quadrature points for integral
    for (unsigned int j = 0; j < COL_NB; j++)
        {
        for (unsigned int i = j; i < ROW_NB; i++)
            {
            double  A0_value = 0.0; // initialize
            for (unsigned int qp = 0; qp < NQ; qp++)
                {
                const double  integrand_0 = (*M_Space_phi_restricted_to_Gamma->Func_f_Value)[j][qp].a*(*M_Space_phi_restricted_to_Gamma->Func_f_Value)[i][qp].a;
                A0_value += integrand_0 * Mesh.Map_Det_Jac_w_Weight[qp].a;
                }
            FE_Tensor_0[j*ROW_NB + i] = A0_value;
            }
        }

    // Copy the lower triangular entries to the upper triangular part (by symmetry)
    for (unsigned int j = 0; j < COL_NB; j++)
        {
        for (unsigned int i = j+1; i < ROW_NB; i++)
            {
            FE_Tensor_0[i*ROW_NB + j] = FE_Tensor_0[j*ROW_NB + i];
            }
        }
}
/***************************************************************************************/
