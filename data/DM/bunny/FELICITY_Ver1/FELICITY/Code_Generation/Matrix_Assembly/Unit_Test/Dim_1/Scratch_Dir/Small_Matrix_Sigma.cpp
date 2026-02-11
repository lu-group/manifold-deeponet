/***************************************************************************************/
/* Tabulate the tensor for the local element contribution */
void SpecificFEM::Tabulate_Tensor(const CLASS_geom_Sigma_embedded_in_Sigma_restricted_to_Sigma& Mesh)
{
    const unsigned int NQ = 8;

    // Compute single entry using quadrature

    // Loop quadrature points for integral
    double  A0_value = 0.0; // initialize
    double  A1_value = 0.0; // initialize
    double  A2_value = 0.0; // initialize
    for (unsigned int qp = 0; qp < NQ; qp++)
        {
        const double  t2 = old_soln_restricted_to_Sigma->Func_f_Value[0][qp].a-exp(geom_Sigma_embedded_in_Sigma_restricted_to_Sigma->Map_PHI[qp].v[0]);
        const double  integrand_0 = t2*t2;
        const double  integrand_1 = my_f_restricted_to_Sigma->Func_f_d_ds[0][qp].a;
        const double  integrand_2 = old_soln_restricted_to_Sigma->Func_f_d_ds[0][qp].a;
        A0_value += integrand_0 * Mesh.Map_Det_Jac_w_Weight[qp].a;
        A1_value += integrand_1 * Mesh.Map_Det_Jac_w_Weight[qp].a;
        A2_value += integrand_2 * Mesh.Map_Det_Jac_w_Weight[qp].a;
        }
    FE_Tensor_0[0] = A0_value;
    FE_Tensor_1[0] = A1_value;
    FE_Tensor_2[0] = A2_value;
}
/***************************************************************************************/
