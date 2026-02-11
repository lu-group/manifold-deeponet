/***************************************************************************************/
/* Tabulate the tensor for the local element contribution */
void SpecificFEM::Tabulate_Tensor(const CLASS_geom_Gamma_embedded_in_Omega_restricted_to_Gamma& Mesh)
{
    const unsigned int NQ = 5;

    // Compute single entry using quadrature

    // Loop quadrature points for integral
    double  A0_value = 0.0; // initialize
    double  A1_value = 0.0; // initialize
    double  A2_value = 0.0; // initialize
    double  A3_value = 0.0; // initialize
    for (unsigned int qp = 0; qp < NQ; qp++)
        {
        const double  integrand_0 = geom_Gamma_embedded_in_Omega_restricted_to_Gamma->Map_PHI[qp].v[0];
        const double  integrand_1 = sin(geom_Gamma_embedded_in_Omega_restricted_to_Gamma->Map_PHI[qp].v[0]);
        const double  integrand_2 = sqrt(2.0)*(geom_Gamma_embedded_in_Omega_restricted_to_Gamma->Map_Normal_Vector[0].v[0]+geom_Gamma_embedded_in_Omega_restricted_to_Gamma->Map_Normal_Vector[0].v[1])*(1.0/2.0);
        const double  integrand_3 = geom_Gamma_embedded_in_Omega_restricted_to_Gamma->Map_Normal_Vector[0].v[0]*geom_Gamma_embedded_in_Omega_restricted_to_Gamma->Map_Tangent_Vector[0].v[0]+geom_Gamma_embedded_in_Omega_restricted_to_Gamma->Map_Normal_Vector[0].v[1]*geom_Gamma_embedded_in_Omega_restricted_to_Gamma->Map_Tangent_Vector[0].v[1];
        A0_value += integrand_0 * Mesh.Map_Det_Jac_w_Weight[qp].a;
        A1_value += integrand_1 * Mesh.Map_Det_Jac_w_Weight[qp].a;
        A2_value += integrand_2 * Mesh.Map_Det_Jac_w_Weight[qp].a;
        A3_value += integrand_3 * Mesh.Map_Det_Jac_w_Weight[qp].a;
        }
    FE_Tensor_0[0] = A0_value;
    FE_Tensor_1[0] = A1_value;
    FE_Tensor_2[0] = A2_value;
    FE_Tensor_3[0] = A3_value;
}
/***************************************************************************************/
