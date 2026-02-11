/***************************************************************************************/
/* Tabulate the tensor for the local element contribution */
void SpecificFEM::Tabulate_Tensor(const CLASS_geom_Gamma_embedded_in_Gamma_restricted_to_Gamma& Mesh)
{
    const unsigned int NQ = 3;

    // Compute element tensor using quadrature

    // Loop quadrature points for integral
    for (unsigned int j = 0; j < COL_NB; j++)
        {
        for (unsigned int i = j; i < ROW_NB; i++)
            {
            double  A0_value = 0.0; // initialize
            for (unsigned int qp = 0; qp < NQ; qp++)
                {
                const double  t2 = geom_Gamma_embedded_in_Gamma_restricted_to_Gamma->Map_PHI[qp].v[0]*geom_Gamma_embedded_in_Gamma_restricted_to_Gamma->Map_PHI[qp].v[0];
                const double  t3 = geom_Gamma_embedded_in_Gamma_restricted_to_Gamma->Map_PHI[qp].v[1]*geom_Gamma_embedded_in_Gamma_restricted_to_Gamma->Map_PHI[qp].v[1];
                const double  t4 = geom_Gamma_embedded_in_Gamma_restricted_to_Gamma->Map_PHI[qp].v[1]*8.0107575696612E-1;
                const double  t5 = geom_Gamma_embedded_in_Gamma_restricted_to_Gamma->Map_PHI[qp].v[0]*6.10307030069875E-1;
                const double  t6 = t2*9.40336654420907E-1;
                const double  t7 = t3*5.83432457237087E-3;
                const double  t8 = t4+t5+t6+t7+3.232981533413648;
                const double  integrand_0 = Vh_phi_restricted_to_Gamma->Func_f_Grad[j][qp].v[0]*Vh_phi_restricted_to_Gamma->Func_f_Grad[i][qp].v[0]*t8+Vh_phi_restricted_to_Gamma->Func_f_Grad[j][qp].v[1]*Vh_phi_restricted_to_Gamma->Func_f_Grad[i][qp].v[1]*t8+Vh_phi_restricted_to_Gamma->Func_f_Grad[j][qp].v[2]*Vh_phi_restricted_to_Gamma->Func_f_Grad[i][qp].v[2]*t8;
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
