/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_GemCarModel_H_
#define ACADOS_SOLVER_GemCarModel_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define GEMCARMODEL_NX     5
#define GEMCARMODEL_NZ     0
#define GEMCARMODEL_NU     2
#define GEMCARMODEL_NP     0
#define GEMCARMODEL_NBX    5
#define GEMCARMODEL_NBX0   5
#define GEMCARMODEL_NBU    2
#define GEMCARMODEL_NSBX   0
#define GEMCARMODEL_NSBU   0
#define GEMCARMODEL_NSH    1
#define GEMCARMODEL_NSH0   0
#define GEMCARMODEL_NSG    0
#define GEMCARMODEL_NSPHI  0
#define GEMCARMODEL_NSHN   0
#define GEMCARMODEL_NSGN   0
#define GEMCARMODEL_NSPHIN 0
#define GEMCARMODEL_NSPHI0 0
#define GEMCARMODEL_NSBXN  0
#define GEMCARMODEL_NS     1
#define GEMCARMODEL_NS0    0
#define GEMCARMODEL_NSN    0
#define GEMCARMODEL_NG     0
#define GEMCARMODEL_NBXN   0
#define GEMCARMODEL_NGN    0
#define GEMCARMODEL_NY0    7
#define GEMCARMODEL_NY     7
#define GEMCARMODEL_NYN    5
#define GEMCARMODEL_N      40
#define GEMCARMODEL_NH     1
#define GEMCARMODEL_NHN    0
#define GEMCARMODEL_NH0    0
#define GEMCARMODEL_NPHI0  0
#define GEMCARMODEL_NPHI   0
#define GEMCARMODEL_NPHIN  0
#define GEMCARMODEL_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct GemCarModel_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics

    external_function_param_casadi *forw_vde_casadi;
    external_function_param_casadi *expl_ode_fun;




    // cost

    external_function_param_casadi *cost_y_fun;
    external_function_param_casadi *cost_y_fun_jac_ut_xt;
    external_function_param_casadi *cost_y_hess;



    external_function_param_casadi cost_y_0_fun;
    external_function_param_casadi cost_y_0_fun_jac_ut_xt;
    external_function_param_casadi cost_y_0_hess;



    external_function_param_casadi cost_y_e_fun;
    external_function_param_casadi cost_y_e_fun_jac_ut_xt;
    external_function_param_casadi cost_y_e_hess;


    // constraints
    external_function_param_casadi *nl_constr_h_fun_jac;
    external_function_param_casadi *nl_constr_h_fun;







} GemCarModel_solver_capsule;

ACADOS_SYMBOL_EXPORT GemCarModel_solver_capsule * GemCarModel_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int GemCarModel_acados_free_capsule(GemCarModel_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int GemCarModel_acados_create(GemCarModel_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int GemCarModel_acados_reset(GemCarModel_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of GemCarModel_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int GemCarModel_acados_create_with_discretization(GemCarModel_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int GemCarModel_acados_update_time_steps(GemCarModel_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int GemCarModel_acados_update_qp_solver_cond_N(GemCarModel_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int GemCarModel_acados_update_params(GemCarModel_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int GemCarModel_acados_update_params_sparse(GemCarModel_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);

ACADOS_SYMBOL_EXPORT int GemCarModel_acados_solve(GemCarModel_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void GemCarModel_acados_batch_solve(GemCarModel_solver_capsule ** capsules, int N_batch);
ACADOS_SYMBOL_EXPORT int GemCarModel_acados_free(GemCarModel_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void GemCarModel_acados_print_stats(GemCarModel_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int GemCarModel_acados_custom_update(GemCarModel_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *GemCarModel_acados_get_nlp_in(GemCarModel_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *GemCarModel_acados_get_nlp_out(GemCarModel_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *GemCarModel_acados_get_sens_out(GemCarModel_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *GemCarModel_acados_get_nlp_solver(GemCarModel_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *GemCarModel_acados_get_nlp_config(GemCarModel_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *GemCarModel_acados_get_nlp_opts(GemCarModel_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *GemCarModel_acados_get_nlp_dims(GemCarModel_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *GemCarModel_acados_get_nlp_plan(GemCarModel_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_GemCarModel_H_
