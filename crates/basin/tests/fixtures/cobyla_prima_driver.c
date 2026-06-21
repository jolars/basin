/* cobyla_prima_driver.c
 *
 * Generates COBYLA reference fixtures for basin's PRIMA cross-validation
 * (docs/newuoa-roadmap.md). PRIMA is the BSD-3 C/Fortran translation of
 * Powell's solvers, vendored at tools/prima (submodule, v0.7.2). This driver
 * links libprima, runs PRIMA's COBYLA on a chosen nonlinearly-constrained test
 * problem, and dumps a self-describing fixture to stdout. See ../README.md for
 * the exact build/link/run commands. CI never rebuilds the fixtures; the
 * committed .tsv files are the artifacts.
 *
 * Usage:
 *   cobyla_prima_driver <problem>
 *     problem in { disk, fletcher, ballsphere }
 *
 * The problems are nonlinear-inequality-constrained only (no box, no linear
 * blocks): m_ineq = m_eq = 0, xl/xu = ±INFINITY, so every constraint flows
 * through PRIMA's nonlinear block `constr(x) <= 0` — exactly the trait-only
 * path basin's `Cobyla` drives (its m_lcon = 0). The sign convention is
 * `constr(x) <= 0` feasible, matching basin's NonlinearInequalityConstraints;
 * Powell's 1994 paper writes `c >= 0`, so basin's c is Powell's -c (no flip is
 * needed here — the problems below are already in <= 0 form).
 *
 * Output (whitespace-separated, %.17e for full f64 round-trip):
 *   # config problem=<p> n=<n> m_nlcon=<m> rho_beg=<..> rho_end=<..> maxfun=<..>
 *   # x0 <x0_0> ... <x0_{n-1}>
 *   <evalindex> <f> <cstrv> <x_0> ... <x_{n-1}>   (one row per calcfc call,
 *   ...                                            PRIMA's order; 0-based index;
 *                                                  cstrv = max(0, max_i constr_i))
 *   # final nf=<nf> rc=<rc> f=<f> cstrv=<cstrv> x= <x_0> ... <x_{n-1}>
 *
 * Unlike the LINCOA/NEWUOA fixtures there is no `npt` field: COBYLA's simplex
 * is always n+1 vertices. The per-eval `cstrv` column lets the parity test
 * guard both the objective AND the constraint functions against C<->Rust drift.
 *
 * IMPORTANT: the objective + constraint definitions below MUST stay textually
 * mirrored with the Rust definitions in crates/basin/src/solver/cobyla/parity.rs
 * (same coefficients, same constraint rows, same <= 0 signs, same index base).
 * The parity test recomputes both at every traced point and rebuilds the same
 * problem, so any drift here surfaces as a test failure.
 */
#include "prima/prima.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KIND_DISK 0
#define KIND_FLETCHER 1
#define KIND_BALLSPHERE 2

static int g_n;
static int g_m;
static int g_kind;
static int g_eval;

/* Objective F(x). Mirrored with `objective` in parity.rs. */
static double objective(const double x[], int n) {
    int i;
    double s = 0.0;
    switch (g_kind) {
    case KIND_DISK: /* x0 * x1 */
        return x[0] * x[1];
    case KIND_FLETCHER: /* -x0 - x1 */
        return -x[0] - x[1];
    case KIND_BALLSPHERE: /* sum (x_i - 2)^2 */
        for (i = 0; i < n; i++) {
            s += (x[i] - 2.0) * (x[i] - 2.0);
        }
        return s;
    }
    return 0.0; /* unreachable */
}

/* Constraints constr(x) <= 0, length g_m. Mirrored with `constraints` in
 * parity.rs. */
static void constraints(const double x[], double constr[]) {
    switch (g_kind) {
    case KIND_DISK: /* x0^2 + x1^2 - 1 <= 0 */
        constr[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
        break;
    case KIND_FLETCHER: /* x0^2 - x1 <= 0 ; x0^2 + x1^2 - 1 <= 0 */
        constr[0] = x[0] * x[0] - x[1];
        constr[1] = x[0] * x[0] + x[1] * x[1] - 1.0;
        break;
    case KIND_BALLSPHERE: /* sum x_i^2 - 1 <= 0 */
    {
        int i;
        double s = 0.0;
        for (i = 0; i < g_n; i++) {
            s += x[i] * x[i];
        }
        constr[0] = s - 1.0;
        break;
    }
    }
}

static void calcfc(const double x[], double *f, double constr[], const void *data) {
    int i;
    double cstrv = 0.0;
    (void)data;
    *f = objective(x, g_n);
    constraints(x, constr);
    for (i = 0; i < g_m; i++) {
        if (constr[i] > cstrv) {
            cstrv = constr[i];
        }
    }
    printf("%d %.17e %.17e", g_eval, *f, cstrv);
    for (i = 0; i < g_n; i++) {
        printf(" %.17e", x[i]);
    }
    printf("\n");
    g_eval++;
}

int main(int argc, char **argv) {
    int i, n, m_nlcon, maxfun, nf, iprint, rc;
    double rho_beg, rho_end, ftarget, f, cstrv;
    double *x0, *x, *xl, *xu, *nlconstr;
    const char *prob;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <problem>\n", argv[0]);
        return 2;
    }
    prob = argv[1];
    rho_end = 1e-6;
    rho_beg = 0.5;

    if (strcmp(prob, "disk") == 0) {
        g_kind = KIND_DISK;
        n = 2;
        m_nlcon = 1;
    } else if (strcmp(prob, "fletcher") == 0) {
        g_kind = KIND_FLETCHER;
        n = 2;
        m_nlcon = 2;
    } else if (strcmp(prob, "ballsphere") == 0) {
        g_kind = KIND_BALLSPHERE;
        n = 3;
        m_nlcon = 1;
    } else {
        fprintf(stderr, "unknown problem: %s\n", prob);
        return 2;
    }

    x0 = (double *)malloc(sizeof(double) * (size_t)n);
    x = (double *)malloc(sizeof(double) * (size_t)n);
    xl = (double *)malloc(sizeof(double) * (size_t)n);
    xu = (double *)malloc(sizeof(double) * (size_t)n);
    nlconstr = (double *)malloc(sizeof(double) * (size_t)m_nlcon);
    for (i = 0; i < n; i++) {
        /* Strictly feasible interior start 0.5 * e (||x0|| < 1; for fletcher
         * x0^2 = 0.25 < x1 = 0.5). Overridden below for disk. */
        x0[i] = 0.5;
        xl[i] = -INFINITY; /* unbounded: no box block */
        xu[i] = INFINITY;
    }
    if (g_kind == KIND_DISK) {
        /* disk has two sign-symmetric minima (+,-) and (-,+); start off the
         * symmetry axis so PRIMA and basin deterministically share one basin. */
        x0[0] = 0.7;
        x0[1] = -0.3;
    }
    for (i = 0; i < n; i++) {
        x[i] = x0[i];
    }

    g_n = n;
    g_m = m_nlcon;
    g_eval = 0;
    maxfun = 500 * n;
    ftarget = -INFINITY;
    iprint = PRIMA_MSG_NONE;
    f = 0.0;
    cstrv = 0.0;
    nf = 0;

    printf("# config problem=%s n=%d m_nlcon=%d rho_beg=%.17e rho_end=%.17e maxfun=%d\n", prob, n,
           m_nlcon, rho_beg, rho_end, maxfun);
    printf("# x0");
    for (i = 0; i < n; i++) {
        printf(" %.17e", x0[i]);
    }
    printf("\n");

    rc = prima_cobyla(m_nlcon, &calcfc, NULL, n, x, &f, &cstrv, nlconstr, 0, NULL, NULL, 0, NULL,
                      NULL, xl, xu, &nf, rho_beg, rho_end, ftarget, maxfun, iprint);

    printf("# final nf=%d rc=%d f=%.17e cstrv=%.17e x=", nf, rc, f, cstrv);
    for (i = 0; i < n; i++) {
        printf(" %.17e", x[i]);
    }
    printf("\n");

    free(x0);
    free(x);
    free(xl);
    free(xu);
    free(nlconstr);
    return 0;
}
