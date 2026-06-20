/* lincoa_prima_driver.c
 *
 * Generates LINCOA reference fixtures for basin's PRIMA cross-validation
 * (docs/newuoa-roadmap.md). PRIMA is the BSD-3 C/Fortran translation of
 * Powell's solvers, vendored at tools/prima (submodule, v0.7.2). This driver
 * links libprima, runs PRIMA's LINCOA on a chosen linearly-constrained test
 * problem, and dumps a self-describing fixture to stdout. See ../README.md for
 * the exact build/link/run commands. CI never rebuilds the fixtures; the
 * committed .tsv files are the artifacts.
 *
 * Usage:
 *   lincoa_prima_driver <problem>
 *     problem in { proj2, crosen2, cquad3 }
 *
 * The problems are linear-inequality-constrained only (no box, no equalities):
 * xl/xu are passed as ±INFINITY so PRIMA's get_lincon drops them, leaving the
 * folded constraint system equal to the explicit `A x <= b` — which is exactly
 * what basin's `Lincoa` folds, so the two solvers see the same feasible region.
 *
 * Output (whitespace-separated, %.17e for full f64 round-trip):
 *   # config problem=<p> n=<n> m_ineq=<m> rho_beg=<..> rho_end=<..> maxfun=<..> npt=<..>
 *   # x0 <x0_0> ... <x0_{n-1}>
 *   # aineq <row-major Aineq, m_ineq*n entries>
 *   # bineq <b_0> ... <b_{m-1}>
 *   <evalindex> <f> <x_0> ... <x_{n-1}>   (one row per objective call, PRIMA's
 *   ...                                     evaluation order; 0-based index)
 *   # final nf=<nf> rc=<rc> f=<f> cstrv=<cstrv> x= <x_0> ... <x_{n-1}>
 *
 * IMPORTANT: the objective + constraint definitions below MUST stay textually
 * mirrored with the Rust definitions in crates/basin/src/solver/lincoa/parity.rs
 * (same coefficients, same constraint rows, same index base). The parity test
 * recomputes the objective at every traced point and rebuilds the same problem,
 * so any drift here surfaces as a test failure.
 */
#include "prima/prima.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KIND_PROJ2 0
#define KIND_CROSEN2 1
#define KIND_CQUAD3 2

static int g_n;
static int g_kind;
static int g_eval;

static double objective(const double x[], int n) {
    int i;
    double s = 0.0;
    switch (g_kind) {
    case KIND_PROJ2: /* (x0-2)^2 + (x1-2)^2 */
        return (x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 2.0) * (x[1] - 2.0);
    case KIND_CROSEN2: /* (1-x0)^2 + 100 (x1 - x0^2)^2 */
    {
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    }
    case KIND_CQUAD3: /* sum (x_i - 2)^2 */
        for (i = 0; i < n; i++) {
            s += (x[i] - 2.0) * (x[i] - 2.0);
        }
        return s;
    }
    return 0.0; /* unreachable */
}

static void calfun(const double x[], double *f, const void *data) {
    int i;
    (void)data;
    *f = objective(x, g_n);
    printf("%d %.17e", g_eval, *f);
    for (i = 0; i < g_n; i++) {
        printf(" %.17e", x[i]);
    }
    printf("\n");
    g_eval++;
}

int main(int argc, char **argv) {
    int i, n, m_ineq, npt, maxfun, nf, iprint, rc;
    double rho_beg, rho_end, ftarget, f, cstrv;
    double *x0, *x, *xl, *xu, *Aineq, *bineq;
    const char *prob;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <problem>\n", argv[0]);
        return 2;
    }
    prob = argv[1];
    rho_end = 1e-6;
    rho_beg = 0.5;

    if (strcmp(prob, "proj2") == 0) {
        g_kind = KIND_PROJ2;
        n = 2;
        m_ineq = 1;
        Aineq = (double *)malloc(sizeof(double) * (size_t)(m_ineq * n));
        bineq = (double *)malloc(sizeof(double) * (size_t)m_ineq);
        Aineq[0] = 1.0;
        Aineq[1] = 1.0; /* x0 + x1 <= 2 */
        bineq[0] = 2.0;
    } else if (strcmp(prob, "crosen2") == 0) {
        g_kind = KIND_CROSEN2;
        n = 2;
        m_ineq = 1;
        Aineq = (double *)malloc(sizeof(double) * (size_t)(m_ineq * n));
        bineq = (double *)malloc(sizeof(double) * (size_t)m_ineq);
        Aineq[0] = 1.0;
        Aineq[1] = 0.0; /* x0 <= 0.5 */
        bineq[0] = 0.5;
    } else if (strcmp(prob, "cquad3") == 0) {
        g_kind = KIND_CQUAD3;
        n = 3;
        m_ineq = 1;
        Aineq = (double *)malloc(sizeof(double) * (size_t)(m_ineq * n));
        bineq = (double *)malloc(sizeof(double) * (size_t)m_ineq);
        Aineq[0] = 1.0;
        Aineq[1] = 1.0;
        Aineq[2] = 1.0; /* x0 + x1 + x2 <= 3 */
        bineq[0] = 3.0;
    } else {
        fprintf(stderr, "unknown problem: %s\n", prob);
        return 2;
    }

    x0 = (double *)malloc(sizeof(double) * (size_t)n);
    x = (double *)malloc(sizeof(double) * (size_t)n);
    xl = (double *)malloc(sizeof(double) * (size_t)n);
    xu = (double *)malloc(sizeof(double) * (size_t)n);
    for (i = 0; i < n; i++) {
        x0[i] = 0.0; /* feasible start for all three problems */
        x[i] = 0.0;
        xl[i] = -INFINITY; /* unbounded: dropped by get_lincon */
        xu[i] = INFINITY;
    }

    g_n = n;
    g_eval = 0;
    npt = 2 * n + 1;
    maxfun = 500 * n;
    ftarget = -INFINITY;
    iprint = PRIMA_MSG_NONE;
    f = 0.0;
    cstrv = 0.0;
    nf = 0;

    printf("# config problem=%s n=%d m_ineq=%d rho_beg=%.17e rho_end=%.17e maxfun=%d npt=%d\n", prob,
           n, m_ineq, rho_beg, rho_end, maxfun, npt);
    printf("# x0");
    for (i = 0; i < n; i++) {
        printf(" %.17e", x0[i]);
    }
    printf("\n# aineq");
    for (i = 0; i < m_ineq * n; i++) {
        printf(" %.17e", Aineq[i]);
    }
    printf("\n# bineq");
    for (i = 0; i < m_ineq; i++) {
        printf(" %.17e", bineq[i]);
    }
    printf("\n");

    rc = prima_lincoa(&calfun, NULL, n, x, &f, &cstrv, m_ineq, Aineq, bineq, 0, NULL, NULL, xl, xu,
                      &nf, rho_beg, rho_end, ftarget, maxfun, npt, iprint);

    printf("# final nf=%d rc=%d f=%.17e cstrv=%.17e x=", nf, rc, f, cstrv);
    for (i = 0; i < n; i++) {
        printf(" %.17e", x[i]);
    }
    printf("\n");

    free(x0);
    free(x);
    free(xl);
    free(xu);
    free(Aineq);
    free(bineq);
    return 0;
}
