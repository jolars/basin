/* bobyqa_prima_driver.c
 *
 * Generates BOBYQA reference fixtures for basin's PRIMA cross-validation
 * (docs/newuoa-roadmap.md). PRIMA is the BSD-3 C/Fortran translation of
 * Powell's solvers, vendored at tools/prima (submodule, v0.7.2). This driver
 * links libprima, runs PRIMA's BOBYQA (bound-constrained) on a chosen test
 * problem, and dumps a self-describing fixture to stdout. See ../README.md for
 * the exact build/link/run commands. CI never rebuilds the fixtures; the
 * committed .tsv files are the artifacts.
 *
 * BOBYQA is the bound-constrained sibling of NEWUOA, so this driver mirrors
 * newuoa_prima_driver.c but calls `prima_bobyqa` (which takes xl/xu) and
 * records the per-problem box in two extra metadata lines (`# xl` / `# xu`).
 *
 * Usage:
 *   bobyqa_prima_driver <problem> <n>
 *     problem in { rosenbrock, sphere, chrosen }
 *
 * Output (whitespace-separated, %.17e for full f64 round-trip):
 *   # config problem=<p> n=<n> rho_beg=<..> rho_end=<..> maxfun=<..> npt=<..>
 *   # x0 <x0_0> ... <x0_{n-1}>
 *   # xl <xl_0> ... <xl_{n-1}>
 *   # xu <xu_0> ... <xu_{n-1}>
 *   <evalindex> <f> <x_0> ... <x_{n-1}>   (one row per objective call, PRIMA's
 *   ...                                     evaluation order; 0-based index)
 *   # final nf=<nf> rc=<rc> f=<f> x= <x_0> ... <x_{n-1}>
 *
 * PRIMA funnels every objective call through `evaluatef` exactly once
 * (tools/prima/fortran/common/evaluate.f90), so the static eval counter here
 * stays in lockstep with PRIMA's reported `nf`. The moderated-extreme-barrier
 * clamping in `evaluatef` is a no-op for these smooth, well-scaled problems.
 *
 * IMPORTANT: the objective definitions below MUST stay textually mirrored with
 * the Rust fns in crates/basin/src/solver/bobyqa/parity.rs (same summation
 * order, same coefficients, same index base). The parity test recomputes the
 * objective at every traced point and asserts agreement, so any drift here
 * surfaces as a test failure.
 */
#include "prima/prima.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KIND_ROSENBROCK 0
#define KIND_SPHERE 1
#define KIND_CHROSEN 2

static int g_n;    /* problem dimension                       */
static int g_kind; /* one of the KIND_* selectors             */
static int g_eval; /* monotonic 0-based objective-call index  */

/* The three test objectives. Mirror exactly in parity.rs. */
static double objective(const double x[], int n) {
    int i;
    double s = 0.0;
    switch (g_kind) {
    case KIND_ROSENBROCK: /* basin coefficient form: sum 100 (x_{i+1}-x_i^2)^2 + (1-x_i)^2 */
        for (i = 0; i < n - 1; i++) {
            double t = x[i + 1] - x[i] * x[i];
            s += 100.0 * t * t + (1.0 - x[i]) * (1.0 - x[i]);
        }
        return s;
    case KIND_SPHERE: /* shifted sphere: sum (x_i - 3)^2 (min at 3, outside the box) */
        for (i = 0; i < n; i++) {
            double d = x[i] - 3.0;
            s += d * d;
        }
        return s;
    case KIND_CHROSEN: /* chrosen.f90 form: sum (x_i-1)^2 + 100 (x_{i+1}-x_i^2)^2 */
        for (i = 0; i < n - 1; i++) {
            double a = x[i] - 1.0;
            double b = x[i + 1] - x[i] * x[i];
            s += a * a + 100.0 * b * b;
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
    int i, n, npt, maxfun, nf, iprint, rc;
    double rho_beg, rho_end, ftarget, f;
    double *x0, *x, *xl, *xu;
    const char *prob;

    if (argc != 3) {
        fprintf(stderr, "usage: %s <problem> <n>\n", argv[0]);
        return 2;
    }
    prob = argv[1];
    n = atoi(argv[2]);
    if (n < 2) {
        fprintf(stderr, "n must be >= 2 (got %d)\n", n);
        return 2;
    }

    x0 = (double *)malloc(sizeof(double) * (size_t)n);
    xl = (double *)malloc(sizeof(double) * (size_t)n);
    xu = (double *)malloc(sizeof(double) * (size_t)n);
    rho_end = 1e-6;

    if (strcmp(prob, "rosenbrock") == 0) {
        /* Interior minimizer (1,..,1); box [-5,5]^n leaves >= 2*rho_beg slack
         * on every coordinate, so the initial design is the plain coordinate
         * cross and bounds never bind: confirms boxing doesn't perturb the
         * unconstrained trajectory. */
        g_kind = KIND_ROSENBROCK;
        for (i = 0; i < n; i++) {
            x0[i] = (i % 2 == 0) ? -1.2 : 1.0;
            xl[i] = -5.0;
            xu[i] = 5.0;
        }
        rho_beg = 0.5;
    } else if (strcmp(prob, "sphere") == 0) {
        /* Unconstrained minimizer (3,..,3) is OUTSIDE the box [-2,2]^n, so the
         * solution is the active corner (2,..,2): exercises TRSBOX active-set
         * and bound-aware init/ALTMOV. */
        g_kind = KIND_SPHERE;
        for (i = 0; i < n; i++) {
            x0[i] = 0.0;
            xl[i] = -2.0;
            xu[i] = 2.0;
        }
        rho_beg = 0.5;
    } else if (strcmp(prob, "chrosen") == 0) {
        /* Chained Rosenbrock; interior minimizer (1,..,1) with a wide box
         * [-10,10]^n that never binds: dimensional-scaling check. */
        g_kind = KIND_CHROSEN;
        for (i = 0; i < n; i++) {
            x0[i] = -1.0;
            xl[i] = -10.0;
            xu[i] = 10.0;
        }
        rho_beg = 0.5; /* chrosen.f90 Delta0 = HALF */
    } else {
        fprintf(stderr, "unknown problem: %s\n", prob);
        free(x0);
        free(xl);
        free(xu);
        return 2;
    }

    g_n = n;
    g_eval = 0;
    npt = 2 * n + 1;
    maxfun = 500 * n;
    ftarget = -INFINITY;
    iprint = PRIMA_MSG_NONE;

    x = (double *)malloc(sizeof(double) * (size_t)n);
    for (i = 0; i < n; i++) {
        x[i] = x0[i];
    }
    f = 0.0;
    nf = 0;

    printf("# config problem=%s n=%d rho_beg=%.17e rho_end=%.17e maxfun=%d npt=%d\n", prob, n,
           rho_beg, rho_end, maxfun, npt);
    printf("# x0");
    for (i = 0; i < n; i++) {
        printf(" %.17e", x0[i]);
    }
    printf("\n");
    printf("# xl");
    for (i = 0; i < n; i++) {
        printf(" %.17e", xl[i]);
    }
    printf("\n");
    printf("# xu");
    for (i = 0; i < n; i++) {
        printf(" %.17e", xu[i]);
    }
    printf("\n");

    rc = prima_bobyqa(&calfun, NULL, n, x, &f, xl, xu, &nf, rho_beg, rho_end, ftarget, maxfun, npt,
                      iprint);

    printf("# final nf=%d rc=%d f=%.17e x=", nf, rc, f);
    for (i = 0; i < n; i++) {
        printf(" %.17e", x[i]);
    }
    printf("\n");

    free(x);
    free(x0);
    free(xl);
    free(xu);
    return 0;
}
