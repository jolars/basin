/* Independent objective callbacks for CG_DESCENT C 1.2. See README.md. */
#include "cg_descent.h"

static int kind;
static int dimension;

static double evaluate(double *x, double *g) {
    if (kind == 0) {
        g[0] = 3.0 * x[0] + x[1];
        g[1] = x[0] + 2.0 * x[1];
        return 0.5 * (x[0] * g[0] + x[1] * g[1]);
    }
    if (kind == 1) {
        double r = x[1] - x[0] * x[0];
        g[0] = -400.0 * x[0] * r + 2.0 * (x[0] - 1.0);
        g[1] = 200.0 * r;
        return 100.0 * r * r + (1.0 - x[0]) * (1.0 - x[0]);
    }
    double mean = 0.0, gradient_mean = 0.0, cost = 0.0;
    for (int i = 0; i < dimension; ++i) mean += x[i];
    mean /= dimension;
    for (int i = 0; i < dimension; ++i) {
        double z = x[i] - 2.0 * mean;
        g[i] = pow(10.0, i) * z;
        cost += 0.5 * z * g[i];
        gradient_mean += g[i];
    }
    gradient_mean /= dimension;
    for (int i = 0; i < dimension; ++i) g[i] -= 2.0 * gradient_mean;
    return cost;
}

static double value(double *x) {
    double g[7];
    return evaluate(x, g);
}

static void gradient(double *g, double *x) {
    evaluate(x, g);
}

int main(void) {
    const char *names[] = {"quadratic", "rosenbrock", "ill_conditioned"};
    puts("# CG_DESCENT C 1.2; gradient infinity tolerance 1e-8; periodic restarts disabled within budget");
    puts("# problem dimension status cost gradient_infinity x...");
    for (kind = 0; kind < 3; ++kind) {
        dimension = kind == 2 ? 7 : 2;
        double x[7], work[28], g[7];
        for (int i = 0; i < dimension; ++i) x[i] = 1.0;
        if (kind == 0) { x[0] = 2.0; x[1] = -1.0; }
        if (kind == 1) x[0] = -1.2;
        cg_stats stats;
        int status = cg_descent(1e-8, x, dimension, value, gradient, work, 0.0, &stats);
        double cost = evaluate(x, g), norm = 0.0;
        for (int i = 0; i < dimension; ++i) norm = fmax(norm, fabs(g[i]));
        printf("%s %d %d %.17e %.17e", names[kind], dimension, status, cost, norm);
        for (int i = 0; i < dimension; ++i) printf(" %.17e", x[i]);
        putchar('\n');
        if (status != 0) return 1;
    }
    return 0;
}
