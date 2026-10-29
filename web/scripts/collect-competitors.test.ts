import assert from "node:assert/strict";
import test from "node:test";
import {
    COMPETITOR_CASES,
    type CompetitorResult,
    validateCompetitorResults,
} from "../src/lib/data/competitors";

function fixture(): CompetitorResult[] {
    return COMPETITOR_CASES.flatMap((c) =>
        c.libraries.flatMap((library) =>
            (c.seeds ?? [undefined]).map((seed) => ({
                solver: c.solver,
                problem: c.problem,
                library,
                n: c.n,
                ...(seed === undefined ? {} : { seed }),
                points: [{ tNs: 10, subopt: 1 }],
                ...(c.solver === "cobyla" || c.solver === "slsqp"
                    ? { violations: [{ tNs: 10, violation: 1e-16 }] }
                    : {}),
                ...(c.solver === "cobyla" ||
                c.solver === "slsqp" ||
                c.solver === "lbfgsb" ||
                c.solver === "lm" ||
                c.solver === "cmaes"
                    ? {
                          summary: {
                              point: Array(c.n).fill(0.75),
                              cost: 0.125,
                              violation: 0,
                              objectiveCalls: 5,
                              residualCalls: 5,
                              jacobianCalls: 3,
                              gradientCalls: 3,
                              constraintCalls: 5,
                              status: "Converged",
                              verified: true,
                              ...(c.solver === "cmaes"
                                  ? {
                                        point: Array(c.n).fill(0),
                                        cost: 5e-9,
                                        iterations: 2,
                                        objectiveCalls:
                                            2 * (library === "basin" ? 17 : 16),
                                        residualCalls: 0,
                                        jacobianCalls: 0,
                                        gradientCalls: 0,
                                        constraintCalls: 0,
                                    }
                                  : {}),
                              ...(c.solver === "lbfgsb"
                                  ? {
                                        iterations: 40,
                                        projectedGradient: 1e-9,
                                        gradientCalls: 5,
                                    }
                                  : {}),
                          },
                      }
                    : {}),
            })),
        ),
    );
}

test("collects exactly one measured row per curated case, library, and seed", () => {
    assert.equal(validateCompetitorResults(fixture()).length, 41);
    assert.throws(
        () => validateCompetitorResults(fixture().slice(1)),
        /Missing result/,
    );
    const rows = fixture();
    assert.throws(
        () => validateCompetitorResults([...rows, rows[0]]),
        /duplicate/,
    );
});

test("rejects measurements from stale problem dimensions", () => {
    const rows = fixture();
    rows[0].n = 2;
    assert.throws(() => validateCompetitorResults(rows), /Invalid dimension/);
});

test("rejects malformed, nonfinite, or reversed trace points", () => {
    for (const points of [
        [{ tNs: 10, subopt: Number.NaN }],
        [{ tNs: 10, subopt: 0 }],
        [
            { tNs: 20, subopt: 1 },
            { tNs: 10, subopt: 1 },
        ],
    ]) {
        const rows = fixture();
        rows[0].points = points;
        assert.throws(
            () => validateCompetitorResults(rows),
            /Invalid trace point/,
        );
    }
});

test("requires constrained measurements and meaningful run diagnostics", () => {
    const rows = fixture();
    const quadratic = rows.find((r) => r.solver === "cobyla");
    assert.ok(quadratic?.summary);
    delete quadratic.violations;
    assert.throws(
        () => validateCompetitorResults(rows),
        /Missing constrained measurements/,
    );
    quadratic.violations = [{ tNs: 10, violation: 1e-16 }];
    quadratic.summary.objectiveCalls = -1;
    assert.throws(
        () => validateCompetitorResults(rows),
        /Invalid callback count/,
    );
});

test("preserves zero violation and rejects negative violation", () => {
    const rows = fixture();
    const quadratic = rows.find((r) => r.solver === "cobyla");
    assert.ok(quadratic?.violations);
    quadratic.violations[0].violation = 0;
    assert.equal(
        validateCompetitorResults(rows).find((r) => r.solver === "cobyla")
            ?.violations?.[0].violation,
        0,
    );
    quadratic.violations[0].violation = -1e-8;
    assert.throws(() => validateCompetitorResults(rows), /Invalid trace point/);
});

test("retains failed runs without treating them as verified solutions", () => {
    const rows = fixture();
    assert.ok(rows[0].summary);
    rows[0].summary.verified = false;
    rows[0].summary.status = "SolverFailed";
    rows[0].summary.cost = null;
    rows[0].summary.point = Array(rows[0].n).fill(null);
    rows[0].summary.residualCalls = 0;
    rows[0].summary.jacobianCalls = 0;
    rows[0].points = [];
    assert.equal(
        validateCompetitorResults(rows)[0].summary?.status,
        "SolverFailed",
    );
    rows[0].summary.verified = true;
    assert.throws(
        () => validateCompetitorResults(rows),
        /Invalid verified solution/,
    );
});

test("requires L-BFGS-B stationarity and fused callback diagnostics", () => {
    const rows = fixture();
    const bounded = rows.find((r) => r.solver === "lbfgsb");
    assert.ok(bounded?.summary);
    delete bounded.summary.projectedGradient;
    assert.throws(
        () => validateCompetitorResults(rows),
        /Invalid L-BFGS-B diagnostics/,
    );
    bounded.summary.projectedGradient = 1e-3;
    assert.throws(
        () => validateCompetitorResults(rows),
        /Invalid verified L-BFGS-B/,
    );
    bounded.summary.verified = false;
    bounded.summary.projectedGradient = null;
    assert.equal(validateCompetitorResults(rows).length, 41);
    bounded.summary.projectedGradient = -1;
    assert.throws(
        () => validateCompetitorResults(rows),
        /Invalid L-BFGS-B diagnostics/,
    );
    bounded.summary.projectedGradient = 1e-9;
    bounded.summary.verified = true;
    bounded.summary.gradientCalls = 0;
    assert.throws(
        () => validateCompetitorResults(rows),
        /Invalid verified L-BFGS-B/,
    );
});

test("requires every stochastic seed and retains failed seeds", () => {
    const rows = fixture();
    const stochastic = rows.find((r) => r.solver === "cmaes");
    assert.ok(stochastic?.summary);
    assert.throws(
        () => validateCompetitorResults(rows.filter((r) => r !== stochastic)),
        /Missing result: cmaes\/rotated_ellipsoid\/basin\/seed0/,
    );
    assert.throws(
        () => validateCompetitorResults([...rows, stochastic]),
        /duplicate/,
    );
    delete stochastic.seed;
    assert.throws(() => validateCompetitorResults(rows), /Unexpected seed/);
    stochastic.seed = 5;
    assert.throws(() => validateCompetitorResults(rows), /Unexpected seed/);
    stochastic.seed = 0;
    stochastic.summary.cost = 1;
    assert.throws(
        () => validateCompetitorResults(rows),
        /Invalid verified CMA-ES/,
    );
    stochastic.summary.verified = false;
    stochastic.summary.status = "EvaluationBudget";
    assert.equal(validateCompetitorResults(rows).length, 41);
    stochastic.summary.objectiveCalls = 100_017;
    assert.throws(
        () => validateCompetitorResults(rows),
        /Invalid CMA-ES diagnostics/,
    );
    stochastic.summary.objectiveCalls = 1;
    stochastic.summary.iterations = 0;
    stochastic.summary.cost = null;
    stochastic.summary.point = Array(stochastic.n).fill(null);
    stochastic.summary.status = "InvalidFunctionValue";
    stochastic.points = [];
    assert.equal(validateCompetitorResults(rows).length, 41);
});
