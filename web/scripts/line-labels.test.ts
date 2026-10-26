import assert from "node:assert/strict";
import test from "node:test";
import {
    COMPETITOR_BENCHMARKS,
    COMPETITOR_CASES,
    LIBRARY_LABELS,
} from "../src/lib/data/competitors";
import {
    type LineLabel,
    labelCorners,
    labelsOverlap,
    type PlotPoint,
    placeLineLabels,
} from "../src/lib/lineLabels";

const bounds = { left: 60, right: 360, top: 16, bottom: 226 };
const line = (label: string, points: PlotPoint[]) => ({
    label,
    points,
    color: "red",
});

function assertLayout(labels: LineLabel[], count: number) {
    assert.equal(labels.length, count);
    for (const [i, label] of labels.entries()) {
        for (const p of labelCorners(label)) {
            assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y));
            assert.ok(p.x >= bounds.left && p.x <= bounds.right, label.label);
            assert.ok(p.y >= bounds.top && p.y <= bounds.bottom, label.label);
        }
        for (const other of labels.slice(i + 1)) {
            assert.ok(
                !labelsOverlap(label, other),
                `${label.label} overlaps ${other.label}`,
            );
        }
    }
}

for (const c of COMPETITOR_CASES) {
    test(`${c.solver}: all labels fit without overlap on measured competitor traces`, () => {
        const rows = COMPETITOR_BENCHMARKS.results.filter(
            (r) => r.solver === c.solver,
        );
        const points = rows.flatMap((r) => r.points);
        const maxTime = Math.max(...points.map((p) => p.tNs));
        const logs = points.map((p) => Math.log10(p.subopt));
        const low = Math.min(...logs);
        const high = Math.max(...logs);
        const lines = rows.map((r) =>
            line(
                LIBRARY_LABELS[r.library],
                r.points.map((p) => ({
                    x:
                        bounds.left +
                        (p.tNs / maxTime) * (bounds.right - bounds.left),
                    y:
                        bounds.bottom -
                        ((Math.log10(p.subopt) - low) / (high - low)) *
                            (bounds.bottom - bounds.top),
                })),
            ),
        );
        const labels = placeLineLabels(lines, bounds);
        assertLayout(labels, rows.length);
        assert.deepEqual(
            new Set(labels.map((l) => l.label)),
            new Set(lines.map((l) => l.label)),
        );
    });
}

test("coincident curves use separate boxes with leaders", () => {
    const points = [
        { x: 60, y: 120 },
        { x: 360, y: 120 },
    ];
    const labels = placeLineLabels(
        [line("basin", points), line("argmin", points)],
        bounds,
    );
    assertLayout(labels, 2);
    assert.ok(labels.every((l) => l.leader !== null));
});

test("shared endpoints do not force separated curves into a shared column", () => {
    const labels = placeLineLabels(
        [
            line("basin", [
                { x: 60, y: 20 },
                { x: 120, y: 60 },
                { x: 360, y: 226 },
            ]),
            line("argmin", [
                { x: 60, y: 20 },
                { x: 300, y: 60 },
                { x: 360, y: 226 },
            ]),
        ],
        bounds,
    );
    assertLayout(labels, 2);
    assert.ok(Math.abs(labels[0].x - labels[1].x) > 40);
    assert.ok(labels.every((l) => Math.abs(l.angle) > 0.1));
});

test("single points, duplicates, and empty series do not produce nonfinite labels", () => {
    const p = { x: bounds.left, y: bounds.bottom };
    assertLayout(
        placeLineLabels([line("basin", [p, p, p]), line("empty", [])], bounds),
        1,
    );
    assertLayout(placeLineLabels([line("point", [p])], bounds), 1);
    assert.deepEqual(placeLineLabels([], bounds), []);
});

test("subdividing a straight trace does not change its label placement", () => {
    const sparse = [
        { x: 60, y: 200 },
        { x: 360, y: 50 },
    ];
    const dense = Array.from({ length: 101 }, (_, i) => ({
        x: 60 + i * 3,
        y: 200 - i * 1.5,
    }));
    const a = placeLineLabels([line("basin", sparse)], bounds)[0];
    const b = placeLineLabels([line("basin", dense)], bounds)[0];
    assert.ok(Math.abs(a.x - b.x) < 1e-8);
    assert.ok(Math.abs(a.y - b.y) < 1e-8);
    assert.ok(Math.abs(a.angle - b.angle) < 1e-8);
});

test("collision checks account for rotated corners", () => {
    const base: LineLabel = {
        label: "test",
        color: "red",
        x: 100,
        y: 100,
        width: 80,
        height: 20,
        angle: Math.PI / 4,
        anchor: { x: 100, y: 100 },
        leader: null,
    };
    assert.ok(labelsOverlap(base, { ...base, x: 130, y: 130 }));
    assert.ok(!labelsOverlap(base, { ...base, x: 130, y: 70 }));
});
