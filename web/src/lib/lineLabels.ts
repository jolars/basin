export type PlotPoint = { x: number; y: number };
type Bounds = { left: number; right: number; top: number; bottom: number };
type Line = { label: string; color: string; points: PlotPoint[] };
export type LineLabel = {
    label: string;
    color: string;
    x: number;
    y: number;
    angle: number;
    width: number;
    height: number;
    anchor: PlotPoint;
    leader: PlotPoint | null;
};
type Candidate = LineLabel & { score: number };

function local(point: PlotPoint, box: LineLabel): PlotPoint {
    const c = Math.cos(box.angle);
    const s = Math.sin(box.angle);
    const dx = point.x - box.x;
    const dy = point.y - box.y;
    return { x: c * dx + s * dy, y: -s * dx + c * dy };
}

function world(point: PlotPoint, box: LineLabel): PlotPoint {
    const c = Math.cos(box.angle);
    const s = Math.sin(box.angle);
    return {
        x: box.x + c * point.x - s * point.y,
        y: box.y + s * point.x + c * point.y,
    };
}

export function labelCorners(box: LineLabel, gap = 0): PlotPoint[] {
    const w = box.width / 2 + gap;
    const h = box.height / 2 + gap;
    return [
        { x: -w, y: -h },
        { x: w, y: -h },
        { x: w, y: h },
        { x: -w, y: h },
    ].map((p) => world(p, box));
}

/** Test the rotated rectangles, including breathing room around each box. */
export function labelsOverlap(a: LineLabel, b: LineLabel): boolean {
    const ac = labelCorners(a, 2);
    const bc = labelCorners(b, 2);
    return [a.angle, b.angle].every((angle) =>
        [angle, angle + Math.PI / 2].every((axis) => {
            const project = (p: PlotPoint) =>
                p.x * Math.cos(axis) + p.y * Math.sin(axis);
            const ap = ac.map(project);
            const bp = bc.map(project);
            return (
                Math.max(...ap) > Math.min(...bp) &&
                Math.max(...bp) > Math.min(...ap)
            );
        }),
    );
}

function pointSegmentDistance(
    p: PlotPoint,
    a: PlotPoint,
    b: PlotPoint,
): number {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const length2 = dx * dx + dy * dy;
    const t = length2
        ? Math.max(
              0,
              Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / length2),
          )
        : 0;
    return Math.hypot(p.x - a.x - t * dx, p.y - a.y - t * dy);
}

function segmentBoxDistance(
    a: PlotPoint,
    b: PlotPoint,
    box: LineLabel,
): number {
    const p = local(a, box);
    const q = local(b, box);
    const w = box.width / 2;
    const h = box.height / 2;
    let enter = 0;
    let exit = 1;
    for (const [start, end, half] of [
        [p.x, q.x, w],
        [p.y, q.y, h],
    ]) {
        const delta = end - start;
        if (delta === 0) {
            if (Math.abs(start) > half) exit = -1;
        } else {
            const t1 = (-half - start) / delta;
            const t2 = (half - start) / delta;
            enter = Math.max(enter, Math.min(t1, t2));
            exit = Math.min(exit, Math.max(t1, t2));
        }
    }
    if (enter <= exit) return 0;
    const pointDistance = (v: PlotPoint) =>
        Math.hypot(
            Math.max(0, Math.abs(v.x) - w),
            Math.max(0, Math.abs(v.y) - h),
        );
    return Math.min(
        pointDistance(p),
        pointDistance(q),
        ...[
            { x: -w, y: -h },
            { x: w, y: -h },
            { x: w, y: h },
            { x: -w, y: h },
        ].map((corner) => pointSegmentDistance(corner, p, q)),
    );
}

function candidates(line: Line, index: number, lines: Line[], bounds: Bounds) {
    const points = line.points;
    const lengths = [0];
    for (let i = 1; i < points.length; i++) {
        lengths.push(
            lengths[i - 1] +
                Math.hypot(
                    points[i].x - points[i - 1].x,
                    points[i].y - points[i - 1].y,
                ),
        );
    }
    const length = lengths[lengths.length - 1];
    const at = (distance: number): PlotPoint => {
        const end = lengths.findIndex((v) => v >= distance);
        if (end <= 0) return points[end === 0 ? 0 : points.length - 1];
        const fraction =
            (distance - lengths[end - 1]) / (lengths[end] - lengths[end - 1]);
        return {
            x:
                points[end - 1].x +
                fraction * (points[end].x - points[end - 1].x),
            y:
                points[end - 1].y +
                fraction * (points[end].y - points[end - 1].y),
        };
    };
    const width = line.label.length * 6.1 + 10;
    const height = 18;
    const result: Candidate[] = [];
    // Sampling in screen distance makes placement independent of trace density.
    for (let i = 0; i <= 48; i++) {
        const distance = (i / 48) * length;
        const anchor = at(distance);
        const before = at(Math.max(0, distance - width / 2));
        const after = at(Math.min(length, distance + width / 2));
        const angle = Math.max(
            -Math.PI / 3,
            Math.min(
                Math.PI / 3,
                Math.atan2(after.y - before.y, after.x - before.x),
            ),
        );
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        const extentX = (Math.abs(c) * width + Math.abs(s) * height) / 2;
        const extentY = (Math.abs(s) * width + Math.abs(c) * height) / 2;
        if (
            2 * extentX + 8 > bounds.right - bounds.left ||
            2 * extentY + 8 > bounds.bottom - bounds.top
        )
            continue;
        for (const offset of [0, -16, 16, -32, 32]) {
            const box: Candidate = {
                label: line.label,
                color: line.color,
                width,
                height,
                angle,
                anchor,
                x: Math.max(
                    bounds.left + extentX + 4,
                    Math.min(bounds.right - extentX - 4, anchor.x - s * offset),
                ),
                y: Math.max(
                    bounds.top + extentY + 4,
                    Math.min(
                        bounds.bottom - extentY - 4,
                        anchor.y + c * offset,
                    ),
                ),
                leader: null,
                score: 0,
            };
            let clearance = Math.min(
                box.x - extentX - bounds.left,
                bounds.right - box.x - extentX,
                box.y - extentY - bounds.top,
                bounds.bottom - box.y - extentY,
            );
            for (let j = 0; j < lines.length; j++) {
                if (j === index) continue;
                const other = lines[j].points;
                for (let k = 0; k < other.length; k++) {
                    const a = other[k];
                    const b = other[Math.max(0, k - 1)];
                    // Distant segments cannot reduce the current clearance.
                    if (
                        Math.max(a.x, b.x) < box.x - extentX - clearance ||
                        Math.min(a.x, b.x) > box.x + extentX + clearance ||
                        Math.max(a.y, b.y) < box.y - extentY - clearance ||
                        Math.min(a.y, b.y) > box.y + extentY + clearance
                    )
                        continue;
                    clearance = Math.min(
                        clearance,
                        segmentBoxDistance(a, b, box),
                    );
                    if (clearance === 0) break;
                }
                if (clearance === 0) break;
            }
            const displacement = Math.hypot(box.x - anchor.x, box.y - anchor.y);
            box.score =
                clearance - displacement * 0.6 - (clearance === 0 ? 100 : 0);
            const relative = local(anchor, box);
            const edge = {
                x: Math.max(-width / 2, Math.min(width / 2, relative.x)),
                y: Math.max(-height / 2, Math.min(height / 2, relative.y)),
            };
            if (Math.hypot(relative.x - edge.x, relative.y - edge.y) > 2) {
                box.leader = world(edge, box);
            }
            result.push(box);
        }
    }
    return result.sort((a, b) => b.score - a.score);
}

/**
 * Inspired by directlabels' angled.boxes: search along the curves for space
 * away from other curves and borders, and rotate boxes with the local slope.
 * https://github.com/tdhock/directlabels/blob/master/R/utility.function.R
 */
export function placeLineLabels(lines: Line[], bounds: Bounds): LineLabel[] {
    const choices = lines
        .map((line, i) =>
            line.points.length ? candidates(line, i, lines, bounds) : [],
        )
        .filter((list) => list.length)
        .sort((a, b) => a[0].score - b[0].score);
    // Keep several arrangements so an early choice cannot block a later label.
    let arrangements: { labels: Candidate[]; score: number }[] = [
        { labels: [], score: 0 },
    ];
    for (const list of choices) {
        let next = arrangements.flatMap((arrangement) =>
            list
                .filter((candidate) =>
                    arrangement.labels.every(
                        (label) => !labelsOverlap(candidate, label),
                    ),
                )
                .map((candidate) => ({
                    labels: [...arrangement.labels, candidate],
                    score: arrangement.score + candidate.score,
                })),
        );
        // If the chart is too small, retain every label as a best effort.
        if (!next.length) {
            next = arrangements.flatMap((arrangement) =>
                list.map((candidate) => ({
                    labels: [...arrangement.labels, candidate],
                    score:
                        arrangement.score +
                        candidate.score -
                        1000 *
                            arrangement.labels.filter((label) =>
                                labelsOverlap(candidate, label),
                            ).length,
                })),
            );
        }
        arrangements = next.sort((a, b) => b.score - a.score).slice(0, 32);
    }
    return arrangements[0].labels;
}
