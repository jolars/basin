//! Generative favicon for `basin`.
//!
//! A pared-down sibling of the full [`logo`](./logo.rs): the same low-poly
//! *isometric basin*, but stripped to read at 16–32 px in a browser tab. Where
//! the logo carries a curved valley, ripples, scattered trees and rocks, a
//! creature and a sun/moon, the favicon keeps only the essentials:
//!
//! - a coarse-faceted bowl (a quadratic basin with damped ripple ridges on its
//!   walls, the floor left clean): the canonical optimization landscape,
//! - a low-poly pool at its floor, carved into the *mesh* itself (the bowl
//!   triangles clipped at the water line), not a shape laid on top,
//! - a single river: a thin water ribbon laid along an actual [`GradientDescent`]
//!   trajectory, real mesh that rides on the surface and is lit per facet by the
//!   slope, so it reads as flowing water on the terrain, not a flat sticker, and
//! - one small conifer on the bank for flavor.
//!
//! Like the logo there are two themes; the night palette is derived from the
//! day one by the same Oklab [`moonlight`] transform. Output is a square,
//! transparent SVG so it reads on any tab chrome.
//!
//! Run with:
//!
//! ```text
//! cargo run --example favicon                  # day      -> basin-favicon.svg
//! cargo run --example favicon out.svg          # day      -> out.svg
//! cargo run --example favicon out.svg dark     # night    -> out.svg
//! cargo run --example favicon out.svg adaptive # self-adapting (day + night)
//! ```
//!
//! The `adaptive` mode ships *both* palettes in one file, switched by
//! `prefers-color-scheme`, so a single `<link rel="icon">` covers light and
//! dark tab chrome; this is the form linked as the site favicon (one SVG, as
//! RealFaviconGenerator expects for a dual light/dark icon). The standalone
//! `day` and `night` renders remain the raster source for the `.ico`/PNG
//! fallbacks, where CSS can't adapt.

use std::fmt::Write as _;

use basin::{BasicState, CostFunction, Gradient, GradientDescent, Problem, Solver, State};

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

/// Grid cells per side (vertices = `GRID + 1`). Coarse so the facets read as
/// chunky low-poly. The river edge no longer depends on this (it comes from the
/// ribbon's own `RIVER_SEGS`), so the terrain can stay coarse.
const GRID: usize = 16;

/// Problem-space framing window `[X0, X1] × [Y0, Y1]`.
const X0: f64 = -3.0;
const X1: f64 = 3.0;
const Y0: f64 = -3.0;
const Y1: f64 = 3.0;

/// Surface shape. A quadratic bowl `K·(u² + ASPECT·v²)` in coordinates rotated by
/// `BOWL_ROT` about the center, so the basin can be round (`ASPECT = 1`) or an
/// oval tilted to taste. On top rides a damped sinusoidal **ripple** (organic
/// low-poly ridges on the walls) that fades to zero on the floor, so the pool
/// and the river's descent stay clean (see [`ripple`]). Tune:
///   - `BOWL_K` overall steepness; `BOWL_ASPECT` ovalness; `BOWL_ROT` tilt;
///   - `RIPPLE_AMP` ridge height (0 = smooth dome), `RIPPLE_FREQ` ridge frequency
///     (low = few big coarse ridges), `RIPPLE_FADE` the bowl height at which the
///     ripple reaches full strength (smaller = ridges start closer to the water).
///
/// Keep `RIPPLE_AMP` modest relative to `BOWL_K`, or the descent can puddle in a
/// ridge instead of reaching the pool.
const BOWL_CX: f64 = 0.29;
const BOWL_CY: f64 = -0.15;
const BOWL_K: f64 = 0.2;
const BOWL_ASPECT: f64 = 0.65;
const BOWL_ROT: f64 = 0.35;
const RIPPLE_AMP: f64 = 0.22;
const RIPPLE_FREQ: f64 = 1.4;
const RIPPLE_FADE: f64 = 0.7;

/// Water line as a normalized height above the basin floor. The lake is every
/// point below it; because the inlet valley floor also dips below it, the lake
/// reaches up the valley as the river arm. Higher = bigger lake + longer arm.
const WATER_LEVEL: f64 = 0.02;

/// Isometric tile half-extents (screen px) and vertical gain. A gentler tile
/// ratio (closer to 3:2 than 2:1) tilts the view a little more overhead, so the
/// diamond is less flat and fills a square frame better.
const TILE_W: f64 = 22.0;
const TILE_H: f64 = 16.0;
const Z_SCREEN: f64 = 96.0; // screen px from basin floor to highest rim
const Z_WORLD: f64 = 16.0; // height gain for facet-normal lighting (grid units)

/// Light direction in world space (`+z` up), pointing *from* the light.
const LIGHT: [f64; 3] = [0.5, -0.35, 0.79];

/// River = gradient descent on the bowl, started **on the window boundary** so
/// the stream enters from the rim (its head is cut by the tile edge, water
/// flowing in from beyond the basin) rather than materialising mid-slope. Small
/// step, no momentum: a clean run down into the pool.
const RIVER_START: [f64; 2] = [-3.4, -2.5];
const RIVER_ALPHA: f64 = 0.06;
const RIVER_BETA: f64 = 0.0;
const RIVER_ITERS: usize = 400;
const RIVER_POINTS: usize = 120; // trajectory vertices captured (descent demo)

/// River carved into the terrain mesh along the descent trajectory: a ribbon
/// tessellated to `RIVER_SEGS` segments (so its edge is smooth, independent of
/// `GRID`), tapering from full width `RIVER_W_SRC` at the source to
/// `RIVER_W_MOUTH` at the mouth (problem units), intersected with the terrain
/// triangles ([`river_shapes`] + [`draw_terrain`]) so the river is real
/// sub-faces of the surface (true heights, slope-lit), not an overlay.
/// `SPRING_R` optionally unions a round source pool at the head (0 = none).
const RIVER_W_SRC: f64 = 0.08;
const RIVER_W_MOUTH: f64 = 0.6;
const RIVER_SEGS: usize = 16;
const SPRING_R: f64 = 0.0;

/// One conifer on the bank: a problem-space point above the water line, on the
/// opposite side from the river so it stays clear of the stream.
const TREE_AT: [f64; 2] = [2.0, -0.9];

/// Square-frame padding (screen px).
const PAD: f64 = 12.0;

/// Teal/slate palette (shared with the logo).
#[allow(dead_code)]
const PAPER: Rgb = Rgb(244, 241, 222); // #f4f1de (unused here; kept for parity)
/// Terrain hypsometric ramp (low elevation → high), the logo's active ramp.
const CUSTOM: [Rgb; 5] = [
    hex("#5E7763"),
    hex("#7C7A62"),
    hex("#9C836F"),
    hex("#B59A85"),
    hex("#DDD5CB"),
];
const TERRAIN_RAMP: &[Rgb] = &CUSTOM;
const WATER: Rgb = Rgb(71, 153, 173); // pool + river (one body)
const TREE_DARK: Rgb = Rgb(54, 90, 82); // canopy shadow side
const TREE_LIT: Rgb = Rgb(92, 138, 122); // canopy lit side
const TRUNK: Rgb = Rgb(52, 46, 39);

// ---------------------------------------------------------------------------
// theme (day or night)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Theme {
    Day,
    Night,
}

/// Pale, cool full-moon tint, used only to lift the night water toward a
/// moonlit sheen so the pool/river don't sink into the dark terrain.
const MOON: Rgb = hex("#d8dcec");

/// Re-tint a daytime color for the night palette: darken, desaturate, and cool
/// toward moonlit blue, all in Oklab so the shift stays perceptually even. The
/// same transform the logo uses, so the two stay in sync.
fn moonlight(c: Rgb) -> Rgb {
    let [l, a, b] = rgb_to_oklab(c);
    oklab_to_rgb([l * 0.75 + 0.035, a * 0.76, b * 0.46 - 0.018])
}

/// All theme-dependent colors, resolved once up front.
struct Palette {
    terrain: Vec<Rgb>,
    water: Rgb,
    tree_dark: Rgb,
    tree_lit: Rgb,
    trunk: Rgb,
}

impl Palette {
    fn new(theme: Theme) -> Self {
        match theme {
            Theme::Day => Self {
                terrain: TERRAIN_RAMP.to_vec(),
                water: WATER,
                tree_dark: TREE_DARK,
                tree_lit: TREE_LIT,
                trunk: TRUNK,
            },
            Theme::Night => Self {
                terrain: TERRAIN_RAMP.iter().map(|&c| moonlight(c)).collect(),
                water: moonlight(WATER).lerp(MOON, 0.14),
                tree_dark: moonlight(TREE_DARK),
                tree_lit: moonlight(TREE_LIT),
                trunk: moonlight(TRUNK),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// surface
// ---------------------------------------------------------------------------

/// Raw surface height at `(x, y)`: a (rotated, possibly oval) quadratic bowl with
/// a damped sinusoidal ripple on its walls. This is the surface the solver
/// descends to trace the river, and the surface the water-plane clip fills to
/// make the lake.
fn raw_height(x: f64, y: f64) -> f64 {
    let dx = x - BOWL_CX;
    let dy = y - BOWL_CY;
    let (ca, sa) = (BOWL_ROT.cos(), BOWL_ROT.sin());
    let u = ca * dx + sa * dy;
    let v = -sa * dx + ca * dy;
    let bowl = BOWL_K * (u * u + BOWL_ASPECT * v * v);
    bowl + ripple(u, v, bowl)
}

/// Damped low-poly ripple in the rotated bowl coordinates: a sum of a few
/// sinusoids (so the ridges read as irregular, not a regular corrugation),
/// scaled by `damp = clamp(bowl / RIPPLE_FADE, 0, 1)` so it vanishes on the floor
/// and grows up the walls, keeping the pool and the descent clean.
fn ripple(u: f64, v: f64, bowl: f64) -> f64 {
    let damp = (bowl / RIPPLE_FADE).min(1.0);
    damp * RIPPLE_AMP
        * ((RIPPLE_FREQ * u + 0.4).sin()
            + 0.7 * (1.7 * RIPPLE_FREQ * v - 0.7).sin()
            + 0.5 * (0.9 * RIPPLE_FREQ * (u + v)).sin())
}

/// Normalization + water level for the surface, sampled once.
struct Surface {
    hmin: f64,
    hmax: f64,
    water: f64,
}

impl Surface {
    fn new() -> Self {
        let n = 96;
        let mut hmin = f64::INFINITY;
        let mut hmax = f64::NEG_INFINITY;
        for j in 0..=n {
            for i in 0..=n {
                let x = X0 + (X1 - X0) * i as f64 / n as f64;
                let y = Y0 + (Y1 - Y0) * j as f64 / n as f64;
                let h = raw_height(x, y);
                hmin = hmin.min(h);
                hmax = hmax.max(h);
            }
        }
        Self {
            hmin,
            hmax,
            water: WATER_LEVEL,
        }
    }

    /// Normalized height in `[0, 1]`.
    fn hn(&self, x: f64, y: f64) -> f64 {
        ((raw_height(x, y) - self.hmin) / (self.hmax - self.hmin)).clamp(0.0, 1.0)
    }

    /// On-surface screen position of a problem point, lifted by `lift`
    /// (normalized units) so the river/tree ride just above their facets.
    fn surface_point(&self, x: f64, y: f64, lift: f64) -> (f64, f64) {
        let gx = (x - X0) / (X1 - X0) * GRID as f64;
        let gy = (y - Y0) / (Y1 - Y0) * GRID as f64;
        project(gx, gy, self.hn(x, y).max(self.water) + lift)
    }
}

/// Isometric projection: grid coords `(gx, gy)` on the ground plane, plus
/// normalized height `h`.
fn project(gx: f64, gy: f64, h: f64) -> (f64, f64) {
    let sx = (gx - gy) * TILE_W;
    let sy = (gx + gy) * TILE_H - h * Z_SCREEN;
    (sx, sy)
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    [v[0] / n, v[1] / n, v[2] / n]
}

/// The bowl as a `CostFunction` + `Gradient` so basin's [`GradientDescent`] can
/// descend the exact terrain we draw (analytic gradient of [`raw_height`]).
struct Bowl;

impl CostFunction for Bowl {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok(raw_height(x[0], x[1]))
    }
}

impl Gradient for Bowl {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        // Central difference of `raw_height`, so the descent follows the actual
        // rippled surface (no need to hand-differentiate the ripple).
        let h = 1e-4;
        let mut g = vec![0.0; 2];
        for (k, gk) in g.iter_mut().enumerate() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[k] += h;
            xm[k] -= h;
            *gk = (raw_height(xp[0], xp[1]) - raw_height(xm[0], xm[1])) / (2.0 * h);
        }
        Ok(g)
    }
}

/// Trace the river: gradient descent on the bowl, driven through basin's
/// `Solver` loop, capturing every iterate then decimating.
fn trace_river() -> Vec<[f64; 2]> {
    let mut solver = GradientDescent::new(RIVER_ALPHA).with_momentum(RIVER_BETA);
    let mut problem = Problem::new(Bowl);
    let mut state = solver
        .init(&mut problem, BasicState::new(RIVER_START.to_vec()))
        .unwrap();
    let mut full = vec![[state.param()[0], state.param()[1]]];
    for _ in 0..RIVER_ITERS {
        let (next, stop) = solver.next_iter(&mut problem, state).unwrap();
        state = next;
        full.push([state.param()[0], state.param()[1]]);
        if stop.is_some() {
            break;
        }
    }
    let stride = (full.len() / RIVER_POINTS).max(1);
    let mut pts: Vec<[f64; 2]> = full.iter().step_by(stride).copied().collect();
    if let Some(&last) = full.last() {
        if pts.last() != Some(&last) {
            pts.push(last);
        }
    }
    pts
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let mut args = std::env::args().skip(1);
    let out_path = args.next().unwrap_or_else(|| "basin-favicon.svg".into());
    // Second arg selects the palette. `adaptive`/`auto` ships *both* day and
    // night in one file, switched by `prefers-color-scheme` (the form linked as
    // the site favicon, a single self-adapting SVG); `dark`/`night` a lone
    // night render; anything else day. The single-theme renders are the raster
    // source for the `.ico`/PNG fallbacks, which can't adapt.
    let mode = args.next().unwrap_or_default();

    let surf = Surface::new();
    let river = trace_river();
    let pool_center = [BOWL_CX, BOWL_CY];

    let (doc, theme_name) = match mode.as_str() {
        "adaptive" | "auto" => {
            let day = render_scene(&surf, &river, &Palette::new(Theme::Day));
            let night = render_scene(&surf, &river, &Palette::new(Theme::Night));
            (compose_adaptive(day, night), "adaptive")
        }
        "dark" | "night" => (
            render_scene(&surf, &river, &Palette::new(Theme::Night)).finish(),
            "night",
        ),
        _ => (
            render_scene(&surf, &river, &Palette::new(Theme::Day)).finish(),
            "day",
        ),
    };
    std::fs::write(&out_path, &doc).expect("write SVG");

    let end = river.last().copied().unwrap_or(pool_center);
    eprintln!(
        "wrote {out_path} ({} bytes, {theme_name}); river: {} steps from {:?} to ({:.3}, {:.3}); pool at ({:.3}, {:.3})",
        doc.len(),
        river.len(),
        RIVER_START,
        end[0],
        end[1],
        pool_center[0],
        pool_center[1],
    );
}

/// Render the full favicon scene (terrain + bank tree) into a fresh [`Svg`] with
/// `pal`. Geometry is palette-independent, so the day and night renders share
/// identical vertices, only the fills differ, which is what lets
/// [`compose_adaptive`] stack them in one self-adapting file.
fn render_scene(surf: &Surface, river: &[[f64; 2]], pal: &Palette) -> Svg {
    let mut svg = Svg::new();
    draw_terrain(&mut svg, surf, pal, river);
    draw_tree(
        &mut svg,
        pal,
        surf.surface_point(TREE_AT[0], TREE_AT[1], 0.0),
    );
    svg
}

/// Fuse a day and a night render (identical geometry, different fills) into one
/// self-adapting favicon: both palettes ship in a single file, wrapped in
/// `#light-icon`/`#dark-icon` groups and toggled by `prefers-color-scheme`, so
/// one `<link rel="icon">` covers light and dark tab chrome. `#light-icon` is
/// the default, so any renderer that ignores the media query (e.g. resvg
/// rasterising the `.ico`) gets the day art. This mirrors the markup
/// RealFaviconGenerator emits for a dual light/dark SVG.
fn compose_adaptive(day: Svg, night: Svg) -> String {
    let (vx, vy, side) = day.viewbox(); // identical geometry → either frames both
    let mut out = String::new();
    let _ = write!(
        out,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{:.2} {:.2} {:.2} {:.2}" width="{:.0}" height="{:.0}">"#,
        vx, vy, side, side, side, side
    );
    out.push('\n');
    out.push_str(
        "<style>#dark-icon{display:none}\
         @media (prefers-color-scheme:dark){#light-icon{display:none}#dark-icon{display:inline}}</style>\n",
    );
    out.push_str("<g id=\"light-icon\">\n");
    out.push_str(&day.body);
    out.push_str("</g>\n<g id=\"dark-icon\">\n");
    out.push_str(&night.body);
    out.push_str("</g>\n</svg>\n");
    out
}

// ---------------------------------------------------------------------------
// drawing
// ---------------------------------------------------------------------------

/// One isometric triangle facet, flat-shaded; drawn back-to-front by the caller.
struct Facet {
    pts: Vec<(f64, f64)>,
    depth: f64,
    color: Rgb,
}

/// The faceted bowl, with the lake clipped out of the mesh and the river laid
/// *into* it at high resolution. Per terrain triangle:
///   - clip at the water plane `z = wl`: the below-water piece is the **lake**,
///     drawn flat at the water line (one flat shade); the above-water piece is
///     **terrain**, elevation-tinted and slope-shaded;
///   - intersect the triangle's ground footprint with the **river ribbon**, a
///     smooth, finely-tessellated strip of convex polygons along the trajectory
///     ([`river_shapes`]), and draw each intersection piece at the terrain
///     height (barycentric) shaded by the triangle's own slope normal. So the
///     river is real mesh that follows the ground's curvature and is lit like it,
///     but its *edge* is as smooth as the ribbon, independent of `GRID`.
///
/// Three back-to-front passes (terrain, river, lake): the river is drawn over the
/// full terrain (same plane, so it just colors its footprint), and the lake last
/// covers any river that dips below the water line. Each facet is stroked in its
/// own fill to hide the anti-aliasing seams between same-color neighbors.
fn draw_terrain(svg: &mut Svg, surf: &Surface, pal: &Palette, river: &[[f64; 2]]) {
    let wl = surf.water;
    let ht: Vec<Vec<f64>> = (0..=GRID)
        .map(|j| {
            (0..=GRID)
                .map(|i| {
                    let (x, y) = node_xy(i, j);
                    surf.hn(x, y)
                })
                .collect()
        })
        .collect();

    let light = normalize3(LIGHT);
    // The lake is flat, so it gets a single shade (lit as a horizontal facet).
    let water_flat = pal.water.shade(0.70 + 0.5 * light[2].max(0.0));
    // The river footprint as convex polygons in grid coords (+ bbox for culling),
    // tessellated finely so its edge is smooth regardless of `GRID`.
    let shapes = river_shapes(river);

    let mut terr: Vec<Facet> = Vec::with_capacity(GRID * GRID * 2);
    let mut rivr: Vec<Facet> = Vec::new();
    let mut lake: Vec<Facet> = Vec::new();
    for j in 0..GRID {
        for i in 0..GRID {
            for tri in [
                [(i, j), (i + 1, j), (i, j + 1)],
                [(i + 1, j), (i + 1, j + 1), (i, j + 1)],
            ] {
                // Triangle vertices in grid coords carrying their true height.
                let verts: [[f64; 3]; 3] = [
                    [tri[0].0 as f64, tri[0].1 as f64, ht[tri[0].1][tri[0].0]],
                    [tri[1].0 as f64, tri[1].1 as f64, ht[tri[1].1][tri[1].0]],
                    [tri[2].0 as f64, tri[2].1 as f64, ht[tri[2].1][tri[2].0]],
                ];
                let depth = verts.iter().map(|p| p[0] + p[1]).sum::<f64>() / 3.0;

                let scaled = |p: [f64; 3]| [p[0], p[1], p[2] * Z_WORLD];
                let mut n = normalize3(cross(
                    sub(scaled(verts[1]), scaled(verts[0])),
                    sub(scaled(verts[2]), scaled(verts[0])),
                ));
                if n[2] < 0.0 {
                    n = [-n[0], -n[1], -n[2]];
                }
                let ndotl = (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]).max(0.0);

                // Terrain (above water) and lake (below water) via the height clip.
                let land = clip_h(&verts, wl, true);
                if land.len() >= 3 {
                    let elev = land.iter().map(|p| p[2]).sum::<f64>() / land.len() as f64;
                    let pts = land.iter().map(|p| project(p[0], p[1], p[2])).collect();
                    terr.push(Facet {
                        pts,
                        depth,
                        color: ramp_sample(&pal.terrain, elev).shade(0.70 + 0.5 * ndotl),
                    });
                }
                let below = clip_h(&verts, wl, false);
                if below.len() >= 3 {
                    let pts = below.iter().map(|p| project(p[0], p[1], wl)).collect();
                    lake.push(Facet {
                        pts,
                        depth,
                        color: water_flat,
                    });
                }

                // River: intersect this triangle's ground footprint with the
                // ribbon polygons; each piece rides on the terrain plane (heights
                // barycentric) and is shaded by the same slope normal.
                let tri2 = [
                    [verts[0][0], verts[0][1]],
                    [verts[1][0], verts[1][1]],
                    [verts[2][0], verts[2][1]],
                ];
                let (tx0, ty0, tx1, ty1) = tri_bbox(&tri2);
                let river_color = pal.water.shade(0.70 + 0.5 * ndotl);
                for (poly, bb) in &shapes {
                    if bb[0] > tx1 || bb[2] < tx0 || bb[1] > ty1 || bb[3] < ty0 {
                        continue;
                    }
                    let inter = clip_convex(&tri2, poly);
                    if inter.len() >= 3 {
                        let pts = inter
                            .iter()
                            .map(|p| project(p[0], p[1], bary_height(*p, &verts)))
                            .collect();
                        rivr.push(Facet {
                            pts,
                            depth,
                            color: river_color,
                        });
                    }
                }
            }
        }
    }

    terr.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());
    rivr.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());
    lake.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());
    for f in terr.iter().chain(rivr.iter()).chain(lake.iter()) {
        svg.polygon(&f.pts, f.color, Some((f.color, 1.0)));
    }
}

/// Clip a triangle `[gx, gy, height]` at the water plane, keeping the `≥ wl` half
/// (`keep_land`) or the `≤ wl` half. Sutherland–Hodgman against one plane; the
/// crossing points land on `z = wl`, so land and lake share the shoreline edge.
fn clip_h(verts: &[[f64; 3]; 3], wl: f64, keep_land: bool) -> Vec<[f64; 3]> {
    let inside = |z: f64| if keep_land { z >= wl } else { z <= wl };
    let mut out: Vec<[f64; 3]> = Vec::with_capacity(4);
    for k in 0..3 {
        let cur = verts[k];
        let nxt = verts[(k + 1) % 3];
        if inside(cur[2]) {
            out.push(cur);
        }
        if inside(cur[2]) != inside(nxt[2]) {
            let t = (wl - cur[2]) / (nxt[2] - cur[2]);
            out.push([
                cur[0] + (nxt[0] - cur[0]) * t,
                cur[1] + (nxt[1] - cur[1]) * t,
                wl,
            ]);
        }
    }
    out
}

/// Problem-space coordinates of grid node `(i, j)`.
fn node_xy(i: usize, j: usize) -> (f64, f64) {
    (
        X0 + (X1 - X0) * i as f64 / GRID as f64,
        Y0 + (Y1 - Y0) * j as f64 / GRID as f64,
    )
}

/// The river footprint as a strip of convex polygons in **grid coordinates**,
/// each paired with its bounding box `[minx, miny, maxx, maxy]` for cheap
/// culling. The strip follows the descent trajectory (tessellated to `RIVER_SEGS`
/// segments, its edges are smooth regardless of the terrain grid), tapering from
/// `RIVER_W_SRC` at the source to `RIVER_W_MOUTH` toward the mouth (problem
/// units). If `SPRING_R > 0`, a round source pool is appended. [`draw_terrain`]
/// intersects each terrain triangle with these to carve the river into the mesh.
fn river_shapes(river: &[[f64; 2]]) -> Vec<(Vec<[f64; 2]>, [f64; 4])> {
    let n0 = river.len();
    if n0 < 2 {
        return Vec::new();
    }
    let stride = (n0 / RIVER_SEGS).max(1);
    let mut path: Vec<[f64; 2]> = river.iter().step_by(stride).copied().collect();
    if path.last() != river.last() {
        path.push(*river.last().unwrap());
    }
    let n = path.len();
    if n < 2 {
        return Vec::new();
    }
    let to_grid = |p: [f64; 2]| {
        [
            (p[0] - X0) / (X1 - X0) * GRID as f64,
            (p[1] - Y0) / (Y1 - Y0) * GRID as f64,
        ]
    };
    let half = |i: usize| -> f64 {
        let t = (i as f64 / (n - 1) as f64).powf(0.75);
        0.5 * (RIVER_W_SRC + (RIVER_W_MOUTH - RIVER_W_SRC) * t)
    };
    let (mut left, mut right) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let prev = path[i.saturating_sub(1)];
        let next = path[(i + 1).min(n - 1)];
        let (mut tx, mut ty) = (next[0] - prev[0], next[1] - prev[1]);
        let tl = (tx * tx + ty * ty).sqrt().max(1e-9);
        tx /= tl;
        ty /= tl;
        let (nx, ny) = (-ty, tx);
        let hw = half(i);
        left.push([path[i][0] + nx * hw, path[i][1] + ny * hw]);
        right.push([path[i][0] - nx * hw, path[i][1] - ny * hw]);
    }
    let with_bbox = |poly: Vec<[f64; 2]>| -> (Vec<[f64; 2]>, [f64; 4]) {
        let bb = [
            poly.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
            poly.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
            poly.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max),
            poly.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max),
        ];
        (poly, bb)
    };
    let mut shapes: Vec<(Vec<[f64; 2]>, [f64; 4])> = Vec::with_capacity(n);
    for i in 0..n - 1 {
        shapes.push(with_bbox(vec![
            to_grid(left[i]),
            to_grid(left[i + 1]),
            to_grid(right[i + 1]),
            to_grid(right[i]),
        ]));
    }
    if SPRING_R > 0.0 {
        let s = river[0];
        let ring: Vec<[f64; 2]> = (0..16)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / 16.0;
                to_grid([s[0] + SPRING_R * a.cos(), s[1] + SPRING_R * a.sin()])
            })
            .collect();
        shapes.push(with_bbox(ring));
    }
    shapes
}

/// Axis-aligned bounding box `(minx, miny, maxx, maxy)` of a triangle's 2D verts.
fn tri_bbox(t: &[[f64; 2]; 3]) -> (f64, f64, f64, f64) {
    let xs = [t[0][0], t[1][0], t[2][0]];
    let ys = [t[0][1], t[1][1], t[2][1]];
    (
        xs.iter().copied().fold(f64::INFINITY, f64::min),
        ys.iter().copied().fold(f64::INFINITY, f64::min),
        xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ys.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// Intersection of convex polygon `subject` with convex polygon `clip`
/// (Sutherland–Hodgman). `clip` is reoriented CCW first; returns `subject ∩ clip`
/// (possibly empty), which is what carves a ribbon piece out of a terrain triangle.
fn clip_convex(subject: &[[f64; 2]], clip: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut cl = clip.to_vec();
    if signed_area(&cl) < 0.0 {
        cl.reverse();
    }
    let mut out = subject.to_vec();
    let m = cl.len();
    for e in 0..m {
        if out.is_empty() {
            break;
        }
        let a = cl[e];
        let b = cl[(e + 1) % m];
        let edge = [b[0] - a[0], b[1] - a[1]];
        // Inside = left of the directed edge (interior of a CCW polygon).
        let inside = |p: [f64; 2]| edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= 0.0;
        let input = std::mem::take(&mut out);
        let k = input.len();
        for i in 0..k {
            let cur = input[i];
            let nxt = input[(i + 1) % k];
            let (ci, ni) = (inside(cur), inside(nxt));
            if ci {
                out.push(cur);
            }
            if ci != ni {
                let t = segment_line_t(cur, nxt, a, b);
                out.push([
                    cur[0] + (nxt[0] - cur[0]) * t,
                    cur[1] + (nxt[1] - cur[1]) * t,
                ]);
            }
        }
    }
    out
}

/// Signed area of a polygon (positive when wound counter-clockwise).
fn signed_area(poly: &[[f64; 2]]) -> f64 {
    let n = poly.len();
    let mut s = 0.0;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        s += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * s
}

/// Parameter `t` along segment `p→q` where it crosses the (infinite) line `a→b`.
fn segment_line_t(p: [f64; 2], q: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let r = [q[0] - p[0], q[1] - p[1]];
    let s = [b[0] - a[0], b[1] - a[1]];
    let denom = r[0] * s[1] - r[1] * s[0];
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    ((a[0] - p[0]) * s[1] - (a[1] - p[1]) * s[0]) / denom
}

/// Height at ground point `p` inside terrain triangle `verts` (`[gx, gy, h]`), by
/// barycentric interpolation. The triangle is planar in height, so this is exact
/// and places a river piece exactly on the terrain surface.
fn bary_height(p: [f64; 2], verts: &[[f64; 3]; 3]) -> f64 {
    let (a, b, c) = (verts[0], verts[1], verts[2]);
    let v0 = [b[0] - a[0], b[1] - a[1]];
    let v1 = [c[0] - a[0], c[1] - a[1]];
    let v2 = [p[0] - a[0], p[1] - a[1]];
    let d00 = v0[0] * v0[0] + v0[1] * v0[1];
    let d01 = v0[0] * v1[0] + v0[1] * v1[1];
    let d11 = v1[0] * v1[0] + v1[1] * v1[1];
    let d20 = v2[0] * v0[0] + v2[1] * v0[1];
    let d21 = v2[0] * v1[0] + v2[1] * v1[1];
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-12 {
        return a[2];
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    (1.0 - v - w) * a[2] + v * b[2] + w * c[2]
}

/// A small low-poly conifer: stacked triangle tiers over a short trunk, each
/// tier split lit/shadow down the center seam. Copied from the logo.
fn draw_tree(svg: &mut Svg, pal: &Palette, base: (f64, f64)) {
    let (bx, by) = base;
    let w = 13.0;
    let trunk_h = 7.0;
    let tier_h = 17.0;

    svg.rect(bx - 1.9, by - trunk_h, 3.8, trunk_h + 1.0, pal.trunk);
    let top = by - trunk_h;
    for t in 0..3 {
        let level = top - t as f64 * (tier_h * 0.6);
        let half = w * (1.0 - t as f64 * 0.22);
        let apex = (bx, level - tier_h);
        svg.polygon(
            &[apex, (bx - half, level), (bx, level)],
            pal.tree_dark,
            None,
        );
        svg.polygon(&[apex, (bx, level), (bx + half, level)], pal.tree_lit, None);
    }
}

// vector helpers
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ---------------------------------------------------------------------------
// minimal SVG writer (square viewBox via tracked bounding box)
// ---------------------------------------------------------------------------

struct Svg {
    body: String,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl Svg {
    fn new() -> Self {
        Self {
            body: String::new(),
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        }
    }

    fn track(&mut self, x: f64, y: f64) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }

    fn points(pts: &[(f64, f64)]) -> String {
        let mut d = String::new();
        for (k, &(x, y)) in pts.iter().enumerate() {
            let _ = write!(d, "{}{:.2},{:.2}", if k == 0 { "" } else { " " }, x, y);
        }
        d
    }

    fn polygon(&mut self, pts: &[(f64, f64)], fill: Rgb, stroke: Option<(Rgb, f64)>) {
        for &(x, y) in pts {
            self.track(x, y);
        }
        let s = match stroke {
            Some((c, w)) => format!(
                r#" stroke="{}" stroke-width="{:.2}" stroke-linejoin="round""#,
                c.hex(),
                w
            ),
            None => String::new(),
        };
        let _ = write!(
            self.body,
            r#"<polygon points="{}" fill="{}"{}/>"#,
            Self::points(pts),
            fill.hex(),
            s
        );
        self.body.push('\n');
    }

    fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: Rgb) {
        self.track(x, y);
        self.track(x + w, y + h);
        let _ = write!(
            self.body,
            r#"<rect x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" fill="{}"/>"#,
            x,
            y,
            w,
            h,
            fill.hex()
        );
        self.body.push('\n');
    }

    /// The square viewBox `(min_x, min_y, side)`: the longer bbox axis plus
    /// padding sets the side, and the basin is centered within it (shorter axis
    /// gets symmetric transparent margins). Shared by [`finish`](Self::finish)
    /// and [`compose_adaptive`] so the stacked day/night renders frame alike.
    fn viewbox(&self) -> (f64, f64, f64) {
        let bw = self.max_x - self.min_x;
        let bh = self.max_y - self.min_y;
        let side = bw.max(bh) + 2.0 * PAD;
        let cx = 0.5 * (self.min_x + self.max_x);
        let cy = 0.5 * (self.min_y + self.max_y);
        (cx - side / 2.0, cy - side / 2.0, side)
    }

    /// Frame to a **square** viewBox and emit the standalone SVG document.
    /// Transparent canvas, no background rect.
    fn finish(self) -> String {
        let (vx, vy, side) = self.viewbox();
        let mut out = String::new();
        let _ = write!(
            out,
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{:.2} {:.2} {:.2} {:.2}" width="{:.0}" height="{:.0}">"#,
            vx, vy, side, side, side, side
        );
        out.push('\n');
        out.push_str(&self.body);
        out.push_str("</svg>\n");
        out
    }
}

// ---------------------------------------------------------------------------
// small color helper (Oklab blending, shared with the logo)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Rgb(u8, u8, u8);

impl Rgb {
    /// Blend toward `other` by `t ∈ [0, 1]` in **Oklab** (perceptual) so
    /// midpoints stay even instead of going muddy.
    fn lerp(self, other: Rgb, t: f64) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        let (p, q) = (rgb_to_oklab(self), rgb_to_oklab(other));
        oklab_to_rgb([
            p[0] + (q[0] - p[0]) * t,
            p[1] + (q[1] - p[1]) * t,
            p[2] + (q[2] - p[2]) * t,
        ])
    }
    fn shade(self, k: f64) -> Rgb {
        let f = |c: u8| (c as f64 * k).round().clamp(0.0, 255.0) as u8;
        Rgb(f(self.0), f(self.1), f(self.2))
    }
    fn hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.0, self.1, self.2)
    }
}

/// Sample a continuous color ramp at `t ∈ [0, 1]` between equally-spaced stops.
fn ramp_sample(stops: &[Rgb], t: f64) -> Rgb {
    let t = t.clamp(0.0, 1.0);
    if stops.len() < 2 {
        return stops[0];
    }
    let s = t * (stops.len() - 1) as f64;
    let i = (s.floor() as usize).min(stops.len() - 2);
    stops[i].lerp(stops[i + 1], s - i as f64)
}

fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn rgb_to_oklab(c: Rgb) -> [f64; 3] {
    let r = srgb_to_linear(c.0 as f64 / 255.0);
    let g = srgb_to_linear(c.1 as f64 / 255.0);
    let b = srgb_to_linear(c.2 as f64 / 255.0);
    let l = (0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b).cbrt();
    let m = (0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b).cbrt();
    let s = (0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b).cbrt();
    [
        0.210_454_255_3 * l + 0.793_617_785_0 * m - 0.004_072_046_8 * s,
        1.977_998_495_1 * l - 2.428_592_205_0 * m + 0.450_593_709_9 * s,
        0.025_904_037_1 * l + 0.782_771_766_2 * m - 0.808_675_766_0 * s,
    ]
}

fn oklab_to_rgb(lab: [f64; 3]) -> Rgb {
    let l = (lab[0] + 0.396_337_777_4 * lab[1] + 0.215_803_757_3 * lab[2]).powi(3);
    let m = (lab[0] - 0.105_561_345_8 * lab[1] - 0.063_854_172_8 * lab[2]).powi(3);
    let s = (lab[0] - 0.089_484_177_5 * lab[1] - 1.291_485_548_0 * lab[2]).powi(3);
    let r = 4.076_741_662_1 * l - 3.307_711_591_3 * m + 0.230_969_929_2 * s;
    let g = -1.268_438_004_6 * l + 2.609_757_401_1 * m - 0.341_319_396_5 * s;
    let b = -0.004_196_086_3 * l - 0.703_418_614_7 * m + 1.707_614_701_0 * s;
    let q = |c: f64| (linear_to_srgb(c).clamp(0.0, 1.0) * 255.0).round() as u8;
    Rgb(q(r), q(g), q(b))
}

/// Parse an HTML hex color (`"#52796f"` or `"52796f"`) into an [`Rgb`]. A
/// `const fn`, so it works inside `const` palette tables.
const fn hex(s: &str) -> Rgb {
    let b = s.as_bytes();
    let o = b.len() - 6;
    Rgb(
        (nibble(b[o]) << 4) | nibble(b[o + 1]),
        (nibble(b[o + 2]) << 4) | nibble(b[o + 3]),
        (nibble(b[o + 4]) << 4) | nibble(b[o + 5]),
    )
}

const fn nibble(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => 0,
    }
}
