//! Generative logo for `basin`.
//!
//! Renders the package's namesake (a geographic *basin*) as a low-poly
//! isometric scene, and does it the way an optimization library should:
//! the terrain is a quadratic basin (the canonical convex optimization
//! landscape) carrying a damped sinusoidal ripple for organic ridges, and
//! the rivulet winding down into the pool is an actual gradient-descent
//! trajectory traced by basin's [`GradientDescent`] solver with heavy-ball
//! momentum, run over that exact surface. The pool is the surface's
//! minimum, exactly where the optimizer comes to rest. Momentum is what
//! makes the descent glide down the slope and settle in the basin floor
//! rather than darting straight in, so the river is, literally,
//! optimization output.
//!
//! There are two themes: a daytime scene (sun, butterfly) and a moonlit
//! night scene (full moon + stars, owl). The night palette is derived from
//! the day one by a single Oklab "moonlight" transform (darken, desaturate,
//! cool), so the two stay in sync when you re-roll the day colors.
//!
//! Run with:
//!
//! ```text
//! cargo run --example logo                 # day theme   -> basin-logo.svg
//! cargo run --example logo out.svg         # day theme   -> out.svg
//! cargo run --example logo out.svg dark    # night theme -> out.svg
//! ```
//!
//! Everything is driven by the constants in the `config` block below;
//! tweak the palette, grid resolution, the framing window, or the
//! momentum parameters to re-roll the look. Output is a standalone SVG.

use std::fmt::Write as _;

use basin::{BasicState, CostFunction, Gradient, GradientDescent, Problem, Solver, State};

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

/// Grid cells per side (vertices = `GRID + 1`). Low enough to read as
/// faceted low-poly, high enough to keep the bowl curve smooth.
const GRID: usize = 22;

/// Problem-space framing window `[X0, X1] × [Y0, Y1]`.
const X0: f64 = -4.2;
const X1: f64 = 4.2;
const Y0: f64 = -4.2;
const Y1: f64 = 4.2;

/// Surface shape: a closed, *curved* (banana) valley. In coordinates
/// `(u, v)` rotated by `BOWL_ROT` about the center, the floor follows the
/// parabola `v = VALLEY_CURVE·u²`; the surface is soft along that floor
/// (`VALLEY_FLOOR_K`) and stiff across it (`VALLEY_WALL_K`), and rises at
/// both ends so the minimum is *enclosed* in the middle. Gradient descent
/// drops to the floor and follows its bend into the pool, a curved river,
/// not a straight radial line, and one that can't spill off an open valley
/// mouth. Centered slightly off origin so the pool lands off-center.
const BOWL_CX: f64 = 0.3;
const BOWL_CY: f64 = -0.3;
const VALLEY_FLOOR_K: f64 = 0.110; // along-valley softness (smaller = longer)
const VALLEY_WALL_K: f64 = 0.020; // across-valley stiffness
const VALLEY_CURVE: f64 = 0.200; // floor bend (parabola coefficient)
const BOWL_ROT: f64 = 0.1; // valley orientation (radians)
const RIM_K: f64 = 0.002;

/// Water depth as a normalized height above the basin floor: a small,
/// fixed pond rather than a percentile flood.
const WATER_LEVEL: f64 = 0.004;

/// Isometric tile half-extents (screen px) and vertical height gain.
const TILE_W: f64 = 20.0;
const TILE_H: f64 = 11.0;
const Z_SCREEN: f64 = 190.0; // screen px from basin floor to highest ridge
const Z_WORLD: f64 = 25.0; // height gain for facet-normal lighting (grid units)

/// River = gradient descent (light heavy-ball momentum) on the surface,
/// from a point up one arm of the curved valley. Momentum is kept low so
/// the descent *hugs* the bending floor rather than flying ballistically
/// across it; the path sweeps down the curve into the pool.
const RIVER_START: [f64; 2] = [-3.7, -2.5];
const RIVER_ALPHA: f64 = 0.05;
const RIVER_BETA: f64 = 0.35;
const RIVER_ITERS: usize = 1100;
const RIVER_POINTS: usize = 500; // trajectory vertices captured

/// River carved into the terrain mesh along the descent trajectory, the same
/// intersect-into-the-mesh treatment as the favicon. A ribbon tessellated to
/// `RIVER_SEGS` segments (so its edge is smooth, independent of `GRID`) tapers
/// from `RIVER_W_SRC` at the source to `RIVER_W_MOUTH` at the mouth (problem
/// units), and is intersected with each terrain triangle so the river is real
/// slope-lit sub-faces of the surface, not a flat ribbon laid on top.
const RIVER_W_SRC: f64 = 0.08;
const RIVER_W_MOUTH: f64 = 0.55;
const RIVER_SEGS: usize = 64;

/// Source spring at x₀: a round pool (problem-space radius) the river wells out
/// of, unioned into the ribbon and likewise carved into the mesh.
const SPRING_R: f64 = 0.2;

/// Trees: scattered at random over *plantable* ground, above the shoreline,
/// below the upper walls, and on gentle enough slopes to read as planted.
/// Placement is fully determined by `TREE_SEED`, so bump it to resample a
/// different arrangement (or raise `TREE_COUNT` for a denser stand).
const TREE_SEED: u64 = 15;
const TREE_COUNT: usize = 6;
const TREE_MIN_DIST: f64 = 1.6; // min separation between trees (problem units)
const TREE_MIN_LIFT: f64 = 0.05; // min normalized height above the water line
const TREE_MAX_HN: f64 = 0.5; // max normalized height (keeps trees off the rim)
const TREE_MAX_SLOPE: f64 = 0.13; // reject ground steeper than this
const TREE_MAX_ATTEMPTS: usize = 4000; // rejection-sampling budget per render

/// Rocks: a few low-poly boulders. One is always placed beside the source
/// spring (a screen-space offset from x₀); the rest are scattered at random,
/// reproducibly from `ROCK_SEED`. Rocks tolerate steeper, higher ground than
/// trees, so their slope/elevation bounds are looser. `ROCK_COUNT` is the
/// total including the spring rock.
const ROCK_SEED: u64 = 3;
const ROCK_COUNT: usize = 5;
const ROCK_SPRING_DX: f64 = -9.0; // spring rock: screen offset from the spring (px)
const ROCK_SPRING_DY: f64 = -5.0; // (screen-space so it nestles beside the water)
const ROCK_MIN_DIST: f64 = 1.4; // min separation between rocks (problem units)
const ROCK_MIN_LIFT: f64 = 0.02; // min normalized height above the water line
const ROCK_MAX_HN: f64 = 0.7; // max normalized height (rocks climb higher)
const ROCK_MAX_SLOPE: f64 = 0.22; // rocks sit on steeper ground than trees
const ROCK_MAX_ATTEMPTS: usize = 4000; // rejection-sampling budget per render
const ROCK_W: f64 = 14.0; // boulder half-width (px)
const ROCK_H: f64 = 16.0; // boulder height (px)

/// Teal/slate palette.
const PAPER: Rgb = Rgb(244, 241, 222); // #f4f1de background
// Terrain hypsometric ramps (low elevation → high), each sampled from the land
// section of a named colormap. To test a palette, change the `TERRAIN_RAMP`
// line at the bottom of this block to `&TURKU`, `&BILBAO`, `&BAMAKO`, or
// `&SANDSTONE`, save, and look at `images/logo.svg` (run `task logo` and it
// re-renders on every save). Add your own by pasting any colormap's stops;
// `ramp_sample` interpolates across however many you give it.
#[allow(dead_code)] // Crameri `lajolla`: vivid red-rock or sunset
const SANDSTONE: [Rgb; 6] = [
    hex("#ca514b"),
    hex("#df6e4f"),
    hex("#e68c51"),
    hex("#eba853"),
    hex("#f2c75c"),
    hex("#fbea93"),
];
#[allow(dead_code)] // Crameri `turku`: muted olive/khaki → warm tan
const TURKU: [Rgb; 6] = [
    hex("#565640"),
    hex("#6d6c4a"),
    hex("#868255"),
    hex("#a79864"),
    hex("#c6a475"),
    hex("#dda888"),
];
#[allow(dead_code)] // Crameri `bilbao`: clay-rose → pale gray-tan
const BILBAO: [Rgb; 6] = [
    hex("#a36b59"),
    hex("#a87d5d"),
    hex("#ad9061"),
    hex("#b5a874"),
    hex("#c1bb9e"),
    hex("#cccac3"),
];
#[allow(dead_code)] // Crameri `bamako`: green lowlands → gold peaks
const BAMAKO: [Rgb; 6] = [
    hex("#335b28"),
    hex("#537014"),
    hex("#788501"),
    hex("#a2930d"),
    hex("#ceb546"),
    hex("#f3d993"),
];
#[allow(dead_code)] // Crameri `bamako`: green lowlands → gold peaks
const CUSTOM: [Rgb; 5] = [
    hex("#5E7763"),
    hex("#7C7A62"),
    hex("#9C836F"),
    hex("#B59A85"),
    hex("#DDD5CB"),
];
/// The active terrain ramp. ← change this one line to test a different palette.
const TERRAIN_RAMP: &[Rgb] = &CUSTOM;
const WATER: Rgb = Rgb(71, 153, 173); // lake + river (one melted body)
const TREE_DARK: Rgb = Rgb(54, 90, 82); // canopy shadow side
const TREE_LIT: Rgb = Rgb(92, 138, 122); // canopy lit side
const TRUNK: Rgb = Rgb(52, 46, 39);
const ROCK_LIT: Rgb = Rgb(150, 161, 160); // boulder sun-facing facet
const ROCK_DARK: Rgb = Rgb(92, 108, 109); // boulder shadow facet
const SUN: Rgb = Rgb(233, 196, 106); // #e9c46a accent

/// Sun: a soft halo around a solid disk in the upper-right. `SUN_R` is the core
/// radius (screen px); the halo scales with it via `SUN_HALO`.
const SUN_R: f64 = 39.0;
const SUN_HALO: f64 = 1.45; // halo radius as a multiple of the core

/// Light direction in world space (`+z` up), pointing *from* the light.
const LIGHT: [f64; 3] = [0.5, -0.35, 0.79];

// ---------------------------------------------------------------------------
// theme (day or night)
// ---------------------------------------------------------------------------

/// Which palette + props the scene renders with. `Day` is the original
/// sunlit look; `Night` re-tints every color through [`moonlight`] and
/// swaps the props (sun → full moon + stars, butterfly → owl).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Theme {
    Day,
    Night,
}

/// Moonlit night accents (the day accents live in the palette block above).
const MOON: Rgb = hex("#d8dcec"); // pale, cool full-moon disk
const MOON_CRATER: Rgb = hex("#b6bdd2"); // slightly darker crater fill
const STAR: Rgb = hex("#e7ebf6"); // faint star points

/// Owl (night creature): moonlit slate body so it stays legible on a dark
/// background, with pale eyes and a warm beak.
const OWL_DARK: Rgb = hex("#535d75"); // body or shadow side
const OWL_LIT: Rgb = hex("#7e88a1"); // belly or lit side
const OWL_EYE: Rgb = hex("#eaeef8"); // pale eye disk
const OWL_PUPIL: Rgb = hex("#2b3142"); // dark pupil
const OWL_BEAK: Rgb = hex("#cba36a"); // small warm beak

/// Butterfly (day creature): warm wings drawn from the cream palette family.
const BFLY_WING: Rgb = hex("#e9c46a"); // upper wings (matches the sun accent)
const BFLY_WING2: Rgb = hex("#e07a5f"); // lower wings (terracotta)
const BFLY_BODY: Rgb = hex("#3d405b"); // dark body + antennae
/// Whole-butterfly tilt in degrees, about its body center (positive = clockwise
/// in SVG's y-down frame, so the head leans right). Keeps it reading as drifting
/// rather than pinned upright; flip the sign or scale to re-aim.
const BFLY_TILT: f64 = -19.0;

/// The creature hovers/perches over this problem-space point in the open
/// foreground meadow. The butterfly floats well above the surface
/// (`CREATURE_HOVER`); the owl perches close to it (`CREATURE_PERCH`).
const CREATURE_AT: [f64; 2] = [2.6, 1.7];
const CREATURE_HOVER: f64 = 0.20; // butterfly lift (normalized height)
const CREATURE_PERCH: f64 = 0.02; // owl lift (sits just above the ground)

/// Re-tint a daytime color for the night palette: darken, desaturate, and
/// cool toward moonlit blue, all in Oklab so the shift stays perceptually
/// even. A small lightness floor keeps nothing pure black. This single
/// transform is what derives the entire night terrain/water/tree/rock ramp
/// from the day palette, so re-rolling the day colors re-rolls night too.
fn moonlight(c: Rgb) -> Rgb {
    let [l, a, b] = rgb_to_oklab(c);
    oklab_to_rgb([
        l * 0.65 + 0.035, // darken to dusk (not black), a lifted floor
        a * 0.86,         // desaturate
        b * 0.56 - 0.018, // desaturate + push toward blue (−b is blue)
    ])
}

/// All theme-dependent colors, resolved once up front. Day uses the config
/// constants directly; Night maps each through [`moonlight`] and supplies
/// its own moon/halo accents. Draw functions read from here rather than the
/// raw constants so a single `Theme` value re-skins the whole scene.
struct Palette {
    theme: Theme,
    terrain: Vec<Rgb>, // elevation ramp stops
    water: Rgb,
    tree_dark: Rgb,
    tree_lit: Rgb,
    trunk: Rgb,
    rock_lit: Rgb,
    rock_dark: Rgb,
    orb: Rgb,      // sun (day) or moon (night) core
    orb_halo: Rgb, // halo color around the orb
}

impl Palette {
    fn new(theme: Theme) -> Self {
        match theme {
            Theme::Day => Self {
                theme,
                terrain: TERRAIN_RAMP.to_vec(),
                water: WATER,
                tree_dark: TREE_DARK,
                tree_lit: TREE_LIT,
                trunk: TRUNK,
                rock_lit: ROCK_LIT,
                rock_dark: ROCK_DARK,
                orb: SUN,
                orb_halo: SUN.lerp(PAPER, 0.55),
            },
            Theme::Night => Self {
                theme,
                terrain: TERRAIN_RAMP.iter().map(|&c| moonlight(c)).collect(),
                // Water is the hero (the river is optimizer output), so lift
                // it a touch toward the moon so it keeps a moonlit sheen
                // rather than sinking into the dark terrain.
                water: moonlight(WATER).lerp(MOON, 0.14),
                tree_dark: moonlight(TREE_DARK),
                tree_lit: moonlight(TREE_LIT),
                trunk: moonlight(TRUNK),
                rock_lit: moonlight(ROCK_LIT),
                rock_dark: moonlight(ROCK_DARK),
                orb: MOON,
                orb_halo: MOON, // halo drawn translucent in draw_sky
            },
        }
    }
}

// ---------------------------------------------------------------------------
// surface
// ---------------------------------------------------------------------------

/// Raw bowl height at problem point `(x, y)`: a quadratic basin with a
/// damped sinusoidal ripple (organic ridges) and a gentle rising rim. The
/// ripple fades to zero at the center so the basin floor, and hence the
/// pool, stays clean and the basin stays unimodal. This is the function
/// basin's solver descends to trace the river.
fn raw_height(x: f64, y: f64) -> f64 {
    let dx = x - BOWL_CX;
    let dy = y - BOWL_CY;
    let (ca, sa) = (BOWL_ROT.cos(), BOWL_ROT.sin());
    let u = ca * dx + sa * dy; // along the valley (soft)
    let v = -sa * dx + ca * dy; // across the valley (stiff)
    let across = v - VALLEY_CURVE * u * u; // signed distance from the curved floor
    let bowl = VALLEY_FLOOR_K * u * u + VALLEY_WALL_K * across * across;
    // Ripple keyed to `across` (perpendicular to the curved floor): it
    // textures the walls but fades to zero on the floor (`across ≈ 0`), so it
    // never traps the descent following the floor. A tiny u-term adds gentle
    // along-valley variation. `damp` grows with height, so the floor stays
    // clean.
    let damp = (bowl / 0.6).min(1.0);
    let w1 = damp * 0.20 * (1.30 * across + 0.4).sin();
    let w2 = damp * 0.10 * (2.40 * across - 0.7).sin();
    let w3 = damp * 0.05 * (0.70 * u + 0.3).sin();
    let rim = RIM_K * (x * x + y * y);
    bowl + w1 + w2 + w3 + rim
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
        let mut vals = Vec::with_capacity((n + 1) * (n + 1));
        for j in 0..=n {
            for i in 0..=n {
                let x = X0 + (X1 - X0) * i as f64 / n as f64;
                let y = Y0 + (Y1 - Y0) * j as f64 / n as f64;
                vals.push(raw_height(x, y));
            }
        }
        let hmin = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let hmax = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
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
    /// (normalized units) so rivers/trees ride just above their facets.
    fn surface_point(&self, x: f64, y: f64, lift: f64) -> (f64, f64) {
        let gx = (x - X0) / (X1 - X0) * GRID as f64;
        let gy = (y - Y0) / (Y1 - Y0) * GRID as f64;
        project(gx, gy, self.hn(x, y) + lift)
    }
}

/// Problem-space coordinates of grid node `(i, j)`.
fn node_xy(i: usize, j: usize) -> (f64, f64) {
    (
        X0 + (X1 - X0) * i as f64 / GRID as f64,
        Y0 + (Y1 - Y0) * j as f64 / GRID as f64,
    )
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

/// The rendered surface, exposed as a `CostFunction` + `Gradient` so
/// basin's [`GradientDescent`] can descend the exact terrain we draw. The
/// gradient is a central difference of [`raw_height`], no need to
/// hand-differentiate the ripple, and it stays perfectly consistent with
/// the surface.
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

/// Trace the river: heavy-ball gradient descent on the bowl, driven
/// through basin's `Solver` loop, capturing every iterate then decimating.
fn trace_river() -> Vec<[f64; 2]> {
    let mut solver = GradientDescent::new(RIVER_ALPHA).with_momentum(RIVER_BETA);
    let mut problem = Problem::new(Bowl);
    let mut state = solver
        .init(&mut problem, BasicState::new(RIVER_START.to_vec()))
        .unwrap();
    let mut full = Vec::with_capacity(RIVER_ITERS + 1);
    full.push([state.param()[0], state.param()[1]]);
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
    let out_path = args.next().unwrap_or_else(|| "basin-logo.svg".into());
    // Optional second arg selects the theme: `dark` or `night` → moonlit,
    // anything else (or absent) → the daytime scene.
    let theme = match args.next().as_deref() {
        Some("dark") | Some("night") => Theme::Night,
        _ => Theme::Day,
    };
    let pal = Palette::new(theme);

    let surf = Surface::new();
    let river = trace_river();
    // The basin minimum *is* the pool center, and where the river settles.
    let pool_center = basin_min();

    let mut svg = Svg::new();
    draw_sky(&mut svg, &pal);
    draw_terrain(&mut svg, &surf, &pal, &river);
    let n_rocks = draw_rocks(&mut svg, &surf, &pal);
    let n_trees = draw_trees(&mut svg, &surf, &pal);
    draw_creature(&mut svg, &surf, &pal); // butterfly (day) or owl (night), on top

    let doc = svg.finish();
    std::fs::write(&out_path, &doc).expect("write SVG");
    let end = river.last().copied().unwrap_or(pool_center);
    let theme_name = match theme {
        Theme::Day => "day",
        Theme::Night => "night",
    };
    eprintln!(
        "wrote {out_path} ({} bytes, {theme_name}); river: {} steps from {:?} to ({:.3}, {:.3}); pool at ({:.3}, {:.3}); {n_trees}/{TREE_COUNT} trees (seed {TREE_SEED}); {n_rocks}/{ROCK_COUNT} rocks (seed {ROCK_SEED})",
        doc.len(),
        river.len(),
        RIVER_START,
        end[0],
        end[1],
        pool_center[0],
        pool_center[1],
    );
}

/// True global minimum of the surface (fine grid search). The pool sits
/// here; on this unimodal basin it's also where the river settles.
fn basin_min() -> [f64; 2] {
    let n = 260;
    let (mut best, mut bv) = ([BOWL_CX, BOWL_CY], f64::INFINITY);
    for j in 0..=n {
        for i in 0..=n {
            let x = X0 + (X1 - X0) * i as f64 / n as f64;
            let y = Y0 + (Y1 - Y0) * j as f64 / n as f64;
            let val = raw_height(x, y);
            if val < bv {
                bv = val;
                best = [x, y];
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// drawing
// ---------------------------------------------------------------------------

/// One isometric triangle facet, flat-shaded by its normal and tinted by
/// elevation; drawn back-to-front by the caller.
struct Facet {
    pts: Vec<(f64, f64)>,
    depth: f64,
    color: Rgb,
}

/// Sample a continuous color ramp at `t ∈ [0, 1]` between equally-spaced anchor
/// stops, a poor-man's colormap lookup. Drop in any number of stops (e.g. a
/// hypsometric/topographic colormap) and this maps normalized elevation to
/// color. Blending goes through [`Rgb::lerp`], which interpolates in Oklab.
fn ramp_sample(stops: &[Rgb], t: f64) -> Rgb {
    let t = t.clamp(0.0, 1.0);
    if stops.len() < 2 {
        return stops[0];
    }
    let s = t * (stops.len() - 1) as f64;
    let i = (s.floor() as usize).min(stops.len() - 2);
    stops[i].lerp(stops[i + 1], s - i as f64)
}

/// sRGB channel (0–1) → linear light.
fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear light → sRGB channel (0–1).
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

fn draw_terrain(svg: &mut Svg, surf: &Surface, pal: &Palette, river: &[[f64; 2]]) {
    let wl = surf.water;
    // True normalized height per node. The lake is carved into *this* mesh by
    // clipping every triangle against the water plane `z = wl`; the river is the
    // smooth ribbon along the descent trajectory ([`river_shapes`]) intersected
    // with each triangle, so both lake and river are real sub-faces of the
    // surface, not shapes laid on top. The river follows the ground's curvature
    // and is lit by the same slope normal; the lake is flat at the water line.
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
    let water_flat = pal.water.shade(0.70 + 0.5 * light[2].max(0.0));
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
                let depth = verts.iter().map(|v| v[0] + v[1]).sum::<f64>() / 3.0;

                // Terrain shade from the true-height surface normal.
                let scaled = |v: [f64; 3]| [v[0], v[1], v[2] * Z_WORLD];
                let mut n = normalize3(cross(
                    sub(scaled(verts[1]), scaled(verts[0])),
                    sub(scaled(verts[2]), scaled(verts[0])),
                ));
                if n[2] < 0.0 {
                    n = [-n[0], -n[1], -n[2]];
                }
                let ndotl = (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]).max(0.0);

                // Terrain (above water) and lake (below water), via the height clip.
                let land = clip_to_water(&verts, wl, true);
                if land.len() >= 3 {
                    let elev = land.iter().map(|v| v[2]).sum::<f64>() / land.len() as f64;
                    let pts = land.iter().map(|v| project(v[0], v[1], v[2])).collect();
                    terr.push(Facet {
                        pts,
                        depth,
                        color: ramp_sample(&pal.terrain, elev).shade(0.70 + 0.5 * ndotl),
                    });
                }
                let below = clip_to_water(&verts, wl, false);
                if below.len() >= 3 {
                    let pts = below.iter().map(|v| project(v[0], v[1], wl)).collect();
                    lake.push(Facet {
                        pts,
                        depth,
                        color: water_flat,
                    });
                }

                // River: intersect this triangle's footprint with the ribbon
                // polygons; each piece rides on the terrain plane (heights
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

    // Three back-to-front passes (terrain, river, lake). Each facet is stroked in
    // its *own* fill color so adjacent same-color facets melt together (covering
    // the anti-aliasing seam that would otherwise show the cream background as a
    // mesh of lines); drawing river then lake last keeps each water edge clean.
    terr.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());
    rivr.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());
    lake.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());
    for f in terr.iter().chain(rivr.iter()).chain(lake.iter()) {
        svg.polygon(&f.pts, f.color, Some((f.color, 1.0)));
    }
}

/// Clip a triangle (grid coords with true height in `z`) against the water plane,
/// keeping the half with `z ≥ wl` (`keep_land`) or `z ≤ wl`. Sutherland–Hodgman
/// against a single half-space; the crossing points land exactly on `z = wl`, so
/// the land and water pieces share the shoreline edge. Returns the clipped
/// polygon's vertices (0, 3 or 4).
fn clip_to_water(verts: &[[f64; 3]; 3], wl: f64, keep_land: bool) -> Vec<[f64; 3]> {
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

/// The river footprint as a strip of convex polygons in **grid coordinates**,
/// each paired with its bounding box `[minx, miny, maxx, maxy]` for cheap
/// culling. The strip follows the descent trajectory (tessellated to `RIVER_SEGS`
/// segments, smooth edge regardless of `GRID`), tapering from `RIVER_W_SRC` at
/// the source to `RIVER_W_MOUTH` toward the mouth (problem units). If
/// `SPRING_R > 0`, a round source pool is appended. [`draw_terrain`] intersects
/// each terrain triangle with these to carve the river into the mesh.
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
/// (possibly empty), which carves a ribbon piece out of a terrain triangle.
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

/// Scatter trees at random over plantable ground, reproducibly from
/// `TREE_SEED`. We rejection-sample problem-space points, keeping those that
/// sit above the shoreline, below the upper walls, on gentle slopes, and far
/// enough from already-placed trees. Returns how many were actually placed;
/// a tight `TREE_MIN_DIST` or a small eligible area can fall short of
/// `TREE_COUNT` within the attempt budget.
fn draw_trees(svg: &mut Svg, surf: &Surface, pal: &Palette) -> usize {
    let wl = surf.water;
    // Slope sampled over one grid cell in each axis, matching the spacing the
    // `TREE_MAX_SLOPE` threshold was tuned against.
    let dx = (X1 - X0) / GRID as f64;
    let dy = (Y1 - Y0) / GRID as f64;
    let mut rng = Rng::new(TREE_SEED);
    let mut chosen: Vec<(f64, f64)> = Vec::new();
    for _ in 0..TREE_MAX_ATTEMPTS {
        if chosen.len() >= TREE_COUNT {
            break;
        }
        let x = X0 + (X1 - X0) * rng.next_f64();
        let y = Y0 + (Y1 - Y0) * rng.next_f64();
        let h = surf.hn(x, y);
        if h < wl + TREE_MIN_LIFT || h > TREE_MAX_HN {
            continue;
        }
        let slope = (surf.hn(x + dx, y) - surf.hn(x - dx, y)).abs()
            + (surf.hn(x, y + dy) - surf.hn(x, y - dy)).abs();
        if slope > TREE_MAX_SLOPE {
            continue;
        }
        if chosen
            .iter()
            .any(|&(cx, cy)| (x - cx).hypot(y - cy) < TREE_MIN_DIST)
        {
            continue;
        }
        chosen.push((x, y));
    }

    chosen.sort_by(|a, b| (a.0 + a.1).partial_cmp(&(b.0 + b.1)).unwrap()); // far trees first
    for &(x, y) in &chosen {
        draw_tree(svg, pal, surf.surface_point(x, y, 0.0));
    }
    chosen.len()
}

/// A small low-poly conifer: stacked triangle tiers over a short trunk,
/// each tier split lit/shadow down the center seam.
fn draw_tree(svg: &mut Svg, pal: &Palette, base: (f64, f64)) {
    let (bx, by) = base;
    let w = 12.0;
    let trunk_h = 7.0;
    let tier_h = 16.0;

    svg.rect(bx - 1.8, by - trunk_h, 3.6, trunk_h + 1.0, pal.trunk);
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

/// Scatter low-poly boulders: one nestled beside the source spring, the rest
/// sampled at random (seeded by `ROCK_SEED`) over rocky ground. Mirrors
/// `draw_trees`' eligibility test with the looser `ROCK_*` bounds. Returns how
/// many were placed (incl. the spring rock).
fn draw_rocks(svg: &mut Svg, surf: &Surface, pal: &Palette) -> usize {
    let wl = surf.water;
    let dx = (X1 - X0) / GRID as f64;
    let dy = (Y1 - Y0) / GRID as f64;
    let mut rng = Rng::new(ROCK_SEED);
    // The spring rock is placed by a screen-space offset (so it sits right next
    // to the water at the same height); the random rocks keep clear of x₀.
    let mut chosen: Vec<(f64, f64)> = vec![(RIVER_START[0], RIVER_START[1])];
    for _ in 0..ROCK_MAX_ATTEMPTS {
        if chosen.len() >= ROCK_COUNT {
            break;
        }
        let x = X0 + (X1 - X0) * rng.next_f64();
        let y = Y0 + (Y1 - Y0) * rng.next_f64();
        let h = surf.hn(x, y);
        if h < wl + ROCK_MIN_LIFT || h > ROCK_MAX_HN {
            continue;
        }
        let slope = (surf.hn(x + dx, y) - surf.hn(x - dx, y)).abs()
            + (surf.hn(x, y + dy) - surf.hn(x, y - dy)).abs();
        if slope > ROCK_MAX_SLOPE {
            continue;
        }
        if chosen
            .iter()
            .any(|&(cx, cy)| (x - cx).hypot(y - cy) < ROCK_MIN_DIST)
        {
            continue;
        }
        chosen.push((x, y));
    }

    // Spring rock first (it sits high/back), then the scattered rocks far-first.
    let (sx, sy) = surf.surface_point(RIVER_START[0], RIVER_START[1], 0.0);
    draw_rock(svg, pal, (sx + ROCK_SPRING_DX, sy + ROCK_SPRING_DY));
    let mut scattered = chosen[1..].to_vec();
    scattered.sort_by(|a, b| (a.0 + a.1).partial_cmp(&(b.0 + b.1)).unwrap());
    for &(x, y) in &scattered {
        draw_rock(svg, pal, surf.surface_point(x, y, 0.0));
    }
    chosen.len()
}

/// A small faceted boulder on the ground at `base`: a shadow (left) facet and
/// a sun-lit (right) facet split by a ridge, matching the trees' lit/shadow
/// treatment. The right side is lit because the sun sits upper-right.
fn draw_rock(svg: &mut Svg, pal: &Palette, base: (f64, f64)) {
    let (bx, by) = base;
    let (w, h) = (ROCK_W, ROCK_H);
    let top = (bx - 0.10 * w, by - h);
    let ul = (bx - 0.95 * w, by - 0.50 * h);
    let ll = (bx - 0.50 * w, by - 0.02 * h);
    let seam = (bx + 0.05 * w, by);
    let lr = (bx + 0.70 * w, by - 0.06 * h);
    let ur = (bx + 0.95 * w, by - 0.55 * h);
    svg.polygon(&[top, ul, ll, seam], pal.rock_dark, None); // shadow (left)
    svg.polygon(&[top, seam, lr, ur], pal.rock_lit, None); // lit (right)
}

/// The orb in the upper-right: a sun with a warm halo by day, a pale full
/// moon with a soft cool glow, a couple of craters, and scattered stars by
/// night. Star/halo extents are kept inside the day frame so both themes
/// render at the same size.
fn draw_sky(svg: &mut Svg, pal: &Palette) {
    let (sx, sy) = project(GRID as f64 * 0.66, 0.0, 1.32);
    let (cx, cy) = (sx, sy - 4.0);
    match pal.theme {
        Theme::Day => {
            svg.circle(cx, cy, SUN_R * SUN_HALO, pal.orb_halo);
            svg.circle(cx, cy, SUN_R, pal.orb);
        }
        Theme::Night => {
            // Stars first, behind the moon's glow. Offsets are relative to the
            // moon and kept within the day frame (terrain rim ≈ y −190).
            for &(dx, dy, r, op) in STARS {
                svg.circle_opacity(cx + dx, cy + dy, r, STAR, op);
            }
            // Soft cool glow: two faint discs, larger and dimmer outward.
            svg.circle_opacity(cx, cy, SUN_R * SUN_HALO, pal.orb_halo, 0.16);
            svg.circle_opacity(cx, cy, SUN_R * 1.18, pal.orb_halo, 0.22);
            // Moon disk + a couple of craters for a non-sun read.
            svg.circle(cx, cy, SUN_R, pal.orb);
            svg.circle(
                cx - 0.32 * SUN_R,
                cy - 0.18 * SUN_R,
                0.20 * SUN_R,
                MOON_CRATER,
            );
            svg.circle(
                cx + 0.22 * SUN_R,
                cy + 0.26 * SUN_R,
                0.14 * SUN_R,
                MOON_CRATER,
            );
            svg.circle(
                cx + 0.10 * SUN_R,
                cy - 0.34 * SUN_R,
                0.10 * SUN_R,
                MOON_CRATER,
            );
        }
    }
}

/// Star field for the night sky: `(dx, dy, radius, opacity)` offsets from the
/// moon center, all up-and-left/right of it in open sky and inside the day
/// frame so the two themes stay the same size.
const STARS: &[(f64, f64, f64, f64)] = &[
    (-150.0, -8.0, 1.9, 0.95),
    (-104.0, -55.0, 1.3, 0.7),
    (-186.0, -40.0, 1.6, 0.85),
    (-70.0, -78.0, 2.1, 1.0),
    (40.0, -70.0, 1.5, 0.85),
    (96.0, -30.0, 1.2, 0.6),
    (128.0, -78.0, 1.8, 0.95),
    (-26.0, -92.0, 1.3, 0.7),
    (150.0, -16.0, 1.1, 0.6),
    (62.0, -94.0, 1.5, 0.8),
];

/// The themed creature: a butterfly hovering over the valley by day, an owl
/// perched in the foreground meadow by night. Drawn last so it sits on top.
fn draw_creature(svg: &mut Svg, surf: &Surface, pal: &Palette) {
    let (x, y) = (CREATURE_AT[0], CREATURE_AT[1]);
    match pal.theme {
        Theme::Day => draw_butterfly(svg, surf.surface_point(x, y, CREATURE_HOVER)),
        Theme::Night => draw_owl(svg, surf.surface_point(x, y, CREATURE_PERCH)),
    }
}

/// A small flat-shaded butterfly at `base` (its body center): a dark body,
/// two larger upper wings and two smaller lower wings (tilted ellipses), and
/// two antennae. Sized to read at logo scale (~22 px wide).
fn draw_butterfly(svg: &mut Svg, base: (f64, f64)) {
    let (bx, by) = base;
    // The whole butterfly is tilted `BFLY_TILT` about its body center. `rot`
    // spins a point about `(bx, by)`; the same angle is folded into each wing's
    // own rotation so the splay survives the tilt.
    let (s, c) = BFLY_TILT.to_radians().sin_cos();
    let rot = |x: f64, y: f64| -> (f64, f64) {
        let (dx, dy) = (x - bx, y - by);
        (bx + dx * c - dy * s, by + dx * s + dy * c)
    };
    // Wings (drawn before the body so the body seam sits on top). The upper
    // pair is splayed outward and the lower pair tucked below so the four
    // wings + central body read clearly as a butterfly, not a single bloom.
    let (ulx, uly) = rot(bx - 8.0, by - 4.5);
    svg.ellipse(ulx, uly, 7.5, 9.5, -38.0 + BFLY_TILT, BFLY_WING); // upper left
    let (urx, ury) = rot(bx + 8.0, by - 4.5);
    svg.ellipse(urx, ury, 7.5, 9.5, 38.0 + BFLY_TILT, BFLY_WING); // upper right
    let (llx, lly) = rot(bx - 6.0, by + 6.0);
    svg.ellipse(llx, lly, 5.2, 6.5, -16.0 + BFLY_TILT, BFLY_WING2); // lower left
    let (lrx, lry) = rot(bx + 6.0, by + 6.0);
    svg.ellipse(lrx, lry, 5.2, 6.5, 16.0 + BFLY_TILT, BFLY_WING2); // lower right
    svg.ellipse(bx, by, 2.1, 9.0, BFLY_TILT, BFLY_BODY); // body (center = pivot)
    // Antennae.
    let (a0x, a0y) = rot(bx, by - 7.0);
    let (alx, aly) = rot(bx - 4.0, by - 13.0);
    let (arx, ary) = rot(bx + 4.0, by - 13.0);
    svg.line(a0x, a0y, alx, aly, 1.0, BFLY_BODY);
    svg.line(a0x, a0y, arx, ary, 1.0, BFLY_BODY);
    svg.circle(alx, aly, 1.2, BFLY_BODY);
    svg.circle(arx, ary, 1.2, BFLY_BODY);
}

/// A small low-poly owl at `base` (the perch point, between its feet): a
/// rounded two-tone body, ear tufts, two pale eyes with dark pupils, and a
/// warm beak. Rendered in moonlit slate so it stays visible on a dark page.
fn draw_owl(svg: &mut Svg, base: (f64, f64)) {
    let (bx, by) = base;
    // Body: a darker egg with a lighter belly offset down-right.
    svg.ellipse(bx, by - 13.0, 11.0, 14.5, 0.0, OWL_DARK);
    svg.ellipse(bx + 1.2, by - 9.0, 7.2, 10.0, 0.0, OWL_LIT);
    // Ear tufts.
    svg.polygon(
        &[
            (bx - 9.5, by - 22.0),
            (bx - 5.0, by - 30.0),
            (bx - 3.5, by - 22.5),
        ],
        OWL_DARK,
        None,
    );
    svg.polygon(
        &[
            (bx + 3.5, by - 22.5),
            (bx + 5.0, by - 30.0),
            (bx + 9.5, by - 22.0),
        ],
        OWL_DARK,
        None,
    );
    // Eyes (pale disks + pupils) and beak.
    svg.circle(bx - 4.6, by - 20.5, 4.0, OWL_EYE);
    svg.circle(bx + 4.6, by - 20.5, 4.0, OWL_EYE);
    svg.circle(bx - 4.6, by - 20.0, 1.8, OWL_PUPIL);
    svg.circle(bx + 4.6, by - 20.0, 1.8, OWL_PUPIL);
    svg.polygon(
        &[
            (bx - 1.8, by - 17.5),
            (bx + 1.8, by - 17.5),
            (bx, by - 14.0),
        ],
        OWL_BEAK,
        None,
    );
    // Feet.
    svg.line(bx - 3.5, by, bx - 3.5, by - 2.6, 1.6, OWL_BEAK);
    svg.line(bx + 3.5, by, bx + 3.5, by - 2.6, 1.6, OWL_BEAK);
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
// tiny deterministic PRNG (SplitMix64), reproducible tree scatter, no deps
// ---------------------------------------------------------------------------

/// Minimal seedable RNG so tree placement is fully determined by `TREE_SEED`
/// and identical across platforms. SplitMix64: fast, well-scrambled, and
/// stateless beyond a single `u64`, enough for sampling a handful of points.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` (top 53 bits → f64 mantissa).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// minimal SVG writer (auto-frames via tracked bounding box)
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

    fn circle(&mut self, cx: f64, cy: f64, r: f64, fill: Rgb) {
        self.track(cx - r, cy - r);
        self.track(cx + r, cy + r);
        let _ = write!(
            self.body,
            r#"<circle cx="{:.2}" cy="{:.2}" r="{:.2}" fill="{}"/>"#,
            cx,
            cy,
            r,
            fill.hex()
        );
        self.body.push('\n');
    }

    /// A circle with `fill-opacity` (for moon glow or stars).
    fn circle_opacity(&mut self, cx: f64, cy: f64, r: f64, fill: Rgb, opacity: f64) {
        self.track(cx - r, cy - r);
        self.track(cx + r, cy + r);
        let _ = write!(
            self.body,
            r#"<circle cx="{:.2}" cy="{:.2}" r="{:.2}" fill="{}" fill-opacity="{:.2}"/>"#,
            cx,
            cy,
            r,
            fill.hex(),
            opacity
        );
        self.body.push('\n');
    }

    /// An axis-aligned ellipse, optionally rotated `rot_deg` about its center
    /// (butterfly wings, owl body). The bounding box is tracked loosely from
    /// the larger radius so rotation can't push it out of frame.
    fn ellipse(&mut self, cx: f64, cy: f64, rx: f64, ry: f64, rot_deg: f64, fill: Rgb) {
        let m = rx.max(ry);
        self.track(cx - m, cy - m);
        self.track(cx + m, cy + m);
        let t = if rot_deg.abs() > 1e-9 {
            format!(r#" transform="rotate({:.2} {:.2} {:.2})""#, rot_deg, cx, cy)
        } else {
            String::new()
        };
        let _ = write!(
            self.body,
            r#"<ellipse cx="{:.2}" cy="{:.2}" rx="{:.2}" ry="{:.2}" fill="{}"{}/>"#,
            cx,
            cy,
            rx,
            ry,
            fill.hex(),
            t
        );
        self.body.push('\n');
    }

    /// A round-capped stroked line segment (antennae, owl feet).
    fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, w: f64, color: Rgb) {
        self.track(x1, y1);
        self.track(x2, y2);
        let _ = write!(
            self.body,
            r#"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{}" stroke-width="{:.2}" stroke-linecap="round"/>"#,
            x1,
            y1,
            x2,
            y2,
            color.hex(),
            w
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

    fn finish(self) -> String {
        let pad = 30.0;
        let min_x = self.min_x - pad;
        let min_y = self.min_y - pad;
        let w = (self.max_x - self.min_x) + 2.0 * pad;
        let h = (self.max_y - self.min_y) + 2.0 * pad;
        let mut out = String::new();
        let _ = write!(
            out,
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{:.2} {:.2} {:.2} {:.2}" width="{:.0}" height="{:.0}">"#,
            min_x, min_y, w, h, w, h
        );
        out.push('\n');
        // No background rect: the logo renders on a transparent canvas.
        out.push_str(&self.body);
        out.push_str("</svg>\n");
        out
    }
}

// ---------------------------------------------------------------------------
// small color helper
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Rgb(u8, u8, u8);

impl Rgb {
    /// Blend toward `other` by `t ∈ [0, 1]`, interpolating in **Oklab**
    /// (perceptual) rather than raw sRGB, so midpoints stay even instead of
    /// going dark/muddy. Every color blend in the logo (terrain ramp, water
    /// glint, sun halo) goes through here, like R's `colorRamp(space="Lab")`.
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

/// Parse an HTML hex color (`"#52796f"` or `"52796f"`) into an [`Rgb`], so
/// palettes can be written with copy-pasted hex codes. A `const fn`, so it
/// works inside the `const` palette tables. Expects exactly 6 hex digits
/// (optionally `#`-prefixed); non-hex digits read as 0.
const fn hex(s: &str) -> Rgb {
    let b = s.as_bytes();
    let o = b.len() - 6; // 1 when a leading '#' is present, else 0
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
