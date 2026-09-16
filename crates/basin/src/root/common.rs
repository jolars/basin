use crate::core::math::Scalar;

use super::{RootError, RootResult, RootTerminationReason};

pub(super) fn num<F: Scalar>(x: f64) -> F {
    F::from_f64(x).unwrap()
}

pub(super) fn same_sign<F: Scalar>(a: F, b: F) -> bool {
    (a > F::zero() && b > F::zero()) || (a < F::zero() && b < F::zero())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Settings<F> {
    pub lower: F,
    pub upper: F,
    pub absolute: F,
    pub relative: F,
    pub max_iter: u64,
    pub guess: Option<F>,
}

impl<F: Scalar> Settings<F> {
    pub fn new(lower: F, upper: F) -> Self {
        Self {
            lower,
            upper,
            absolute: num(1e-12),
            relative: num::<F>(4.0) * F::epsilon(),
            max_iter: 100,
            guess: None,
        }
    }

    pub fn validate<E>(&self) -> Result<(), RootError<E, F>> {
        if !self.lower.is_finite()
            || !self.upper.is_finite()
            || self.lower >= self.upper
            || !(self.upper - self.lower).is_finite()
        {
            return Err(RootError::InvalidInterval {
                lower: self.lower,
                upper: self.upper,
            });
        }
        if let Some(x) = self
            .guess
            .filter(|&x| !x.is_finite() || x <= self.lower || x >= self.upper)
        {
            return Err(RootError::InvalidInitialGuess { x });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Point<F> {
    pub x: F,
    pub value: F,
    pub derivatives: Option<(F, Option<F>)>,
}

impl<F: Scalar> Point<F> {
    fn new<E>(x: F, value: F) -> Result<Self, RootError<E, F>> {
        if !value.is_finite() {
            return Err(RootError::NonFiniteValue { x, value });
        }
        Ok(Self {
            x,
            value,
            derivatives: None,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Counts {
    pub function: u64,
    pub derivative: u64,
    pub second_derivative: u64,
    pub callback: u64,
}

pub(super) trait Evaluator<F: Scalar> {
    type Error;
    fn evaluate(
        &mut self,
        x: F,
        counts: &mut Counts,
    ) -> Result<Point<F>, RootError<Self::Error, F>>;
    fn derivatives(
        &mut self,
        _point: &mut Point<F>,
        _counts: &mut Counts,
    ) -> Result<(), RootError<Self::Error, F>> {
        Ok(())
    }
}

pub(super) struct ValueOnly<C>(pub C);

impl<F: Scalar, E, C: FnMut(F) -> Result<F, E>> Evaluator<F> for ValueOnly<C> {
    type Error = E;
    fn evaluate(
        &mut self,
        x: F,
        counts: &mut Counts,
    ) -> Result<Point<F>, RootError<E, F>> {
        counts.function += 1;
        counts.callback += 1;
        Point::new(x, (self.0)(x).map_err(RootError::Evaluation)?)
    }
}

pub(super) struct Separate<C, D, DD> {
    pub function: C,
    pub derivative: D,
    pub second: Option<DD>,
}

impl<F: Scalar, E, C, D, DD> Evaluator<F> for Separate<C, D, DD>
where
    C: FnMut(F) -> Result<F, E>,
    D: FnMut(F) -> Result<F, E>,
    DD: FnMut(F) -> Result<F, E>,
{
    type Error = E;
    fn evaluate(
        &mut self,
        x: F,
        counts: &mut Counts,
    ) -> Result<Point<F>, RootError<E, F>> {
        ValueOnly(&mut self.function).evaluate(x, counts)
    }
    fn derivatives(
        &mut self,
        point: &mut Point<F>,
        counts: &mut Counts,
    ) -> Result<(), RootError<E, F>> {
        if point.derivatives.is_none() {
            counts.derivative += 1;
            counts.callback += 1;
            let d =
                (self.derivative)(point.x).map_err(RootError::Evaluation)?;
            let dd = if let Some(second) = &mut self.second {
                counts.second_derivative += 1;
                counts.callback += 1;
                Some(second(point.x).map_err(RootError::Evaluation)?)
            } else {
                None
            };
            point.derivatives = Some((d, dd));
        }
        Ok(())
    }
}

pub(super) struct Combined<C>(pub C);

impl<F: Scalar, E, C: FnMut(F) -> Result<(F, F, Option<F>), E>> Evaluator<F>
    for Combined<C>
{
    type Error = E;
    fn evaluate(
        &mut self,
        x: F,
        counts: &mut Counts,
    ) -> Result<Point<F>, RootError<E, F>> {
        counts.function += 1;
        counts.derivative += 1;
        counts.callback += 1;
        let (value, d, dd) = (self.0)(x).map_err(RootError::Evaluation)?;
        counts.second_derivative += u64::from(dd.is_some());
        let mut point = Point::new(x, value)?;
        point.derivatives = Some((d, dd));
        Ok(point)
    }
}

pub(super) struct Bracket<F> {
    pub a: Point<F>,
    pub b: Point<F>,
}

impl<F: Scalar> Bracket<F> {
    pub fn initialize<V: Evaluator<F>>(
        settings: &Settings<F>,
        eval: &mut V,
        counts: &mut Counts,
    ) -> Result<Self, RootError<V::Error, F>> {
        settings.validate()?;
        let a = eval.evaluate(settings.lower, counts)?;
        if a.value == F::zero() {
            return Ok(Self { a, b: a });
        }
        let b = eval.evaluate(settings.upper, counts)?;
        if b.value == F::zero() {
            return Ok(Self { a: b, b });
        }
        if same_sign(a.value, b.value) {
            return Err(RootError::NotBracketed {
                lower: a.x,
                upper: b.x,
                f_lower: a.value,
                f_upper: b.value,
            });
        }
        Ok(Self { a, b })
    }

    pub fn width(&self) -> F {
        self.b.x - self.a.x
    }
    pub fn midpoint(&self) -> F {
        self.a.x + num::<F>(0.5) * self.width()
    }
    pub fn contains(&self, x: F) -> bool {
        x.is_finite() && self.a.x < x && x < self.b.x
    }
    pub fn best(&self) -> Point<F> {
        if self.a.value.abs() <= self.b.value.abs() {
            self.a
        } else {
            self.b
        }
    }
    pub fn converged(&self, settings: &Settings<F>) -> bool {
        self.best().value == F::zero()
            || self.width()
                <= settings.absolute + settings.relative * self.best().x.abs()
    }

    pub fn update(&mut self, point: Point<F>) -> Point<F> {
        if point.value == F::zero() {
            let discarded = self.a;
            self.a = point;
            self.b = point;
            discarded
        } else if same_sign(point.value, self.a.value) {
            std::mem::replace(&mut self.a, point)
        } else {
            std::mem::replace(&mut self.b, point)
        }
    }

    pub fn result(
        &self,
        settings: &Settings<F>,
        iterations: u64,
        counts: Counts,
    ) -> RootResult<F> {
        let best = self.best();
        let reason = if self.converged(settings) {
            RootTerminationReason::Converged
        } else {
            RootTerminationReason::MaxIter
        };
        RootResult::new(
            best.x,
            best.value,
            (self.a.x, self.b.x),
            (iterations, counts.function),
            reason,
        )
        .with_counts(counts)
    }
}

pub(super) fn secant<F: Scalar>(a: Point<F>, b: Point<F>) -> F {
    // Ratios keep the calculation usable when the signed values span most
    // of the exponent range. Subtract from the endpoint nearer the zero.
    if a.value.abs() <= b.value.abs() {
        let ratio = a.value / b.value;
        a.x - (b.x - a.x) * (ratio / (F::one() - ratio))
    } else {
        let ratio = b.value / a.value;
        b.x - (a.x - b.x) * (ratio / (F::one() - ratio))
    }
}

pub(super) fn hybrid<F: Scalar, V: Evaluator<F>>(
    settings: &Settings<F>,
    mut eval: V,
    use_derivatives: bool,
) -> Result<RootResult<F>, RootError<V::Error, F>> {
    let mut counts = Counts::default();
    let mut bracket = Bracket::initialize(settings, &mut eval, &mut counts)?;
    let mut current = bracket.b;
    let mut previous = bracket.a;
    let mut checkpoint_width = bracket.width();
    let mut steps = 0;
    for iteration in 0..settings.max_iter {
        if bracket.converged(settings) {
            return Ok(bracket.result(settings, iteration, counts));
        }
        let force_bisection =
            steps == 2 && bracket.width() > num::<F>(0.5) * checkpoint_width;
        if steps == 2 {
            checkpoint_width = bracket.width();
            steps = 0;
        }
        let mut candidate = secant(previous, current);
        if use_derivatives && iteration == 0 {
            candidate = settings.guess.unwrap_or_else(|| bracket.midpoint());
        } else if use_derivatives && !force_bisection {
            eval.derivatives(&mut current, &mut counts)?;
            if let Some((d, dd)) = current
                .derivatives
                .filter(|(d, _)| d.is_finite() && *d != F::zero())
            {
                let delta = current.value / d;
                let newton = current.x - delta;
                if bracket.contains(newton) {
                    candidate = newton;
                }
                if let Some(dd) = dd {
                    let adjustment = (delta * num::<F>(0.5)) * (dd / d);
                    if adjustment.is_finite() && adjustment.abs() < F::one() {
                        let halley =
                            current.x - delta / (F::one() - adjustment);
                        if bracket.contains(halley) {
                            candidate = halley;
                        }
                    }
                }
            }
        }
        if force_bisection
            || !bracket.contains(candidate)
            || candidate == current.x
        {
            candidate = bracket.midpoint();
        }
        let point = eval.evaluate(candidate, &mut counts)?;
        bracket.update(point);
        previous = current;
        current = point;
        steps += 1;
    }
    Ok(bracket.result(settings, settings.max_iter, counts))
}

macro_rules! root_builders {
    () => {
        /// Set the finite, strictly positive absolute position tolerance.
        ///
        /// # Panics
        /// Panics if the value is non-finite or nonpositive.
        pub fn with_absolute_position_tolerance(mut self, value: F) -> Self {
            assert!(value.is_finite() && value > F::zero(), "absolute position tolerance must be finite and positive");
            self.settings.absolute = value;
            self
        }

        /// Set the finite relative position tolerance, at least `4ε_F`.
        ///
        /// # Panics
        /// Panics if the value is non-finite or less than `4ε_F`.
        pub fn with_relative_position_tolerance(mut self, value: F) -> Self {
            assert!(value.is_finite() && value >= F::from_f64(4.0).unwrap() * F::epsilon(), "relative position tolerance must be finite and at least four times machine epsilon");
            self.settings.relative = value;
            self
        }

        /// Set the iteration limit. Zero evaluates only the endpoints.
        ///
        /// See the solver's documentation for what constitutes an iteration.
        pub fn with_max_iter(mut self, value: u64) -> Self {
            self.settings.max_iter = value;
            self
        }
    };
}
pub(super) use root_builders;
