//! NIST StRD data and model formulas, preserved independently of solver code.

use super::dual::Real;

#[derive(Clone, Debug)]
pub struct Nist {
    pub name: &'static str,
    pub family: &'static str,
    pub validation: bool,
    pub starts: [Vec<f64>; 2],
    pub certified: Vec<f64>,
    pub parameter_rounding: Vec<f64>,
    pub rss: f64,
    /// Each row contains the response, then the predictor(s).
    pub data: Vec<Vec<f64>>,
}

impl Nist {
    fn parse(
        name: &'static str,
        family: &'static str,
        validation: bool,
        text: &str,
    ) -> Self {
        let mut starts = [vec![], vec![]];
        let mut certified = vec![];
        let mut parameter_rounding = vec![];
        for line in text.lines() {
            let words: Vec<_> = line.split_whitespace().collect();
            if words.len() >= 6 && words[0].starts_with('b') && words[1] == "="
            {
                starts[0].push(words[2].parse().unwrap());
                starts[1].push(words[3].parse().unwrap());
                certified.push(words[4].parse().unwrap());
                let (mantissa, exponent) = words[4].split_once('E').unwrap();
                let decimals = mantissa.split_once('.').unwrap().1.len() as i32;
                parameter_rounding.push(
                    0.5 * 10_f64
                        .powi(exponent.parse::<i32>().unwrap() - decimals),
                );
            }
        }
        let field = |prefix: &str| -> f64 {
            text.lines()
                .find_map(|line| line.trim().strip_prefix(prefix))
                .unwrap()
                .trim()
                .parse()
                .unwrap()
        };
        let data = text
            .rsplit_once("Data:")
            .unwrap()
            .1
            .lines()
            .skip(1)
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                line.split_whitespace()
                    .map(|x| x.parse().unwrap())
                    .collect()
            })
            .collect::<Vec<Vec<f64>>>();
        assert_eq!(
            data.len(),
            field("Number of Observations:") as usize,
            "{name}"
        );
        assert!(
            !certified.is_empty()
                && certified.len() <= super::dual::MAX_PARAMETERS
        );
        Self {
            name,
            family,
            validation,
            starts,
            certified,
            parameter_rounding,
            rss: field("Residual Sum of Squares:"),
            data,
        }
    }

    pub fn residuals<T: Real>(&self, p: &[T]) -> Vec<T> {
        self.data
            .iter()
            .map(|row| {
                let x = T::number(row[1]);
                let one = T::number(1.0);
                let two = T::number(2.0);
                let exponential_sum = || {
                    p[0] * (-p[1] * x).exp()
                        + p[2] * (-p[3] * x).exp()
                        + p[4] * (-p[5] * x).exp()
                };
                let prediction = match self.name {
                    "Misra1a" | "BoxBOD" => p[0] * (one - (-p[1] * x).exp()),
                    "Chwirut1" | "Chwirut2" => {
                        (-p[0] * x).exp() / (p[1] + p[2] * x)
                    }
                    "Lanczos1" | "Lanczos2" | "Lanczos3" => exponential_sum(),
                    "Gauss1" | "Gauss2" | "Gauss3" => {
                        p[0] * (-p[1] * x).exp()
                            + p[2] * (-((x - p[3]) / p[4]).square()).exp()
                            + p[5] * (-((x - p[6]) / p[7]).square()).exp()
                    }
                    "DanWood" => p[0] * x.pow(p[1]),
                    "Misra1b" => {
                        p[0] * (one - one / (one + p[1] * x / two).square())
                    }
                    "Kirby2" => {
                        (p[0] + x * (p[1] + x * p[2]))
                            / (one + x * (p[3] + x * p[4]))
                    }
                    "Hahn1" | "Thurber" => {
                        (p[0] + x * (p[1] + x * (p[2] + x * p[3])))
                            / (one + x * (p[4] + x * (p[5] + x * p[6])))
                    }
                    "Nelson" => {
                        p[0] - p[1] * x * (-p[2] * T::number(row[2])).exp()
                    }
                    "MGH17" => {
                        p[0] + p[1] * (-x * p[3]).exp()
                            + p[2] * (-x * p[4]).exp()
                    }
                    "Misra1c" => {
                        p[0] * (one
                            - (one + two * p[1] * x).pow(T::number(-0.5)))
                    }
                    "Misra1d" => p[0] * p[1] * x / (one + p[1] * x),
                    "Roszman1" => {
                        p[0] - p[1] * x
                            - (p[2] / (x - p[3])).atan()
                                / T::number(std::f64::consts::PI)
                    }
                    "ENSO" => {
                        let t = T::number(std::f64::consts::TAU) * x;
                        p[0] + p[1] * (t / T::number(12.0)).cos()
                            + p[2] * (t / T::number(12.0)).sin()
                            + p[4] * (t / p[3]).cos()
                            + p[5] * (t / p[3]).sin()
                            + p[7] * (t / p[6]).cos()
                            + p[8] * (t / p[6]).sin()
                    }
                    "MGH09" => {
                        p[0] * (x.square() + x * p[1])
                            / (x.square() + x * p[2] + p[3])
                    }
                    "Rat42" => p[0] / (one + (p[1] - p[2] * x).exp()),
                    "MGH10" => p[0] * (p[1] / (x + p[2])).exp(),
                    "Eckerle4" => {
                        p[0] / p[1]
                            * (T::number(-0.5) * ((x - p[2]) / p[1]).square())
                                .exp()
                    }
                    "Rat43" => {
                        p[0] / (one + (p[1] - p[2] * x).exp()).pow(one / p[3])
                    }
                    "Bennett5" => p[0] * (p[1] + x).pow(-one / p[2]),
                    _ => unreachable!("unregistered NIST model"),
                };
                // NIST fits log(y), not y, for Nelson.
                let response = if self.name == "Nelson" {
                    T::number(row[0]).ln()
                } else {
                    T::number(row[0])
                };
                prediction - response
            })
            .collect()
    }
}

pub fn cases() -> Vec<Nist> {
    macro_rules! case {
        ($name:literal, $family:literal, $validation:literal) => {
            Nist::parse(
                $name,
                $family,
                $validation,
                include_str!(concat!("../../data/nist/", $name, ".dat")),
            )
        };
    }
    vec![
        case!("Misra1a", "saturation", false),
        case!("Chwirut2", "exponential_ratio", false),
        case!("Chwirut1", "exponential_ratio", false),
        case!("Lanczos3", "exponential_sum", false),
        case!("Gauss1", "gaussian", true),
        case!("Gauss2", "gaussian", true),
        case!("DanWood", "power", false),
        case!("Misra1b", "saturation", false),
        case!("Kirby2", "rational", true),
        case!("Hahn1", "rational", true),
        case!("Nelson", "log_response", true),
        case!("MGH17", "exponential_sum", false),
        case!("Lanczos1", "exponential_sum", false),
        case!("Lanczos2", "exponential_sum", false),
        case!("Gauss3", "gaussian", true),
        case!("Misra1c", "saturation", false),
        case!("Misra1d", "saturation", false),
        case!("Roszman1", "arctangent", false),
        case!("ENSO", "harmonic", true),
        case!("MGH09", "rational", true),
        case!("Thurber", "rational", true),
        case!("BoxBOD", "saturation", false),
        case!("Rat42", "logistic", false),
        case!("MGH10", "reciprocal_exponential", true),
        case!("Eckerle4", "gaussian", true),
        case!("Rat43", "logistic", false),
        case!("Bennett5", "power", false),
    ]
}
