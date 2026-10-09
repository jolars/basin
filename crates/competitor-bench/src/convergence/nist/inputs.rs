// Source snapshots retain the published decimal precision.
use super::Source;

pub(super) const SOURCES: &[Source] = &[
    Source {
        id: "Bennett5",
        family: "nist-power",
        partition: "development",
        parameters: 3,
        observations: 154,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Bennett5.dat"
        ),
    },
    Source {
        id: "BoxBOD",
        family: "nist-misra",
        partition: "development",
        parameters: 2,
        observations: 6,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/BoxBOD.dat"
        ),
    },
    Source {
        id: "Chwirut1",
        family: "nist-chwirut",
        partition: "validation",
        parameters: 3,
        observations: 214,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Chwirut1.dat"
        ),
    },
    Source {
        id: "Chwirut2",
        family: "nist-chwirut",
        partition: "validation",
        parameters: 3,
        observations: 54,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Chwirut2.dat"
        ),
    },
    Source {
        id: "DanWood",
        family: "nist-power",
        partition: "development",
        parameters: 2,
        observations: 6,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/DanWood.dat"
        ),
    },
    Source {
        id: "ENSO",
        family: "nist-periodic",
        partition: "validation",
        parameters: 9,
        observations: 168,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/ENSO.dat"
        ),
    },
    Source {
        id: "Eckerle4",
        family: "nist-gaussian",
        partition: "validation",
        parameters: 3,
        observations: 35,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Eckerle4.dat"
        ),
    },
    Source {
        id: "Gauss1",
        family: "nist-gaussian",
        partition: "validation",
        parameters: 8,
        observations: 250,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Gauss1.dat"
        ),
    },
    Source {
        id: "Gauss2",
        family: "nist-gaussian",
        partition: "validation",
        parameters: 8,
        observations: 250,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Gauss2.dat"
        ),
    },
    Source {
        id: "Gauss3",
        family: "nist-gaussian",
        partition: "validation",
        parameters: 8,
        observations: 250,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Gauss3.dat"
        ),
    },
    Source {
        id: "Hahn1",
        family: "nist-rational",
        partition: "validation",
        parameters: 7,
        observations: 236,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Hahn1.dat"
        ),
    },
    Source {
        id: "Kirby2",
        family: "nist-rational",
        partition: "validation",
        parameters: 5,
        observations: 151,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Kirby2.dat"
        ),
    },
    Source {
        id: "Lanczos1",
        family: "nist-exponential-mixture",
        partition: "development",
        parameters: 6,
        observations: 24,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Lanczos1.dat"
        ),
    },
    Source {
        id: "Lanczos2",
        family: "nist-exponential-mixture",
        partition: "development",
        parameters: 6,
        observations: 24,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Lanczos2.dat"
        ),
    },
    Source {
        id: "Lanczos3",
        family: "nist-exponential-mixture",
        partition: "development",
        parameters: 6,
        observations: 24,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Lanczos3.dat"
        ),
    },
    Source {
        id: "MGH09",
        family: "nist-rational",
        partition: "validation",
        parameters: 4,
        observations: 11,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/MGH09.dat"
        ),
    },
    Source {
        id: "MGH10",
        family: "nist-mgh10",
        partition: "development",
        parameters: 3,
        observations: 16,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/MGH10.dat"
        ),
    },
    Source {
        id: "MGH17",
        family: "nist-exponential-mixture",
        partition: "development",
        parameters: 5,
        observations: 33,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/MGH17.dat"
        ),
    },
    Source {
        id: "Misra1a",
        family: "nist-misra",
        partition: "development",
        parameters: 2,
        observations: 14,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Misra1a.dat"
        ),
    },
    Source {
        id: "Misra1b",
        family: "nist-misra",
        partition: "development",
        parameters: 2,
        observations: 14,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Misra1b.dat"
        ),
    },
    Source {
        id: "Misra1c",
        family: "nist-misra",
        partition: "development",
        parameters: 2,
        observations: 14,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Misra1c.dat"
        ),
    },
    Source {
        id: "Misra1d",
        family: "nist-misra",
        partition: "development",
        parameters: 2,
        observations: 14,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Misra1d.dat"
        ),
    },
    Source {
        id: "Nelson",
        family: "nist-nelson",
        partition: "development",
        parameters: 3,
        observations: 128,
        predictors: 2,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Nelson.dat"
        ),
    },
    Source {
        id: "Rat42",
        family: "nist-logistic",
        partition: "development",
        parameters: 3,
        observations: 9,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Rat42.dat"
        ),
    },
    Source {
        id: "Rat43",
        family: "nist-logistic",
        partition: "development",
        parameters: 4,
        observations: 15,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Rat43.dat"
        ),
    },
    Source {
        id: "Roszman1",
        family: "nist-roszman",
        partition: "validation",
        parameters: 4,
        observations: 25,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Roszman1.dat"
        ),
    },
    Source {
        id: "Thurber",
        family: "nist-rational",
        partition: "validation",
        parameters: 7,
        observations: 37,
        predictors: 1,
        text: include_str!(
            "../../../../../dev/convergence-defaults/nist/Thurber.dat"
        ),
    },
];
