use crate::core::math::{MatrixIdentity, MatrixIndex, Scalar, VectorLen};

/// Constructs BFGS's inverse Hessian at the start of each fresh solve.
///
/// [`Bfgs`](super::Bfgs) calls [`initialize`](Self::initialize) once, after
/// evaluating the starting point's cost and gradient, including on point warm
/// starts. Exact continuation skips this call and retains the current model
/// and the first-update rescaling flag.
///
/// Implementations must return a finite, symmetric positive-definite `n × n`
/// matrix, where `n` is the parameter dimension. They must construct a fresh
/// model on each call rather than reuse an evolving approximation. An empty
/// `0 × 0` matrix is valid for an empty parameter vector.
///
/// Closures and functions `Fn(&V, &V) -> M` implement this trait when
/// `V: VectorLen` and `M: MatrixIndex<F>`. They receive the point and its
/// gradient, in that order. Their returned matrices are checked for shape,
/// finite entries, exact symmetry, and positive definiteness using a pure-Rust
/// Cholesky factorization. Invalid matrices panic during initialization.
/// A named implementation can support custom storage without entry access,
/// and is responsible for enforcing the matrix contract itself.
///
/// With `serde`, a named implementation may derive serialization. Closures
/// support owned checkpoints without serialization or cloning.
pub trait InitialInverseHessian<V, M, F: Scalar = f64> {
    /// Construct a fresh model from the evaluated starting point and gradient.
    fn initialize(&self, param: &V, gradient: &V) -> M;

    /// Whether to replace the model with `(sᵀy / yᵀy) I` immediately before
    /// the first accepted curvature update. Defaults to `false` to preserve
    /// custom scaling. The default identity initializer returns `true`.
    fn rescale_on_first_update(&self) -> bool {
        false
    }
}

/// BFGS's default identity seed with rescaling on the first curvature update.
///
/// Requires only the matrix's existing [`MatrixIdentity`] capability.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IdentityInverseHessian;

impl<V: VectorLen, M: MatrixIdentity, F: Scalar> InitialInverseHessian<V, M, F>
    for IdentityInverseHessian
{
    fn initialize(&self, param: &V, _: &V) -> M {
        M::identity(param.vec_len())
    }

    fn rescale_on_first_update(&self) -> bool {
        true
    }
}

/// A fixed seed configured by [`Bfgs::with_initial_inverse_hessian`](super::Bfgs::with_initial_inverse_hessian).
///
/// Each fresh solve checks the seed and clones it into the working model.
/// BFGS updates never modify the stored seed. Requires `M: Clone +
/// MatrixIndex<F>` only when this initializer is selected.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FixedInverseHessian<M>(pub(super) M);

impl<V: VectorLen, M: Clone + MatrixIndex<F>, F: Scalar>
    InitialInverseHessian<V, M, F> for FixedInverseHessian<M>
{
    fn initialize(&self, param: &V, _: &V) -> M {
        validate::<M, F>(&self.0, param.vec_len());
        self.0.clone()
    }
}

impl<V: VectorLen, M: MatrixIndex<F>, F: Scalar, Init: Fn(&V, &V) -> M>
    InitialInverseHessian<V, M, F> for Init
{
    fn initialize(&self, param: &V, gradient: &V) -> M {
        let matrix = self(param, gradient);
        validate::<M, F>(&matrix, param.vec_len());
        matrix
    }
}

fn validate<M: MatrixIndex<F>, F: Scalar>(matrix: &M, n: usize) {
    assert!(
        matrix.matrix_rows() == n && matrix.matrix_cols() == n,
        "BFGS initial inverse Hessian must match the parameter dimension"
    );
    // Check both triangles because matvec uses the whole matrix, unlike a
    // factorization that can treat just one triangle as authoritative.
    for i in 0..n {
        for j in 0..=i {
            let a = matrix.matrix_entry(i, j);
            let b = matrix.matrix_entry(j, i);
            assert!(
                a.is_finite() && b.is_finite() && a == b,
                "BFGS initial inverse Hessian must be finite and symmetric"
            );
        }
    }

    // Validate custom seeds without requiring backend factorization support.
    let mut l = vec![F::zero(); n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix.matrix_entry(i, j);
            for k in 0..j {
                value = value - l[i * n + k] * l[j * n + k];
            }
            assert!(
                value.is_finite() && (i != j || value > F::zero()),
                "BFGS initial inverse Hessian must be positive definite"
            );
            l[i * n + j] = if i == j {
                value.sqrt()
            } else {
                value / l[j * n + j]
            };
            assert!(
                l[i * n + j].is_finite(),
                "BFGS initial inverse Hessian factorization must be finite"
            );
        }
    }
}
