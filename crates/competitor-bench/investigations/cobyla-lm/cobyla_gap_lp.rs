// Appended to the isolated copy of cobyla 1.0.2's private nlopt_cobyla module.
// Prefix padding keeps the translated routine's one-based pointer shifts inside
// their allocations. No upstream numerical statements are changed.
pub struct GapLp {
    n: usize,
    m: usize,
    a: Vec<f64>,
    b: Vec<f64>,
    d: Vec<f64>,
    iact: Vec<i32>,
    z: Vec<f64>,
    zdota: Vec<f64>,
    vmultc: Vec<f64>,
    sdirn: Vec<f64>,
    dxnew: Vec<f64>,
    vmultd: Vec<f64>,
}
impl GapLp {
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            n,
            m,
            a: vec![0.0; n * (m + 1) + n + 1],
            b: vec![0.0; m + 2],
            d: vec![0.0; n + 1],
            iact: vec![0; m + 2],
            z: vec![0.0; n * n + n + 1],
            zdota: vec![0.0; n + 1],
            vmultc: vec![0.0; m + 2],
            sdirn: vec![0.0; n + 1],
            dxnew: vec![0.0; n + 1],
            vmultd: vec![0.0; m + 2],
        }
    }
    pub fn solve(
        &mut self,
        a: &[f64],
        b: &[f64],
        delta: f64,
        g: &[f64],
    ) -> &[f64] {
        assert_eq!(a.len(), self.n * self.m);
        assert_eq!(b.len(), self.m);
        assert_eq!(g.len(), self.n);
        let off = self.n + 1;
        // PRIMA uses A'd <= b and min g'd; Powell's LP uses A'd >= b
        // and an objective column that points downhill.
        for (dst, src) in self.a[off..].iter_mut().zip(a.iter().chain(g)) {
            *dst = -*src;
        }
        for (dst, src) in self.b[1..].iter_mut().zip(b) {
            *dst = -*src;
        }
        self.b[self.m + 1] = 0.0;
        let (mut n, mut m, mut rho, mut full) =
            (self.n as i32, self.m as i32, delta, 0);
        // All scratch buffers include the translated routine's prefix and full
        // active-set capacity; the call is synchronous and retains no pointers.
        let rc = unsafe {
            trstlp(
                &mut n,
                &mut m,
                self.a.as_mut_ptr().add(off),
                self.b.as_mut_ptr().add(1),
                &mut rho,
                self.d.as_mut_ptr().add(1),
                &mut full,
                self.iact.as_mut_ptr().add(1),
                self.z.as_mut_ptr().add(off),
                self.zdota.as_mut_ptr().add(1),
                self.vmultc.as_mut_ptr().add(1),
                self.sdirn.as_mut_ptr().add(1),
                self.dxnew.as_mut_ptr().add(1),
                self.vmultd.as_mut_ptr().add(1),
            )
        };
        assert_eq!(rc, NLOPT_SUCCESS);
        &self.d[1..]
    }
}

#[cfg(feature = "capture")]
thread_local! {
    static GAP_COUNTS: std::cell::RefCell<std::collections::BTreeMap<&'static str, usize>> = const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
}
#[cfg(feature = "capture")]
pub fn gap_record(name: &'static str) {
    GAP_COUNTS.with(|w| *w.borrow_mut().entry(name).or_default() += 1);
}
#[cfg(feature = "capture")]
pub fn gap_counts() -> std::collections::BTreeMap<&'static str, usize> {
    GAP_COUNTS.with(|w| w.borrow().clone())
}
