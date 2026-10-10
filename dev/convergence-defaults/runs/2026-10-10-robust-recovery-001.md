# Rejected global robust damping correction

The [manifest](2026-10-10-robust-recovery-001.toml) and [retained
report](../robust-recovery-rejected-results.json) preserve this failed recovery
attempt. Source `1f63614` and planned manifest `97ea26e` preceded 720 extended
and 336 original ablations. The independent accounting and stopping checks pass,
but the paired recovery gate rejects the correction.

Reducing Nielsen's damping by the smallest old/new diagonal ratio recovers the
four original rank-deficient `f32` stalls. It also introduces four inaccurate
full-rank arctangent default returns, under both factorizations and precisions.
Each gradient configuration therefore still passes only 68 of 72 extended
quality checks. In `f64`, escaping parameters approach `9.75e13`; in `f32`, they
exceed `1.11e4`. Their robust gradients become small in the loss's flat tails
despite large parameter and objective errors. Every returned-point check retains
the original parameter and gradient thresholds.

The shared damping reduction removes effective damping from coordinates whose
curvature remains clipped. An improving total objective can accept this escape
while other coordinates approach their minima. The failed attempt therefore
requires a more conservative compensation for growth shared by every coordinate.
[The follow-up](2026-10-10-robust-recovery-002.md) reruns the same frozen
controls and quality gate with that correction. No convergence threshold or
progress-policy default is selected here.
