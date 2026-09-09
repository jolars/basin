! Investigation-only bindings. The pinned PRIMA sources remain unmodified.
module basin_prima_bridge
use, intrinsic :: iso_c_binding
implicit none
contains

subroutine basin_prima_solve(n, m, x, callback, data, budget, f, cv, nf, info) bind(C)
    use cobyla_mod, only : cobyla
    use consts_mod, only : RP
    use cintrf_mod, only : evalcobjcon
    integer(c_int), value :: n, m, budget
    real(c_double), intent(inout) :: x(n)
    type(c_funptr), value :: callback
    type(c_ptr), value :: data
    real(c_double), intent(out) :: f, cv
    integer(c_int), intent(out) :: nf, info

    ! Explicit schedule constants avoid the C API's derived ETA2 rounding.
    ! All constraints, including bounds, arrive through the nonlinear block.
    call cobyla(calcfc, m, x, f, cstrv=cv, nf=nf, info=info, &
        rhobeg=0.5_RP, rhoend=sqrt(epsilon(1.0_RP))*0.5_RP, &
        ctol=sqrt(epsilon(1.0_RP)), cweight=1.0e8_RP, &
        eta1=0.1_RP, eta2=0.7_RP, gamma1=0.5_RP, gamma2=2.0_RP, &
        ftarget=-huge(1.0_RP), maxfun=budget, maxfilt=2000, maxhist=0, iprint=0)
contains
    subroutine calcfc(point, cost, constraints)
        real(RP), intent(in) :: point(:)
        real(RP), intent(out) :: cost, constraints(:)
        call evalcobjcon(callback, data, point, cost, constraints)
    end subroutine
end subroutine

subroutine basin_prima_lp(n, m, a, b, delta, g, d) bind(C)
    use trustregion_cobyla_mod, only : trstlp
    integer(c_int), value :: n, m
    real(c_double), intent(in) :: a(n,m), b(m), g(n)
    real(c_double), value :: delta
    real(c_double), intent(out) :: d(n)
    d = trstlp(a, b, delta, g)
end subroutine
end module
