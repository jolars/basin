! Regenerate with Jacob Williams SLSQP 1.6.1. See README.md.
program driver
  use slsqp_kinds, only: wp
  use slsqp_core, only: slsqp, slsqpb_data, linmin_data
  implicit none
  integer, parameter :: n=4, m=2, la=2, lw=10000
  real(wp) :: x(n), xl(n), xu(n), f, c(la), g(n+1), a(la,n+1), w(lw), acc
  type(slsqpb_data) :: sd
  type(linmin_data) :: ld
  integer :: mode, iter, k
  x=[1.0_wp,5.0_wp,5.0_wp,1.0_wp]
  xl=1.0_wp; xu=5.0_wp; w=0.0_wp; acc=1.0e-10_wp
  mode=0; iter=100; k=0
  call evaluate
  write(*,'(i3,5es25.16)') k,f,x
  do
    call slsqp(m,1,la,n,x,xl,xu,f,c,g,a,acc,iter,mode,w,lw,sd,ld, &
               0.1_wp,1.0_wp,-1.0_wp,-1.0_wp,-1.0_wp,0,1,0.0_wp)
    if (abs(mode)/=1) then
      if (mode==0) then
        k=k+1
        call evaluate
        write(*,'(i3,5es25.16)') k,f,x
      end if
      write(*,'(a,i0)') '# mode=',mode
      exit
    end if
    call evaluate
    if (mode==-1) then
      k=k+1
      write(*,'(i3,5es25.16)') k,f,x
    end if
  end do
contains
  subroutine evaluate
    integer :: j
    f=x(1)*x(4)*(x(1)+x(2)+x(3))+x(3)
    c(1)=sum(x*x)-40.0_wp
    c(2)=product(x)-25.0_wp
    g=0.0_wp; a=0.0_wp
    g(1)=x(4)*(2*x(1)+x(2)+x(3))
    g(2)=x(1)*x(4); g(3)=x(1)*x(4)+1
    g(4)=x(1)*(x(1)+x(2)+x(3))
    a(1,1:n)=2*x
    do j=1,n
      a(2,j)=product(x)/x(j)
    end do
  end subroutine
end program
