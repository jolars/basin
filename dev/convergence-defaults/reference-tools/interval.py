"""Outward-rounded Decimal intervals and second-order forward differentiation.

Only the elementary functions used by development NIST models are supported.
Decimal exp, ln, and sqrt are correctly rounded to nearest; enclosing them
requires adjacent representable Decimal endpoints, even in a directed context.
"""

from decimal import Context, Decimal as D, ROUND_CEILING, ROUND_FLOOR

PRECISION = 100
DOWN = Context(prec=PRECISION, rounding=ROUND_FLOOR)
UP = Context(prec=PRECISION, rounding=ROUND_CEILING)
NEAR = Context(prec=PRECISION)


class Interval:
    def __init__(self, lo, hi=None):
        self.lo = D(lo)
        self.hi = self.lo if hi is None else D(hi)
        if not (self.lo.is_finite() and self.hi.is_finite() and self.lo <= self.hi):
            raise ValueError('invalid interval')

    @staticmethod
    def of(value):
        return value if isinstance(value, Interval) else Interval(value)

    def __add__(self, other):
        if isinstance(other, Jet):
            return NotImplemented
        other = self.of(other)
        return Interval(DOWN.add(self.lo, other.lo), UP.add(self.hi, other.hi))

    __radd__ = __add__

    def __neg__(self):
        return Interval(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other):
        if isinstance(other, Jet):
            return NotImplemented
        return self + -self.of(other)

    def __rsub__(self, other):
        return self.of(other) - self

    def __mul__(self, other):
        if isinstance(other, Jet):
            return NotImplemented
        other = self.of(other)
        pairs = [(a, b) for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return Interval(min(DOWN.multiply(a, b) for a, b in pairs),
                        max(UP.multiply(a, b) for a, b in pairs))

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lo <= 0 <= self.hi:
            raise ValueError('division by an interval containing zero')
        return Interval(DOWN.divide(D(1), self.hi), UP.divide(D(1), self.lo))

    def __truediv__(self, other):
        if isinstance(other, Jet):
            return NotImplemented
        return self * self.of(other).reciprocal()

    def __rtruediv__(self, other):
        return self.of(other) / self

    def __pow__(self, n):
        if not isinstance(n, int):
            raise ValueError('only integer interval powers are supported')
        if n < 0:
            return (self ** -n).reciprocal()
        if n == 0:
            return Interval(1)
        if n == 1:
            return self
        if n == 2:
            lower = D(0) if self.lo <= 0 <= self.hi else min(
                DOWN.multiply(self.lo, self.lo), DOWN.multiply(self.hi, self.hi))
            return Interval(lower, max(UP.multiply(self.lo, self.lo), UP.multiply(self.hi, self.hi)))
        return self ** (n // 2) * self ** (n - n // 2)

    def elementary(self, name):
        if name == 'ln' and self.lo <= 0:
            raise ValueError('nonpositive logarithm domain')
        if name == 'sqrt' and self.lo < 0:
            raise ValueError('negative square root domain')
        low = getattr(NEAR, name)(self.lo)
        high = getattr(NEAR, name)(self.hi)
        return Interval(NEAR.next_minus(low), NEAR.next_plus(high))

    def exp(self):
        return self.elementary('exp')

    def ln(self):
        return self.elementary('ln')

    def sqrt(self):
        return self.elementary('sqrt')

    def magnitude(self):
        return max(self.lo.copy_abs(), self.hi.copy_abs())

    def midpoint(self):
        return NEAR.divide(NEAR.add(self.lo, self.hi), D(2))

    def width(self):
        return UP.subtract(self.hi, self.lo)

    def json(self):
        return [str(self.lo), str(self.hi)]


class Jet:
    """Value, gradient, and full Hessian over Decimal or Interval scalars."""

    def __init__(self, value, gradient, hessian):
        self.v, self.g, self.h = value, gradient, hessian

    @classmethod
    def constant(cls, value, n):
        zero = value * 0
        return cls(value, [zero] * n, [[zero] * n for _ in range(n)])

    @classmethod
    def variable(cls, value, n, j):
        result = cls.constant(value, n)
        result.g[j] = value * 0 + 1
        return result

    def of(self, other):
        if isinstance(other, Jet):
            return other
        value = Interval.of(other) if isinstance(self.v, Interval) else D(other)
        return self.constant(value, len(self.g))

    def __add__(self, other):
        other = self.of(other)
        n = len(self.g)
        return Jet(self.v + other.v, [a + b for a, b in zip(self.g, other.g)],
                   [[self.h[i][j] + other.h[i][j] for j in range(n)] for i in range(n)])

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -self.of(other)

    def __rsub__(self, other):
        return self.of(other) - self

    def __mul__(self, other):
        other = self.of(other)
        n = len(self.g)
        return Jet(self.v * other.v,
                   [self.g[i] * other.v + self.v * other.g[i] for i in range(n)],
                   [[self.h[i][j] * other.v + self.g[i] * other.g[j]
                     + self.g[j] * other.g[i] + self.v * other.h[i][j]
                     for j in range(n)] for i in range(n)])

    __rmul__ = __mul__

    def unary(self, value, first, second):
        n = len(self.g)
        return Jet(value, [first * g for g in self.g],
                   [[first * self.h[i][j] + second * self.g[i] * self.g[j]
                     for j in range(n)] for i in range(n)])

    def reciprocal(self):
        return self.unary(1 / self.v, -1 / self.v ** 2, 2 / self.v ** 3)

    def __truediv__(self, other):
        return self * self.of(other).reciprocal()

    def __rtruediv__(self, other):
        return self.of(other) / self

    def __pow__(self, n):
        if n == 0:
            return self.of(1)
        if n == 1:
            return self
        return self.unary(self.v ** n, n * self.v ** (n - 1), n * (n - 1) * self.v ** (n - 2))

    def exp(self):
        value = self.v.exp()
        return self.unary(value, value, value)

    def ln(self):
        return self.unary(self.v.ln(), 1 / self.v, -1 / self.v ** 2)

    def sqrt(self):
        value = self.v.sqrt()
        return self.unary(value, 1 / (2 * value), -1 / (4 * value ** 3))
