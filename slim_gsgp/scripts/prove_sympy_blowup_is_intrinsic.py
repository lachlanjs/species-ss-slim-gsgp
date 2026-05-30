"""prove_sympy_blowup_is_intrinsic.py

Demonstrate that the blow-up is INTRINSIC to Sympy's rational simplification,
not a bug in our conversion code. This script uses ONLY pure sympy — no
slim_gsgp imports at all.

Experiment 1: sum of N fractions with distinct denominators. We call
sympy.cancel (which normalizes to a single p/q) and measure:
    - time
    - number of terms in the EXPANDED common denominator q
The denominator term count grows exponentially in N — that is the explosion.

Experiment 2: the same expression but with '/' replaced by '*' (no
divisions). Same N, but cancel/simplify stay tiny and instant. This isolates
divisions as the cause.
"""
from __future__ import annotations

import time
import signal

import sympy as sp


class _TO(Exception):
    pass


def _timed(fn, seconds):
    def _h(s, f):
        raise _TO()
    old = signal.signal(signal.SIGALRM, _h)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    t0 = time.time()
    try:
        return fn(), time.time() - t0, False
    except _TO:
        return None, time.time() - t0, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def experiment_with_divisions():
    print("=" * 70)
    print("EXPERIMENT 1: sum of N fractions  ->  sympy.cancel")
    print("  expr_N = sum_i  x_i / (x_{i+1} + 1)")
    print("=" * 70)
    print(f"{'N (fractions)':>14} {'cancel time':>14} {'terms in denom':>16}")
    print("-" * 46)
    for n in range(2, 13):
        xs = sp.symbols(f"x0:{n + 1}")
        expr = sum(xs[i] / (xs[i + 1] + sp.Integer(1)) for i in range(n))
        res, dt, to = _timed(lambda e=expr: sp.cancel(e), 5.0)
        if to:
            print(f"{n:>14} {'TIMEOUT(>5s)':>14} {'(exploded)':>16}")
            continue
        _, den = sp.fraction(res)
        den_terms = len(sp.Add.make_args(sp.expand(den)))
        print(f"{n:>14} {dt * 1000:>11.1f}ms {den_terms:>16}")


def experiment_without_divisions():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: same shape but '/' replaced by '*'  (no divisions)")
    print("  expr_N = sum_i  x_i * (x_{i+1} + 1)")
    print("=" * 70)
    print(f"{'N (terms)':>14} {'cancel time':>14} {'simplify time':>16}")
    print("-" * 46)
    for n in range(2, 13):
        xs = sp.symbols(f"x0:{n + 1}")
        expr = sum(xs[i] * (xs[i + 1] + sp.Integer(1)) for i in range(n))
        _, dt_c, to_c = _timed(lambda e=expr: sp.cancel(e), 5.0)
        _, dt_s, to_s = _timed(lambda e=expr: sp.simplify(e), 5.0)
        fc = "TIMEOUT" if to_c else f"{dt_c * 1000:8.1f}ms"
        fs = "TIMEOUT" if to_s else f"{dt_s * 1000:8.1f}ms"
        print(f"{n:>14} {fc:>14} {fs:>16}")


def main():
    print("Pure Sympy (version %s). No slim_gsgp code involved.\n" % sp.__version__)
    experiment_with_divisions()
    experiment_without_divisions()
    print("""
CONCLUSION
----------
Experiment 1: with divisions, sympy.cancel must form a common denominator =
product of all (x_i + 1). Expanding that product yields ~2^N terms, so both
the denominator size and the time explode exponentially. This is the standard,
documented behaviour of rational normalization — not a bug.

Experiment 2: remove the divisions and the very same number of terms is
simplified instantly. Divisions are the sole cause.

Therefore the hang is intrinsic to Sympy's algorithms; our code merely fed it
a division-heavy expression. The correct engineering responses are: bound the
call (timeout) and/or avoid asking Sympy to combine denominators (opaque
division / per-block).
""")


if __name__ == "__main__":
    main()
