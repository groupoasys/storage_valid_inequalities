"""
storage_cuts.py
===============

Separation, strengthening and verification of the submodular capacity cuts
(VIc_D) / (VId_D) for the operational polytope of an energy storage unit.

Everything is written for the *normalized* model of the paper
(Delta = eta_c = eta_d = 1): c_t and d_t are the energies stored in /
withdrawn from the unit in period t, cbar and dbar are the (effective)
per-period energy limits and [slo, sup] the SoC bounds.  Physical data are
converted once on entry (see ``Storage.from_physical``) and the cuts are
converted back on exit (``Cut.physical``).

Main objects
------------
Storage            instance data + reachable SoC bounds + singleton limits
Storage.evaluate   O(T) evaluation of a set Omega: greedy trajectory,
                   thresholds kappa_t, f(Omega), all gains pi_tau, D_Omega
                   and the first period at which (C1)/(C2) fail
Cut                a cut (Omega, D, pi, f, lifted coefficients) of the
                   charging family of some orientation of the unit; it
                   exports coefficients in the original (d, c, u) space
Point              an LP point (c*, d*, u*) in normalized units
separate           separation routine (Algorithm 1 of Section 3.6: closed-form
                   scan of the windows, dominance moves, local search as
                   escalation, memoization; discharging cuts by reflection)
select_cuts        most violated cut(s) per period with simultaneous charging and discharging
strengthen_pool    strengthening stage (Algorithm 2): lifting where (C3) fails
lift_coefficient   exact lifting coefficient alpha*_t of c_t, t notin Omega
                   (closed form over relaxation patterns, Prop. in Sec. 3.6)
max_over_P01       exact DP oracle: max of a linear function over the
                   binary set P^{0-1} (validity checks, C4 rotations)

Periods are 1-indexed everywhere (period 0 is the initial condition).
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

TOL = 1e-9          # structural tolerance (equalities of the model)


# ---------------------------------------------------------------------------
# 1. Instance
# ---------------------------------------------------------------------------
class Storage:
    """Normalized storage unit on the horizon [1, T].

    Attributes (1-indexed lists where applicable):
      T, slo, sup, s0, cb, db
      sig_lo[t], sig_hi[t] for t = 0..T : reachable SoC bounds (eq:sigma_bounds)
      cbar[t], dbar[t]     for t = 1..T : singleton limits (eq:singleton_cuts)
    """

    def __init__(self, T: int, slo: float, sup: float, s0: float, cb: float, db: float):
        if not (slo <= s0 <= sup) or slo >= sup or cb <= 0 or db <= 0:
            raise ValueError("inconsistent storage data")
        self.T = int(T)
        self.slo, self.sup, self.s0 = float(slo), float(sup), float(s0)
        self.cb = min(float(cb), sup - slo)         # effective limits (Sec. 2)
        self.db = min(float(db), sup - slo)
        self.sig_lo = [max(s0 - t * self.db, slo) for t in range(T + 1)]
        self.sig_hi = [min(s0 + t * self.cb, sup) for t in range(T + 1)]
        self.cbar = [None] + [min(self.cb, sup - self.sig_lo[t - 1]) for t in range(1, T + 1)]
        self.dbar = [None] + [min(self.db, self.sig_hi[t - 1] - slo) for t in range(1, T + 1)]

    # -- physical <-> normalized -------------------------------------------
    @classmethod
    def from_physical(cls, T, Emin, Emax, E0, PcMax, PdMax, eta_c, eta_d, dt=1.0):
        return cls(T, Emin, Emax, E0, dt * eta_c * PcMax, dt * PdMax / eta_d)

    def mirror(self) -> "Storage":
        """The mirrored unit of Appendix A: limits exchanged, SoC reflected."""
        return Storage(self.T, self.slo, self.sup, self.slo + self.sup - self.s0, self.db, self.cb)

    # -- elementary maps -----------------------------------------------------
    def phi_plus(self, s: float) -> float:
        return min(s + self.cb, self.sup)

    def phi_minus(self, s: float) -> float:
        return max(s - self.db, self.slo)

    # -- greedy rule with period-dependent box bounds -----------------------
    def rank(self, A: Set[int], lo: Dict[int, float], hi: Dict[int, float],
             first: int = 1, last: Optional[int] = None) -> float:
        """max x(A) over the storage set with bounds lo_t <= x_t <= hi_t
        (lo_t <= 0 <= hi_t), by the time-greedy rule (Lemma greedy_polymatroid
        with period-dependent bounds).  Periods before ``first`` are assumed
        to discharge at full power (no bound there), periods after ``last``
        are irrelevant."""
        if last is None:
            last = self.T
        s = self.sig_lo[first - 1]
        tot = 0.0
        slo, sup = self.slo, self.sup
        for t in range(first, last + 1):
            if t in A:
                xt = min(hi.get(t, self.cb), sup - s)
                tot += xt
            else:
                xt = max(lo.get(t, -self.db), slo - s)
            s += xt
        return tot

    def f(self, A: Iterable[int]) -> float:
        """f(A) on [T] from s0 (plain greedy)."""
        A = set(A)
        if not A:
            return 0.0
        return self.rank(A, {}, {}, first=min(A), last=max(A))

    # -- O(T) evaluation of a set -------------------------------------------
    def evaluate(self, Omega: Iterable[int]) -> "SetData":
        """Two-pass evaluation of Omega (Prop. two_passes), O(k-l+1).

        Forward pass: greedy trajectory from sig_lo[l-1]; checks (C2) at the
        gaps and (C1) at l-1 and at the gaps.  Backward pass: thresholds
        kappa_t of Lemma thresholds.  Then pi_tau from eq:pi_two_passes.
        """
        Om = tuple(sorted(set(Omega)))
        if not Om:
            raise ValueError("empty set")
        return self._evaluate(Om)

    def _evaluate(self, Om: Tuple[int, ...]) -> "SetData":
        """evaluate() for a set already given as a sorted tuple of distinct
        periods (no checks).  The arithmetic is the same, operation by
        operation, as in the reference implementation; only the bookkeeping
        differs (0-based lists over [l, k] instead of sets and dictionaries)."""
        l, k = Om[0], Om[-1]
        slo, sup, cb, db = self.slo, self.sup, self.cb, self.db
        cbar = self.cbar
        n = k - l + 1
        inw = [False] * n                       # inw[i]: period l+i belongs to Omega
        for t in Om:
            inw[t - l] = True
        # suffix sums of cbar over Omega: suf[i+1] = sum_{q in Omega, q > l+i} cbar_q, suf[0] = sum over Omega
        suf = [0.0] * (n + 1)
        acc = 0.0
        for i in range(n - 1, -1, -1):
            suf[i + 1] = acc
            if inw[i]:
                acc += cbar[l + i]
        suf[0] = acc
        s = self.sig_lo[l - 1]
        s_prev = [0.0] * n            # SoC at the end of t-1, index t-l
        s_end = [0.0] * n             # SoC at the end of t,   index t-l
        x = [0.0] * n
        fval = 0.0
        first_fail = (l - 1) if (sup - s >= suf[0] - TOL) else None
        first_kind = 'C1' if first_fail is not None else None
        for i in range(n):
            s_prev[i] = s
            if inw[i]:
                xt = cb if s + cb <= sup else sup - s
                fval += xt
                x[i] = xt
                s += xt
                s_end[i] = s
            else:
                xt = -db if s - db >= slo else slo - s
                x[i] = xt
                s += xt
                s_end[i] = s
                if first_fail is None:
                    if s <= slo + TOL:                      # (C2) fails at t
                        first_fail, first_kind = l + i, 'C2'
                    elif sup - s >= suf[i + 1] - TOL:       # (C1) fails at t
                        first_fail, first_kind = l + i, 'C1'
        kappa = [0.0] * (n + 1)       # kappa[t-l+1] = threshold at the end of t, t in [l-1,k]
        ka = sup
        kappa[n] = ka
        for i in range(n - 1, -1, -1):
            if inw[i]:
                ka = ka - cb
                if ka < slo:
                    ka = slo
            else:
                ka = ka + db
                if ka > sup:
                    ka = sup
            kappa[i] = ka
        pi: Dict[int, float] = {}
        dbar = self.dbar
        Dset = []
        for t in Om:
            i = t - l
            sp = s_prev[i]
            ka = kappa[i + 1]
            php = sp + cb
            if php > sup:
                php = sup
            phm = sp - db
            if phm < slo:
                phm = slo
            a = php - ka
            b = phm - ka
            p = (a if a > 0.0 else 0.0) - (b if b > 0.0 else 0.0) - x[i]
            pi[t] = p
            if abs(p - dbar[t]) <= 1e-7:
                Dset.append(t)
        return SetData(Om, l, k, fval, pi, frozenset(Dset), x, s_prev, s_end, kappa,
                       first_fail, first_kind, self.sig_lo[l - 1])


    def assess(self, Omega, pt: "Point", eps: float, canonical: bool = False) -> "Assessment":
        """evaluate(Omega) plus, in the same pass, the quantities that the
        separation needs at the point pt: robust and actual violation with
        their pointwise-optimal D (Section choice_of_D) and the free periods
        of Lemma free_removal.

        Ties.  A period t with pi_t = dbar_t (t in D_Omega) is always placed
        in D.  By the singleton cut, d*_t <= dbar_t (1 - u*_t) <= dbar_t
        (1 - c*_t / cbar_t), so the continuous term never gives a larger
        credit than the binary one; when both credits coincide (singleton
        cut tight at pt) the violation is the same, but the cut with t in D
        dominates the other one, which is the sum of it and of the singleton
        cut.  With this rule every cut returned satisfies
        D_Omega <= D <= {t : pi_t > 0}, the range of Theorem th:facets."""
        ev = self._evaluate(Omega) if canonical else self.evaluate(Omega)   # canonical: sorted tuple of distinct periods
        c, d, u = pt.c, pt.d, pt.u
        cbar, dbar = self.cbar, self.dbar
        pi = ev.pi
        v_rob = v_act = -ev.f
        D_rob, D_act, free = [], [], []
        for t in ev.Omega:
            p = pi[t]
            ct, dt_, ut = c[t], d[t], u[t]
            v_rob += ct
            v_act += ct
            if p > TOL:
                in_DOm = t in ev.D          # pi_t = dbar_t: ties are resolved in favor of D
                cr = p * (1.0 - ct / cbar[t])
                if in_DOm or dt_ < cr - TOL:
                    v_rob -= dt_; D_rob.append(t)
                else:
                    v_rob -= cr
                ca = p * (1.0 - ut)
                if in_DOm or dt_ < ca - TOL:
                    v_act -= dt_; D_act.append(t)
                else:
                    v_act -= ca
                if ct <= eps and ut <= eps and dt_ >= p - TOL:
                    free.append(t)
            else:
                v_rob -= p * dt_ / dbar[t]
                v_act -= p * (1.0 - ut)
                if ct <= eps:
                    free.append(t)
        return Assessment(ev, v_rob, v_act, frozenset(D_rob), frozenset(D_act), free)


class SetData:
    """Output of Storage.evaluate.

    Omega, l, k, f, pi (gains), D (= D_Omega), fail_at / fail_kind (first
    period where (C1) or (C2) fails).  The greedy trajectory and the
    thresholds are kept as lists over [l, k] and exposed as dictionaries
    indexed by period, built on first access (the separation does not need
    them):
      x[t]       greedy net charging, t in [l, k]
      s_prev[t]  SoC at the end of t-1, t in [l, k]
      s_end[t]   SoC at the end of t,   t in [l-1, k]
      kappa[t]   thresholds,            t in [l-1, k]
    """
    __slots__ = ('Omega', 'l', 'k', 'f', 'pi', 'D', 'fail_at', 'fail_kind',
                 '_x', '_s_prev', '_s_end', '_kappa', '_s_start', '_dicts')

    def __init__(self, Omega, l, k, f, pi, D, x, s_prev, s_end, kappa, fail_at, fail_kind, s_start):
        self.Omega, self.l, self.k, self.f, self.pi, self.D = Omega, l, k, f, pi, D
        self.fail_at, self.fail_kind = fail_at, fail_kind
        self._x, self._s_prev, self._s_end, self._kappa, self._s_start = x, s_prev, s_end, kappa, s_start
        self._dicts = None

    def _build(self):
        l, k = self.l, self.k
        self._dicts = ({t: self._x[t - l] for t in range(l, k + 1)},
                       {t: self._s_prev[t - l] for t in range(l, k + 1)},
                       {t: self._s_end[t - l] for t in range(l, k + 1)} | {l - 1: self._s_start},
                       {t: self._kappa[t - l + 1] for t in range(l - 1, k + 1)})
        return self._dicts

    @property
    def x(self) -> Dict[int, float]:
        return (self._dicts or self._build())[0]

    @property
    def s_prev(self) -> Dict[int, float]:
        return (self._dicts or self._build())[1]

    @property
    def s_end(self) -> Dict[int, float]:
        return (self._dicts or self._build())[2]

    @property
    def kappa(self) -> Dict[int, float]:
        return (self._dicts or self._build())[3]


@dataclass
class Assessment:
    """Output of Storage.assess."""
    ev: "SetData"
    v_rob: float
    v_act: float
    D_rob: frozenset
    D_act: frozenset
    free: List[int]


# ---------------------------------------------------------------------------
# 2. Points and cuts
# ---------------------------------------------------------------------------
@dataclass
class Point:
    """LP point in normalized units (1-indexed lists of length T+1)."""
    c: List[float]
    d: List[float]
    u: List[float]

    @classmethod
    def from_physical(cls, pc, pd, u, eta_c, eta_d, dt=1.0):
        T = len(pc)
        return cls([0.0] + [dt * eta_c * pc[t] for t in range(T)],
                   [0.0] + [dt * pd[t] / eta_d for t in range(T)],
                   [0.0] + [float(u[t]) for t in range(T)])

    def mirror(self) -> "Point":
        return Point(list(self.d), list(self.c), [0.0] + [1.0 - v for v in self.u[1:]])

    def simultaneous_periods(self, eps: float) -> List[int]:
        """Periods with simultaneous charging and discharging."""
        return [t for t in range(1, len(self.c)) if self.c[t] > eps and self.d[t] > eps]

    gamma = simultaneous_periods      # alias kept for the auxiliary scripts


@dataclass
class Cut:
    """A cut of the charging family of the unit ``inst`` (which may be the
    mirror of the user's unit, in which case ``mirrored`` is True and the
    exported coefficients are reflected back).

        sum_{t in Omega} c_t + sum_{t in lifts} alpha_t c_t
            - sum_{t in D} d_t - sum_{tau in Omega \\ D} pi_tau (1 - u_tau) <= f
    """
    inst: Storage
    Omega: Tuple[int, ...]
    D: frozenset
    pi: Dict[int, float]
    f: float
    lifts: Dict[int, float] = field(default_factory=dict)   # t -> alpha_t (t notin Omega)
    mirrored: bool = False
    origin: str = ''                                        # bookkeeping

    # -- identity ------------------------------------------------------------
    def key(self):
        return (self.mirrored, self.Omega, tuple(sorted(self.D)),
                tuple((t, round(a, 9)) for t, a in sorted(self.lifts.items())))

    @property
    def kind(self) -> str:
        return 'discharge' if self.mirrored else 'charge'

    # -- coefficients in the cut's own orientation ------------------------------
    def own_coeffs(self):
        """(a_c, a_d, a_u, rhs) with a_c c + a_d d + a_u u <= rhs in the
        variables of ``inst``."""
        a_c = {t: 1.0 for t in self.Omega}
        for t, al in self.lifts.items():
            a_c[t] = a_c.get(t, 0.0) + al
        a_d = {t: -1.0 for t in self.D}
        a_u = {t: self.pi[t] for t in self.Omega if t not in self.D}
        rhs = self.f + sum(a_u.values())
        return a_c, a_d, a_u, rhs

    def coeffs(self):
        """Coefficients in the user's (d, c, u) space (reflected if mirrored)."""
        a_c, a_d, a_u, rhs = self.own_coeffs()
        if not self.mirrored:
            return a_c, a_d, a_u, rhs
        # c' = d, d' = c, u' = 1 - u
        return dict(a_d), dict(a_c), {t: -v for t, v in a_u.items()}, rhs - sum(a_u.values())

    def physical(self, eta_c: float, eta_d: float, dt: float = 1.0):
        """Coefficients for the physical variables (p^c, p^d, u):
        c = dt*eta_c*p^c, d = dt*p^d/eta_d."""
        a_c, a_d, a_u, rhs = self.coeffs()
        return ({t: v * dt * eta_c for t, v in a_c.items()},
                {t: v * dt / eta_d for t, v in a_d.items()}, a_u, rhs)

    # -- evaluation at a point (user's orientation) ----------------------------
    def value(self, pt: Point) -> float:
        a_c, a_d, a_u, rhs = self.coeffs()
        return (sum(v * pt.c[t] for t, v in a_c.items()) + sum(v * pt.d[t] for t, v in a_d.items())
                + sum(v * pt.u[t] for t, v in a_u.items()) - rhs)

    def violation(self, pt: Point) -> float:
        return self.value(pt)

    def slack(self, pt: Point) -> float:
        return -self.value(pt)

    def describe(self) -> str:
        a_c, a_d, a_u, rhs = self.coeffs()
        terms = []
        for t, v in sorted(a_c.items()):
            terms.append(f"{v:+.4g} c{t}")
        for t, v in sorted(a_d.items()):
            terms.append(f"{v:+.4g} d{t}")
        for t, v in sorted(a_u.items()):
            terms.append(f"{v:+.4g} u{t}")
        return " ".join(terms) + f" <= {rhs:.6g}   [{self.kind}, Omega={list(self.Omega)}, D={sorted(self.D)}, lifts={ {t: round(a, 4) for t, a in self.lifts.items()} }]"


# ---------------------------------------------------------------------------
# 3. Choice of D and violation of a set at a point
# ---------------------------------------------------------------------------
def choose_D(inst: Storage, ev: SetData, pt: Point, rule: str = 'robust') -> frozenset:
    """Pointwise choice of the continuous/binary term (Prop. term_comparison).

    rule = 'robust': tau in D iff pi_tau > 0 and d*_tau < pi_tau (1 - c*_tau/cbar_tau)
                     (largest violation that survives any u in the box
                      [c*/cbar, 1 - d*/dbar] left free by the LP point);
    rule = 'actual': tau in D iff pi_tau > 0 and d*_tau < pi_tau (1 - u*_tau).
    Periods with pi_tau = dbar_tau always go to D, periods with pi_tau <= 0 never.
    """
    D = set(ev.D)
    for t in ev.Omega:
        p = ev.pi[t]
        if t in ev.D or p <= TOL:
            continue
        if rule == 'robust':
            one_minus_u = 1.0 - pt.c[t] / inst.cbar[t]
        else:
            one_minus_u = 1.0 - pt.u[t]
        if pt.d[t] < p * one_minus_u - TOL:
            D.add(t)
    return frozenset(D)


def make_cut(inst: Storage, ev: SetData, D: frozenset, mirrored: bool, origin: str = '') -> Cut:
    return Cut(inst, ev.Omega, D, dict(ev.pi), ev.f, {}, mirrored, origin)


def point_violation(ev: SetData, D: frozenset, pt: Point) -> float:
    """Violation of the cut (Omega, D) at the point of the cut's own orientation."""
    v = -ev.f
    for t in ev.Omega:
        v += pt.c[t]
        if t in D:
            v -= pt.d[t]
        else:
            v -= ev.pi[t] * (1.0 - pt.u[t])
    return v


# ---------------------------------------------------------------------------
# 4. Separation
# ---------------------------------------------------------------------------
@dataclass
class SepStats:
    evaluations: int = 0
    windows: int = 0
    expanded: int = 0
    cuts: int = 0


_SCAN_CHUNK = 4_000_000        # largest number of entries of the [period, window] arrays of _scan_block


def _scan_block(inst: Storage, pt: Point, eps: float, L0: int, K0: int, keep=None, reduced: bool = False) -> dict:
    """Closed-form evaluation of the windows [l,k] with L0 <= l < k <= K0.
    ``keep`` is a boolean array [l - L0, k - L0] selecting the windows to
    return (None: all of them).  With ``reduced``, each entry also carries the
    window minus its free periods, as a sorted tuple (None if the window has
    no free period).

    The windows are selected first, and the gains are then computed only for
    them, in arrays indexed [period, window]: the cost is O(n W) for a block
    of n periods with W selected windows, instead of O(n^3).  The windows are
    processed in chunks, so the memory used is bounded by _SCAN_CHUNK entries
    per array whatever the horizon."""
    import numpy as np
    slo, sup, cb, db = inst.slo, inst.sup, inst.cb, inst.db
    n = K0 - L0 + 1
    cbar = np.asarray(inst.cbar[L0:K0 + 1]); dbar = np.asarray(inst.dbar[L0:K0 + 1])
    sig = np.asarray(inst.sig_lo[L0 - 1:K0])                    # sig_lo[l-1], l in block
    idx = np.arange(n)
    # windows to return: l < k, selected by ``keep``, and (C1) holds at l-1
    cs = np.concatenate([[0.0], np.cumsum(cbar)])                # cs[i] = sum of cbar over the first i periods of the block
    sel = (idx[None, :] > idx[:, None]) & ~((sup - sig)[:, None] >= cs[None, 1:] - cs[:n, None] - TOL)
    if keep is not None:
        sel &= keep
    ii_all, jj_all = np.nonzero(sel)                             # row-major: l ascending, then k ascending
    if len(ii_all) == 0:
        return {}
    c = np.asarray(pt.c[L0:K0 + 1]); d = np.asarray(pt.d[L0:K0 + 1]); u = np.asarray(pt.u[L0:K0 + 1])
    dif = idx[None, :] - idx[:, None]
    s_prev = np.minimum(sig[:, None] + dif * cb, sup)            # [l, tau]: SoC at the end of tau-1 for windows starting at l
    x = np.minimum(cb, sup - s_prev)                             # [l, tau]: greedy charge of tau
    phip = np.minimum(s_prev + cb, sup); phim = np.maximum(s_prev - db, slo)
    kap = np.maximum(sup - dif * cb, slo)                        # [tau, k]: threshold at the end of tau for windows ending at k
    w_rob = (1.0 - c / cbar)[:, None]; w_neg = (d / dbar)[:, None]; w_act = (1.0 - u)[:, None]
    d_col = d[:, None]; c_col = c[:, None]; idx_col = idx[:, None]
    idle = c <= eps
    any_idle = bool(idle.any())
    if any_idle:
        idle_col = idle[:, None]; u_col = (u <= eps)[:, None]
    out = {}
    step = max(1, _SCAN_CHUNK // n)
    for w0 in range(0, len(ii_all), step):
        ii = ii_all[w0:w0 + step]; jj = jj_all[w0:w0 + step]
        W = len(ii)
        # gains of the windows of the chunk, [tau, window]
        kw = kap[:, jj]
        xw = x[ii, :].T
        PIw = np.maximum(phip[ii, :].T - kw, 0.0) - np.maximum(phim[ii, :].T - kw, 0.0) - xw
        cred_rob = np.where(PIw > TOL, np.minimum(d_col, PIw * w_rob), PIw * w_neg)
        cred_act = np.minimum(d_col, PIw * w_act)
        win = (idx_col >= ii[None, :]) & (idx_col <= jj[None, :])                         # [tau, window]: tau in [l, k]
        base = c_col - xw
        # violations: sums over tau in increasing order (cumsum fixes the order of the additions)
        vr = np.cumsum(np.where(win, base - cred_rob, 0.0), axis=0)[-1].tolist()
        va = np.cumsum(np.where(win, base - cred_act, 0.0), axis=0)[-1].tolist()
        li = (ii + L0).tolist(); kj = (jj + L0).tolist()
        # free periods (Lemma free_removal) of the windows of the chunk, all at once
        winT = win.T                                                                      # [window, tau]
        if any_idle:
            FW = winT & (idle_col & ((PIw <= TOL) | (u_col & (d_col >= PIw - TOL)))).T
            rows, cols = np.nonzero(FW)
            f_end = np.cumsum(np.bincount(rows, minlength=W)).tolist()
            f_per = (cols + L0).tolist()
        else:
            FW = None
            f_end = [0] * W
            f_per = []
        if reduced:
            rows, cols = np.nonzero(winT & ~FW if FW is not None else winT)
            r_end = np.cumsum(np.bincount(rows, minlength=W)).tolist()
            r_per = (cols + L0).tolist()
        fs = rs = 0
        for w in range(W):
            fe = f_end[w]
            free = f_per[fs:fe]
            fs = fe
            if reduced:
                re_ = r_end[w]
                out[(li[w], kj[w])] = (vr[w], va[w], free, tuple(r_per[rs:re_]) if free else None)
                rs = re_
            else:
                out[(li[w], kj[w])] = (vr[w], va[w], free)
    return out


def window_scan(inst: Storage, pt: Point, eps: float, Gam: Optional[Sequence[int]] = None,
                envelopes: Optional[Sequence[Tuple[int, int, int]]] = None, _reduced: bool = False):
    """Stage 1 of the separation: contiguous sets [l,k] in closed form.

    For Omega = [l,k] the greedy trajectory and the thresholds are explicit,
        s_{tau-1} = min{sig_lo_{l-1} + (tau-l) cb, sup},  kappa_tau = max{sup - (k-tau) cb, slo}.
    Returns a dict (l,k) -> (v_rob, v_act, free) with the robust and the actual
    violation (pointwise-optimal D) and the free periods (Lemma free_removal)
    of each window at which (C1) holds at l-1, restricted to
      * the windows [l,k] with t in [l,k] and l0 <= l, k <= k0 for some
        envelope (l0, t, k0) in ``envelopes`` (bottleneck envelopes E_t), or
      * the windows meeting Gam if envelopes is None (all windows if Gam is None).
    The envelopes are merged into disjoint blocks and each block is evaluated
    with vectorized operations, so the cost is O(sum of |block|^3) instead of O(T^3).
    (``_reduced`` is for internal use: see _scan_block.)"""
    import numpy as np
    T = inst.T
    if envelopes is None:
        keep = None
        if Gam is not None:
            keep = np.zeros((T, T), dtype=bool)
            for g in Gam:                                        # windows with l <= g <= k
                keep[:g, g - 1:] = True
        return _scan_block(inst, pt, eps, 1, T, keep, _reduced)
    # merge overlapping envelopes into blocks
    env = sorted(envelopes)
    blocks = []
    for (l0, t, k0) in env:
        if blocks and l0 <= blocks[-1][1]:
            blocks[-1][1] = max(blocks[-1][1], k0)
            blocks[-1][2].append((l0, t, k0))
        else:
            blocks.append([l0, k0, [(l0, t, k0)]])
    out = {}
    for L0, K0, envs in blocks:
        if K0 <= L0:
            continue
        n = K0 - L0 + 1
        keep = np.zeros((n, n), dtype=bool)
        for (a, t, b) in envs:                                   # windows with a <= l <= t <= k <= b
            keep[a - L0:t - L0 + 1, t - L0:b - L0 + 1] = True
        out.update(_scan_block(inst, pt, eps, L0, K0, keep, _reduced))
    return out


def _separate_oriented(inst: Storage, pt: Point, eps: float, mirrored: bool,
                       windows: str, both_D: bool, stop_at_first: bool,
                       max_branch: int, stats: SepStats, n_starts: int = 2,
                       local_search: str = 'escalate') -> List[Cut]:
    """Charging-cut separation for the unit ``inst`` at the point ``pt``
    (both already in the cut's orientation).

    Stage 1  every window [l,k] meeting J (periods with simultaneous charging
             and discharging) is evaluated in closed form
             (exact separation over the contiguous subfamily);
    Stage 2  every such window is *settled*: the dominance-preserving moves
             (decomposition at a period where (C1)/(C2) fail, removal of the
             free periods of Lemma free_removal) are applied until a set with
             no such move is reached; cuts are tried at every set visited;
    Stage 3  local search (removal of the period that increases the violation
             most) from the settled bottleneck windows and from the n_starts
             settled sets with the largest violation around each t in J;
             run only if Stages 1-2 found nothing (local_search='escalate'),
             always ('always') or never ('never').
    No set is evaluated twice (memoization)."""
    T = inst.T
    Gam = pt.simultaneous_periods(eps)
    if not Gam:
        return []
    Gset = set(Gam)
    if sys.getrecursionlimit() < 4 * T + 200:               # settle() is recursive, with depth at most T
        sys.setrecursionlimit(4 * T + 200)
    cache: Dict[Tuple[int, ...], Assessment] = {}

    def assess(Om: Tuple[int, ...]) -> Assessment:
        a = cache.get(Om)
        if a is None:
            a = inst.assess(Om, pt, eps, canonical=True)     # every set built below is a sorted tuple
            cache[Om] = a
            stats.evaluations += 1
        return a

    found: Dict[tuple, Cut] = {}
    tried: Set[Tuple[int, ...]] = set()

    def try_cuts(a: Assessment) -> float:
        """Register the violated cuts of Omega; return the best violation."""
        ev = a.ev
        if ev.Omega in tried:
            return a.v_act
        tried.add(ev.Omega)
        best = -math.inf
        for D, v in ((a.D_rob, None), (a.D_act, a.v_act)) if both_D else ((a.D_rob, None),):
            if v is None:
                v = point_violation(ev, D, pt)      # actual violation of the robust-D cut
            best = max(best, v)
            if v > eps:
                cut = make_cut(inst, ev, D, mirrored, origin='window' if ev.k - ev.l + 1 == len(ev.Omega) else 'gapped')
                if cut.key() not in found:
                    found[cut.key()] = cut
                    stats.cuts += 1
        return best

    def admissible(Om: Tuple[int, ...]) -> bool:
        return len(Om) >= 2 and not Gset.isdisjoint(Om)

    settled_cache: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}

    def settle(Om: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Apply the dominance-preserving moves until none applies; try cuts
        on the way; return the irreducible sets reached.  The result is
        memoized for every set visited, not only for the starting one (the
        lists returned are shared: callers must not modify them)."""
        out = settled_cache.get(Om)
        if out is not None:
            return out
        if not admissible(Om):
            return []
        a = assess(Om)
        ev = a.ev
        if ev.fail_at is not None:                          # (a) decomposition
            t = ev.fail_at
            if ev.fail_kind == 'C1':
                out = settle(tuple(q for q in Om if q <= t)) if t > ev.l - 1 else []
            else:
                left = settle(tuple(q for q in Om if q <= t))
                out = left + settle(tuple(q for q in Om if q > t))
        else:
            best = try_cuts(a)                              # (b) cuts of Omega
            out = None
            if not (best > eps and stop_at_first) and a.free:   # (c) free periods
                fs = set(a.free)
                red = tuple(q for q in Om if q not in fs)
                if admissible(red):
                    out = settle(red)
            if out is None:
                out = [Om]
        settled_cache[Om] = out
        return out

    # ---- stage 1 + 2: windows (closed form) and their settled sets --------------
    irreducible: Dict[Tuple[int, ...], float] = {}
    s_star = [inst.s0]
    for t in range(1, T + 1):
        s_star.append(s_star[-1] + pt.c[t] - pt.d[t])
    if windows == 'envelope':
        # windows inside the bottleneck envelope E_t of some t in J: from the period
        # after the last empty SoC of y* before t to the first full SoC after t
        env = []
        for g in Gam:
            l0 = g
            while l0 > 1 and s_star[l0 - 1] > inst.slo + 1e-7:
                l0 -= 1
            k0 = g
            while k0 < T and s_star[k0] < inst.sup - 1e-7:
                k0 += 1
            env.append((l0, g, k0))
        scan = window_scan(inst, pt, eps, envelopes=env, _reduced=True)
    elif windows in ('all', 'scan'):
        scan = window_scan(inst, pt, eps, Gam, _reduced=True)
    else:
        scan = {}
    if scan:
        for (l, k), (v_rob, v_act, free, red) in scan.items():
            if v_rob > eps or v_act > eps:
                try_cuts(assess(tuple(range(l, k + 1))))
            if free:                                        # start from the reduced set
                if admissible(red):
                    for Om in settle(red):
                        irreducible[Om] = assess(Om).v_rob
            else:
                irreducible[tuple(range(l, k + 1))] = v_rob     # the window itself is irreducible
    stats.windows += len(scan)
    if local_search == 'never' or (local_search == 'escalate' and found):
        return list(found.values())
    # ---- starting sets of the local search ------------------------------------
    starts: List[Tuple[int, ...]] = []
    for g in Gam:
        l = g
        while l > 1 and abs(s_star[l - 1] - inst.slo) > 1e-7:
            l -= 1
        k = g
        while k < T and abs(s_star[k] - inst.sup) > 1e-7:
            k += 1
        if k > l:
            for Om in settle(tuple(range(l, k + 1))):
                if Om not in starts:
                    starts.append(Om)
        cand = sorted(((v, Om) for Om, v in irreducible.items() if g in Om), reverse=True)
        for _, Om in cand[:n_starts]:
            if Om not in starts:
                starts.append(Om)

    # ---- stage 3: local search with memoization --------------------------------
    expanded: Set[Tuple[int, ...]] = set()

    def score(Om: Tuple[int, ...]) -> float:
        a = assess(Om)
        return -math.inf if a.ev.fail_at is not None else a.v_rob

    def explore(start: Tuple[int, ...]):
        stack = [start]
        while stack:
            Om = stack.pop()
            if Om in expanded or not admissible(Om):
                continue
            expanded.add(Om)
            stats.expanded += 1
            for S in settle(Om):
                if S != Om:
                    stack.append(S)
                    continue
                best = try_cuts(assess(S))
                if best > eps and stop_at_first:
                    continue
                cands = []
                for t in S:
                    new = tuple(q for q in S if q != t)
                    if not admissible(new) or new in expanded:
                        continue
                    cands.append((score(new), t not in Gset, t, new))
                cands.sort(reverse=True)
                for _, _, _, new in cands[:max_branch]:
                    stack.append(new)

    for Om in sorted(starts, key=lambda z: (len(z), z)):
        explore(Om)
    return list(found.values())


def separate(inst: Storage, pt: Point, eps: float = 1e-6, windows: str = 'envelope',
             both_D: bool = True, stop_at_first: bool = False, max_branch: int = 1,
             kinds: Sequence[str] = ('charge', 'discharge'),
             stats: Optional[SepStats] = None, n_starts: int = 2,
             local_search: str = 'escalate') -> List[Cut]:
    """Separate charging and discharging cuts at the LP point ``pt``.

    windows      'all'  : every window [l,k] is scanned in closed form (exact
                           over the subfamily of contiguous sets, O(T^3)); the
                           gapped-set exploration starts from the bottleneck
                           windows and the n_starts most violated windows per t in J;
                 'bottleneck': only the bottleneck windows of the paper (as in
                           the original heuristic), for the exploration.
    both_D       also try the 'actual-u' choice of D (both cuts are valid).
    stop_at_first stop reducing a set once a violated cut has been found in it.
    max_branch   number of children kept in the local search (1 = steepest ascent).
    Returns the list of distinct violated cuts (user's orientation), sorted by
    decreasing violation.
    """
    if stats is None:
        stats = SepStats()
    out: List[Cut] = []
    if 'charge' in kinds:
        out += _separate_oriented(inst, pt, eps, False, windows, both_D, stop_at_first, max_branch, stats, n_starts, local_search)
    if 'discharge' in kinds:
        out += _separate_oriented(inst.mirror(), pt.mirror(), eps, True, windows, both_D, stop_at_first, max_branch, stats, n_starts, local_search)
    out.sort(key=lambda c: -c.violation(pt))
    return out


def select_cuts(cuts: List[Cut], pt: Point, eps: float = 1e-6, per_period: int = 1,
                max_total: Optional[int] = None) -> List[Cut]:
    """Keep, for each period t with simultaneous charging and discharging and
    each family (charging / discharging cuts), the ``per_period`` most violated
    cuts whose Omega contains t (in the
    cut's own orientation, Omega is the set of counted periods); then the
    most violated remaining ones up to ``max_total``."""
    Gam = pt.simultaneous_periods(eps)
    ranked = sorted(cuts, key=lambda c: -c.violation(pt))
    chosen: Dict[tuple, Cut] = {}
    for g in Gam:
        for mirrored in (False, True):
            n = 0
            for c in ranked:
                if c.mirrored == mirrored and g in c.Omega:
                    if c.key() not in chosen:
                        chosen[c.key()] = c
                    n += 1
                    if n >= per_period:
                        break
    out = list(chosen.values())
    if max_total is not None:
        for c in ranked:
            if len(out) >= max_total:
                break
            if c.key() not in chosen:
                chosen[c.key()] = c; out.append(c)
    out.sort(key=lambda c: -c.violation(pt))
    return out


# ---------------------------------------------------------------------------
# 5. Lifting of c_t, t notin Omega (failure of (C3))
# ---------------------------------------------------------------------------
def lift_coefficient(cut: Cut, t: int, alpha_cap: Optional[float] = None,
                     max_patterns: int = 1 << 18) -> float:
    """Exact lifting coefficient alpha*_t of c_t in the cut (own orientation).

    alpha*_t = min over relaxation patterns R subseteq Omega \\ D (and over the
    modes of the periods already lifted) of the largest alpha such that
        max_{x in X^{R}_t} [ x(B_R) + sum_{j lifted, u_j=1} alpha_j x_j + alpha x_t ]
            <= Delta(R) + f(B_R),
    where X^R_t is the g-polymatroid of net-charging vectors with the sign
    constraints of the pattern and x_t >= 0, Delta(R) = f(Omega) + pi(R) - f(B_R).
    The maximum is evaluated by the weighted greedy; the piece of the convex
    function of alpha on which the bound is attained is found by scanning.
    ``alpha_cap``: stop (return a value >= alpha_cap) once alpha* is known to
    exceed it (only used for pruning; the returned value is then not exact
    but still valid, i.e., <= alpha*).  Returns +inf if c_t can never be
    positive together with the pattern constraints (never happens under
    Assumption interior_s0).
    """
    inst = cut.inst
    if t in cut.Omega or t in cut.lifts:
        raise ValueError("period already in the cut")
    Om = set(cut.Omega)
    cand = sorted(Om - cut.D)
    L = sorted(cut.lifts)
    if (1 << (len(cand) + len(L))) > max_patterns:
        raise RuntimeError("too many relaxation patterns; use lift_coefficient_dp")
    first = min(min(Om), t, *L) if L else min(min(Om), t)
    last = max(max(Om), t, *L) if L else max(max(Om), t)
    fO = cut.f
    best = math.inf
    pruning = (len(L) == 0)
    cbar_t = inst.cbar[t]

    # DFS over subsets of cand (supersets are pruned when Delta(R) is already too large)
    def dfs(idx: int, R: Set[int]):
        nonlocal best
        B = Om - R
        Delta = fO + sum(cut.pi[q] for q in R) - inst.f(B)
        if pruning and best < math.inf and Delta >= best * cbar_t - TOL and best <= 1.0:
            return      # alpha_R >= min(1, Delta/cbar_t) >= best for this R and its supersets
        if alpha_cap is not None and best <= alpha_cap:
            return
        # modes of the lifted periods: enumerate L1 subseteq L (periods in charging mode)
        for r in range(len(L) + 1):
            for L1 in itertools.combinations(L, r):
                lo = {q: 0.0 for q in (Om - cut.D - R)}
                lo.update({q: 0.0 for q in L1})
                lo[t] = 0.0
                hi = {q: 0.0 for q in R}
                hi.update({q: 0.0 for q in set(L) - set(L1)})
                classes = [(1.0, set(B))] + [(cut.lifts[j], {j}) for j in L1]
                # group equal weights
                grouped: Dict[float, Set[int]] = {}
                for w, C in classes:
                    grouped.setdefault(w, set()).update(C)
                ws = sorted(grouped, reverse=True)          # w_1 > ... > w_m
                S_sets = []
                S: Set[int] = set()
                for w in ws:
                    S = S | grouped[w]
                    S_sets.append(set(S))
                m = len(ws)
                bS = [0.0] + [inst.rank(S_sets[i], lo, hi, first, last) for i in range(m)]
                bSt = [inst.rank({t}, lo, hi, first, last)] + \
                      [inst.rank(S_sets[i] | {t}, lo, hi, first, last) for i in range(m)]
                target = Delta + inst.f(B)

                def F(i: int, alpha: float) -> float:
                    # t processed after classes 1..i
                    v = 0.0
                    for j in range(1, i + 1):
                        v += ws[j - 1] * (bS[j] - bS[j - 1])
                    v += alpha * (bSt[i] - bS[i])
                    for j in range(i + 1, m + 1):
                        v += ws[j - 1] * (bSt[j] - bSt[j - 1])
                    return v
                # pieces from alpha = 0 upwards: piece i covers [w_{i+1}, w_i]
                alpha_R = math.inf
                lo_a = 0.0
                for i in range(m, -1, -1):
                    hi_a = ws[i - 1] if i >= 1 else math.inf
                    g = bSt[i] - bS[i]
                    F_lo = F(i, lo_a)
                    if F_lo > target + 1e-7:
                        alpha_R = lo_a      # should not happen (base cut valid)
                        break
                    if g > TOL:
                        root = lo_a + (target - F_lo) / g
                        if root <= hi_a + TOL:
                            alpha_R = max(root, 0.0)
                            break
                    if hi_a == math.inf:
                        alpha_R = math.inf
                        break
                    lo_a = hi_a
                best = min(best, alpha_R)
        for j in range(idx, len(cand)):
            dfs(j + 1, R | {cand[j]})

    dfs(0, set())
    return best


def lifted_cut(cut: Cut, t: int, alpha: Optional[float] = None) -> Cut:
    """Return the cut with the term alpha c_t added (alpha = alpha*_t by default)."""
    if alpha is None:
        alpha = lift_coefficient(cut, t)
    lifts = dict(cut.lifts)
    lifts[t] = alpha
    return Cut(cut.inst, cut.Omega, cut.D, dict(cut.pi), cut.f, lifts, cut.mirrored, cut.origin + '+lift')


def C3_holds(cut: Cut, t: int) -> bool:
    """(C3) at t notin Omega  <=>  alpha*_t = 0."""
    return lift_coefficient(cut, t, alpha_cap=1e-9) <= 1e-9


def strengthen_pool(inst: Storage, pool: List[Cut], pt: Point, eps: float = 1e-6,
                    slack_tol: float = 1e-7, max_lifts_per_cut: int = 3,
                    only_simultaneous: bool = False) -> List[Cut]:
    """Post-processing when the separation finds nothing but J != emptyset
    (J: periods with simultaneous charging and discharging): lift, in the
    tight cuts of the pool, the periods outside Omega
    (own orientation) whose lifting coefficient is positive and cuts the point.
    Lifting is sequential (one period at a time, the previous lifts included).
    Returns the new (lifted) cuts, which dominate their base cuts."""
    new: List[Cut] = []
    Gam = set(pt.simultaneous_periods(eps))
    for cut in pool:
        sl = cut.slack(pt)
        if sl > slack_tol:
            continue
        ptm = pt.mirror() if cut.mirrored else pt
        cur = cut
        for _ in range(max_lifts_per_cut):
            cands = [t for t in range(1, inst.T + 1)
                     if t not in cur.Omega and t not in cur.lifts and ptm.c[t] > eps
                     and (not only_simultaneous or t in Gam)]
            best_t, best_gain, best_alpha = None, 0.0, 0.0
            for t in cands:
                try:
                    a = lift_coefficient(cur, t)
                except RuntimeError:
                    a = lift_coefficient_dp(cur, t)
                gain = a * ptm.c[t] - cur.slack(pt)
                if a > 1e-9 and gain > best_gain + 1e-12:
                    best_t, best_gain, best_alpha = t, gain, a
            if best_t is None:
                break
            cur = lifted_cut(cur, best_t, best_alpha)
            new.append(cur)
    return new


# ---------------------------------------------------------------------------
# 6. Exact DP oracle over P^{0-1} (validity checks, C4 rotations, fallback)
# ---------------------------------------------------------------------------
def soc_grid(inst: Storage):
    """Lattice of SoC levels of the extreme points (Prop. extreme_points).

    Forward-anchored levels  G_fwd = {beta + a cb - b db : beta in {s0, slo, sup}}
    are the SoC levels reached from the start of a segment (s0, or a tight
    level) by full-power steps; backward-anchored levels
    G_bwd = {beta - a cb + b db : beta in {slo, sup}} are the levels from which
    full-power steps lead exactly to the tight level that closes the segment.
    Levels closer than ``mtol`` (relative) are merged.  Returns an object with
    the sorted grid, the two flags and a tolerant lookup ``find``."""
    T, slo, sup, cb, db = inst.T, inst.slo, inst.sup, inst.cb, inst.db
    scale = max(1.0, abs(slo), abs(sup))
    mtol = 1e-9 * scale
    raw = []                                   # (value, is_fwd, is_bwd)
    for beta in (slo, sup, inst.s0):
        for a in range(T + 1):
            for b in range(T + 1 - a):
                v = beta + a * cb - b * db
                if slo - mtol <= v <= sup + mtol:
                    raw.append((min(max(v, slo), sup), True, False))
    for beta in (slo, sup):
        for a in range(T + 1):
            for b in range(T + 1 - a):
                v = beta - a * cb + b * db
                if slo - mtol <= v <= sup + mtol:
                    v = min(max(v, slo), sup)
                    raw.append((v, False, not (abs(v - slo) <= mtol or abs(v - sup) <= mtol)))
    raw.sort()
    grid, is_fwd, is_bwd = [], [], []
    for v, f, b in raw:
        if grid and abs(v - grid[-1]) <= mtol:
            is_fwd[-1] = is_fwd[-1] or f
            is_bwd[-1] = is_bwd[-1] or b
        else:
            grid.append(v); is_fwd.append(f); is_bwd.append(b)
    return _Grid(grid, is_fwd, is_bwd, mtol)


class _Grid:
    def __init__(self, grid, is_fwd, is_bwd, mtol):
        self.grid, self.is_fwd, self.is_bwd, self.mtol = grid, is_fwd, is_bwd, mtol

    def find(self, v: float):
        """index of the grid level equal to v (within mtol), or None."""
        import bisect
        g = self.grid
        j = bisect.bisect_left(g, v - self.mtol)
        if j < len(g) and abs(g[j] - v) <= self.mtol:
            return j
        return None


def max_over_P01(inst: Storage, a_c: Dict[int, float], a_d: Dict[int, float], a_u: Dict[int, float],
                 grid=None, return_arg: bool = False):
    """Exact maximum of sum a_c c + a_d d + a_u u over P^{0-1}.

    DP over the periods with state (SoC level, phase), where the phase says
    whether the current segment (between tight periods) has already used its
    partial-power period: 'before' states live on G_fwd and may take a full,
    idle, clipped (ending tight) or partial step; 'after' states live on
    G_bwd and may only take full or idle steps, until a tight level is hit.
    By Prop. extreme_points this enumerates exactly the extreme points, so
    the DP value is the exact maximum over the polytope.  O(T |grid|)."""
    from collections import deque
    if grid is None:
        grid = soc_grid(inst)
    G, is_fwd, is_bwd = grid.grid, grid.is_fwd, grid.is_bwd
    find = grid.find
    n = len(G)
    slo, sup, cb, db = inst.slo, inst.sup, inst.cb, inst.db
    T = inst.T
    NEG = -math.inf
    i_lo, i_hi = find(slo), find(sup)
    Vb = [0.0 if is_fwd[i] else NEG for i in range(n)]      # V_{T+1}, phase before
    Va = [NEG] * n                                           # phase after (tail with a partial period: not a vertex)
    args = []
    for t in range(T, 0, -1):
        ac, ad, au = a_c.get(t, 0.0), a_d.get(t, 0.0), a_u.get(t, 0.0)
        idle = max(au, 0.0)
        Nb, Na = [NEG] * n, [NEG] * n
        Ab, Aa = [None] * n, [None] * n
        # -- sliding-window maxima over G_bwd for the partial steps (phase before -> after)
        Uc = [ac * G[j] + Va[j] if is_bwd[j] else NEG for j in range(n)]     # partial charge to G[j]
        Ud = [-ad * G[j] + Va[j] if is_bwd[j] else NEG for j in range(n)]    # partial discharge to G[j]
        dq = deque(); jmax = 0
        best_c = [None] * n
        for i in range(n):
            top = min(G[i] + cb, sup)
            while jmax < n and G[jmax] < top - grid.mtol:
                if is_bwd[jmax]:
                    while dq and Uc[dq[-1]] <= Uc[jmax]:
                        dq.pop()
                    dq.append(jmax)
                jmax += 1
            while dq and G[dq[0]] <= G[i] + grid.mtol:
                dq.popleft()
            best_c[i] = dq[0] if dq else None
        dq = deque(); jmin = 0
        best_d = [None] * n
        for i in range(n):
            bot = max(G[i] - db, slo)
            while jmin <= i and G[jmin] < G[i] - grid.mtol:
                if is_bwd[jmin]:
                    while dq and Ud[dq[-1]] <= Ud[jmin]:
                        dq.pop()
                    dq.append(jmin)
                jmin += 1
            while dq and G[dq[0]] <= bot + grid.mtol:
                dq.popleft()
            best_d[i] = dq[0] if dq else None
        for i in range(n):
            s = G[i]
            # ---------------- phase before ----------------
            if is_fwd[i]:
                cand = []
                if Vb[i] > NEG:
                    cand.append((idle + Vb[i], ('idle', 'b', i)))
                # full / clipped charge
                sp = s + cb
                if sp >= sup - grid.mtol:
                    x = sup - s
                    if Vb[i_hi] > NEG:
                        cand.append((ac * x + au + Vb[i_hi], ('c', 'b', i_hi)))
                else:
                    j = find(sp)
                    if j is not None and is_fwd[j] and Vb[j] > NEG:
                        cand.append((ac * cb + au + Vb[j], ('c', 'b', j)))
                # full / clipped discharge
                sp = s - db
                if sp <= slo + grid.mtol:
                    x = s - slo
                    if Vb[i_lo] > NEG:
                        cand.append((ad * x + Vb[i_lo], ('d', 'b', i_lo)))
                else:
                    j = find(sp)
                    if j is not None and is_fwd[j] and Vb[j] > NEG:
                        cand.append((ad * db + Vb[j], ('d', 'b', j)))
                # partial steps
                j = best_c[i]
                if j is not None and Uc[j] > NEG:
                    cand.append((Uc[j] - ac * s + au, ('c', 'a', j)))
                j = best_d[i]
                if j is not None and Ud[j] > NEG:
                    cand.append((Ud[j] + ad * s, ('d', 'a', j)))
                if cand:
                    Nb[i], Ab[i] = max(cand)
            # ---------------- phase after ----------------
            if is_bwd[i]:
                cand = []
                if Va[i] > NEG:
                    cand.append((idle + Va[i], ('idle', 'a', i)))
                sp = s + cb
                if sp <= sup + grid.mtol:
                    j = find(min(sp, sup))
                    if j is not None:
                        if j == i_hi and Vb[j] > NEG:
                            cand.append((ac * cb + au + Vb[j], ('c', 'b', j)))
                        elif is_bwd[j] and Va[j] > NEG:
                            cand.append((ac * cb + au + Va[j], ('c', 'a', j)))
                sp = s - db
                if sp >= slo - grid.mtol:
                    j = find(max(sp, slo))
                    if j is not None:
                        if j == i_lo and Vb[j] > NEG:
                            cand.append((ad * db + Vb[j], ('d', 'b', j)))
                        elif is_bwd[j] and Va[j] > NEG:
                            cand.append((ad * db + Va[j], ('d', 'a', j)))
                if cand:
                    Na[i], Aa[i] = max(cand)
        Vb, Va = Nb, Na
        args.append((Ab, Aa))
    i0 = find(inst.s0)
    val = Vb[i0]
    if not return_arg:
        return val
    args.reverse()
    x, u = [0.0] * (T + 1), [0] * (T + 1)
    i, ph = i0, 'b'
    for t in range(1, T + 1):
        rec = (args[t - 1][0] if ph == 'b' else args[t - 1][1])[i]
        kind, ph2, j = rec
        x[t] = G[j] - G[i]
        u[t] = 1 if kind == 'c' else (0 if kind == 'd' else (1 if a_u.get(t, 0.0) > 0 else 0))
        i, ph = j, ph2
    return val, x, u


def max_violation_exact(cut: Cut, grid=None) -> float:
    """Exact max violation of the cut over P^{0-1} (<= 0 iff valid)."""
    a_c, a_d, a_u, rhs = cut.coeffs()
    inst = cut.inst.mirror() if cut.mirrored else cut.inst
    return max_over_P01(inst, a_c, a_d, a_u, grid) - rhs


def max_parameter(inst: Storage, a_c, a_d, a_u, rhs, b_c, b_d, b_u, rho, grid=None,
                  tol: float = 1e-9, max_iter: int = 60) -> float:
    """Largest eps >= 0 such that  (a + eps b) . y <= rhs + eps rho  holds for
    every y in P^{0-1}, given that a . y <= rhs is valid.  Dinkelbach iterations
    on the exact DP oracle: eps_{k+1} = min over the maximizer y_k of
    (rhs - a.y_k)/(b.y_k - rho); finite convergence (finitely many vertices).
    Returns +inf if b.y <= rho on the whole set."""
    if grid is None:
        grid = soc_grid(inst)
    T = inst.T
    keys = set(a_c) | set(b_c) | set(a_d) | set(b_d) | set(a_u) | set(b_u)

    def combo(eps):
        ac = {t: a_c.get(t, 0.0) + eps * b_c.get(t, 0.0) for t in keys}
        ad = {t: a_d.get(t, 0.0) + eps * b_d.get(t, 0.0) for t in keys}
        au = {t: a_u.get(t, 0.0) + eps * b_u.get(t, 0.0) for t in keys}
        return ac, ad, au
    # first the direction alone: if max (b.y - rho) <= 0 there is no constraint
    mb = max_over_P01(inst, b_c, b_d, b_u, grid)
    if mb <= rho + tol:
        return math.inf
    eps = math.inf
    for _ in range(max_iter):
        if eps == math.inf:
            # maximizer of the direction: a finite starting value
            val, x, u = max_over_P01(inst, b_c, b_d, b_u, grid, return_arg=True)
        else:
            ac, ad, au = combo(eps)
            val, x, u = max_over_P01(inst, ac, ad, au, grid, return_arg=True)
            if val <= rhs + eps * rho + tol:
                return max(eps, 0.0)
        # evaluate a.y, b.y at the maximizer
        ay = by = 0.0
        for t in range(1, T + 1):
            c = max(x[t], 0.0) if u[t] == 1 else 0.0
            d = max(-x[t], 0.0) if u[t] == 0 else 0.0
            ay += a_c.get(t, 0.0) * c + a_d.get(t, 0.0) * d + a_u.get(t, 0.0) * u[t]
            by += b_c.get(t, 0.0) * c + b_d.get(t, 0.0) * d + b_u.get(t, 0.0) * u[t]
        den = by - rho
        if den <= tol:
            return max(eps, 0.0) if eps < math.inf else 0.0
        new_eps = (rhs - ay) / den
        if eps < math.inf and new_eps >= eps - tol:
            return max(eps, 0.0)
        eps = max(new_eps, 0.0)
    return eps


def lift_coefficient_dp(cut: Cut, t: int, grid=None) -> float:
    """alpha*_t by Dinkelbach iterations on the DP oracle (any |Omega \\ D|)."""
    a_c, a_d, a_u, rhs = cut.own_coeffs()
    return max_parameter(cut.inst, a_c, a_d, a_u, rhs, {t: 1.0}, {}, {}, 0.0, grid)


def L_coeffs(cut: Cut, q: int):
    """Coefficients (a_c, a_d, a_u, const) of L_q(y) - slo in the cut's own
    orientation (eq:L_t):  L_q = s_q - [c(Om<=q) - d(D cap Om<=q) - sum pi (1-u)]."""
    inst = cut.inst
    a_c, a_d, a_u = {}, {}, {}
    for j in range(1, q + 1):
        a_c[j] = a_c.get(j, 0.0) + 1.0
        a_d[j] = a_d.get(j, 0.0) - 1.0
    for j in cut.Omega:
        if j > q:
            continue
        a_c[j] -= 1.0
        if j in cut.D:
            a_d[j] += 1.0
        else:
            a_u[j] = a_u.get(j, 0.0) - cut.pi[j]
    const = inst.s0 - inst.slo + sum(cut.pi[j] for j in cut.Omega if j <= q and j not in cut.D)
    return a_c, a_d, a_u, const


def rotation_range(cut: Cut, q: int, grid=None, signs=(+1.0, -1.0)):
    """Largest eps_plus >= 0 and eps_minus <= 0 such that
    cut + eps (L_q - slo) <= f  is valid (Dinkelbach on the DP oracle).
    (C4) fails at q  <=>  eps_plus > 0 and eps_minus < 0.  An infinite value
    means that the corresponding direction, sg (L_q - slo) <= 0, is itself a
    valid inequality (the rotation is then dominated by cut + that
    inequality).  Only the requested signs are computed (the other entry is
    returned as 0)."""
    inst = cut.inst
    a_c, a_d, a_u, rhs = cut.own_coeffs()
    lc, ld, lu, lconst = L_coeffs(cut, q)
    if grid is None:
        grid = soc_grid(inst)
    out = {+1.0: 0.0, -1.0: 0.0}
    for sg in signs:
        # direction g(y) = sg (L_q(y) - slo) = sg (L.y + lconst): b = sg L, rho = -sg lconst
        e = max_parameter(inst, a_c, a_d, a_u, rhs, {t: sg * v for t, v in lc.items()},
                          {t: sg * v for t, v in ld.items()}, {t: sg * v for t, v in lu.items()},
                          -sg * lconst, grid)
        out[sg] = sg * e            # +-inf when the direction sg (L_q - slo) <= 0 on all of P^{0-1}
    return out[+1.0], out[-1.0]


def rotated_cut_coeffs(cut: Cut, q: int, e: float):
    """Coefficients (own orientation) of  cut + e (L_q - slo) <= f."""
    a_c, a_d, a_u, rhs = cut.own_coeffs()
    lc, ld, lu, lconst = L_coeffs(cut, q)
    ac = {j: a_c.get(j, 0.0) + e * lc.get(j, 0.0) for j in set(a_c) | set(lc)}
    ad = {j: a_d.get(j, 0.0) + e * ld.get(j, 0.0) for j in set(a_d) | set(ld)}
    au = {j: a_u.get(j, 0.0) + e * lu.get(j, 0.0) for j in set(a_u) | set(lu)}
    return ac, ad, au, rhs - e * lconst
