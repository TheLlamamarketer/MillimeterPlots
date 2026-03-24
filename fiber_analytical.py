
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize, Bounds
from numba import njit


@njit(cache=True)
def _trapz_uniform(y: np.ndarray, du: float) -> float:
    n = y.size
    if n <= 1:
        return 0.0

    acc = 0.5 * (y[0] + y[n - 1])
    for i in range(1, n - 1):
        acc += y[i]
    return acc * du


@njit(cache=True)
def _minimum_length_from_yu(y_u: np.ndarray, du: float) -> float:
    n = y_u.size
    if n <= 1:
        return 0.0

    acc = 0.5 * (abs(y_u[0]) + abs(y_u[n - 1]))
    for i in range(1, n - 1):
        acc += abs(y_u[i])
    return acc * du


@njit(cache=True)
def _arc_length_from_yu(a: float, y_u: np.ndarray, du: float) -> float:
    n = y_u.size
    if n <= 1:
        return 0.0

    first = np.sqrt(a * a + y_u[0] * y_u[0])
    last = np.sqrt(a * a + y_u[n - 1] * y_u[n - 1])
    acc = 0.5 * (first + last)

    for i in range(1, n - 1):
        yu = y_u[i]
        acc += np.sqrt(a * a + yu * yu)

    return acc * du


@njit(cache=True)
def _bending_energy_from_yu(a: float, y_u: np.ndarray, y_uu: np.ndarray, du: float) -> float:
    n = y_u.size
    if n <= 1:
        return 0.0

    def integrand(i: int) -> float:
        yu = y_u[i]
        yuu = y_uu[i]
        base = a * a + yu * yu
        denom = base * base * np.sqrt(base)
        if denom < 1.0e-14:
            denom = 1.0e-14
        num = 0.5 * (a * yuu) * (a * yuu)
        return num / denom

    acc = 0.5 * (integrand(0) + integrand(n - 1))
    for i in range(1, n - 1):
        acc += integrand(i)

    return acc * du


@njit(cache=True)
def _first_derivative_uniform(y: np.ndarray, du: float) -> np.ndarray:
    n = y.size
    out = np.empty_like(y)

    if n == 0:
        return out
    if n == 1:
        out[0] = 0.0
        return out
    if n == 2:
        d = (y[1] - y[0]) / du
        out[0] = d
        out[1] = d
        return out

    # Second-order one-sided stencil at boundaries.
    out[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * du)
    out[n - 1] = (3.0 * y[n - 1] - 4.0 * y[n - 2] + y[n - 3]) / (2.0 * du)

    # Central second-order stencil in the interior.
    for i in range(1, n - 1):
        out[i] = (y[i + 1] - y[i - 1]) / (2.0 * du)

    return out


@njit(cache=True)
def _second_derivative_uniform(y: np.ndarray, du: float) -> np.ndarray:
    n = y.size
    out = np.empty_like(y)

    if n <= 3:
        for i in range(n):
            out[i] = 0.0
        return out

    du2 = du * du

    # Second-order one-sided stencil at boundaries.
    out[0] = (2.0 * y[0] - 5.0 * y[1] + 4.0 * y[2] - y[3]) / du2
    out[n - 1] = (2.0 * y[n - 1] - 5.0 * y[n - 2] + 4.0 * y[n - 3] - y[n - 4]) / du2

    # Central second-order stencil in the interior.
    for i in range(1, n - 1):
        out[i] = (y[i + 1] - 2.0 * y[i] + y[i - 1]) / du2

    return out


@njit(cache=True)
def _min_gap_to_curve_numba(
    center_x: float,
    center_y: float,
    radius_eff: float,
    x_curve: np.ndarray,
    y_curve: np.ndarray,
) -> float:
    n = x_curve.size
    if n == 0:
        return np.inf

    min_dist_sq = np.inf
    for i in range(n):
        dx = x_curve[i] - center_x
        dy = y_curve[i] - center_y
        dist_sq = dx * dx + dy * dy
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq

    return np.sqrt(min_dist_sq) - radius_eff


# ============================================================
# Configuration
# ============================================================

@dataclass
class FiberDesignConfig:
    # Geometry
    b: float = 0.17                 # interior touch height
    total_length: float = 1.02      # fixed arc length of lower fiber
    fiber_diameter: float = 0.001

    # Pegs
    peg_radii: tuple = (0.03, 0.03, 0.03, 0.03)
    peg_side_signs: tuple = (-1, +1, -1, +1)
    peg_touch_upper: tuple = (False, False, True, False)

    # Curve family
    n_free_shape: int = 2

    # Sampling
    n_grid: int = 801

    # Multi start optimization
    n_starts: int = 12
    random_seed: int = 1

    # Bounds
    lambda_min: float = 0.15
    lambda_max: float = 0.85
    f_max: float = 0.20
    coeff_abs_max: float = 3.0
    contact_u_min: float = 0.03
    contact_u_max: float = 0.97
    
    # Physical topology
    touch_buffer_u: float = 0.06
    min_peg_spacing_u: float = 0.08
    min_contact_span_u: float = 0.45

    # Glass fiber bend limit
    min_bend_radius: float = 0.002

    # Prevent geometric collapse in x-direction.
    min_horizontal_scale_a: float = 1.0e-3

    # Objective weights
    w_f: float = 2.0
    w_exit: float = 1.0
    w_bend: float = 1.0e-2
    w_length: float = 1.0e5
    w_above_b: float = 1.0e6
    w_lower_pen: float = 1.0e6
    w_upper_pen: float = 1.0e6
    w_upper_touch: float = 1.0e4
    w_param_reg: float = 1.0e-3
    w_topology: float = 1.0e6
    w_kappa_limit: float = 1.0e6
    w_a_min: float = 1.0e8

    # Feasibility tolerances for reporting
    length_tol: float = 2.0e-4
    length_shortfall_tol: float = 1.0e-10
    gap_tol: float = 2.0e-4
    above_b_tol: float = 2.0e-5


# ============================================================
# Core solver
# ============================================================

class MirroredFiberDesigner:
    """
    Lower fiber:
        gamma_1(x) = (x, y(x))

    Upper fiber:
        gamma_2(x) = (x, 2*b + d - y(x))

    Hard geometric conditions built into the curve family:
        y(0)   = 0
        y'(0)  = 0
        y(x_c) = b
        y'(x_c)= 0
        y(a)   = b - f

    Here x_c = lambda * a, with lambda in (0, 1), and f >= 0.
    The end slope y'(a) is not enforced, only penalized.
    """

    def __init__(self, cfg: FiberDesignConfig):
        self.cfg = cfg
        self.n_pegs = len(cfg.peg_radii)

        if len(cfg.peg_side_signs) != self.n_pegs:
            raise ValueError("peg_side_signs must have same length as peg_radii")
        if len(cfg.peg_touch_upper) != self.n_pegs:
            raise ValueError("peg_touch_upper must have same length as peg_radii")

        self.u = np.linspace(0.0, 1.0, cfg.n_grid)
        self.du = self.u[1] - self.u[0]

    # --------------------------------------------------------
    # Curve family
    # --------------------------------------------------------

    def _base_poly_coeffs(self, lam: float, f: float) -> np.ndarray:
        """
        Build the quartic base polynomial in dimensionless form:

            p(u) = c2 u^2 + c3 u^3 + c4 u^4

        with:
            p(0)   = 0
            p'(0)  = 0
            p(lam) = 1
            p'(lam)= 0
            p(1)   = 1 - f/b

        Then y(x) = b * p(x/a) + nullspace terms.
        """
        beta = f / self.cfg.b

        A = np.array([
            [lam**2,     lam**3,      lam**4],
            [2.0 * lam,  3.0 * lam**2, 4.0 * lam**3],
            [1.0,        1.0,         1.0],
        ], dtype=float)

        rhs = np.array([1.0, 0.0, 1.0 - beta], dtype=float)
        return np.linalg.solve(A, rhs)

    def _shape_basis(self, u: np.ndarray, lam: float) -> np.ndarray:
        """
        Nullspace basis that preserves all hard conditions.

        q_k(u) = u^2 (u-lam)^2 (1-u) u^k

        This vanishes at u=0, u=lam, u=1.
        Its derivative also vanishes at u=0 and u=lam.
        """
        n_free = self.cfg.n_free_shape
        if n_free <= 0:
            return np.empty((u.size, 0), dtype=float)

        core = u**2 * (u - lam)**2 * (1.0 - u)
        cols = [core * u**k for k in range(n_free)]
        return np.column_stack(cols)

    def curve_u(self, z: np.ndarray):
        """
        z layout:
            z[0] = lambda
            z[1] = f
            z[2:2+n_free_shape] = free shape coeffs
            z[2+n_free_shape:]  = peg contact parameters u_i
        """
        cfg = self.cfg

        lam = float(z[0])
        f = float(z[1])
        c = np.asarray(z[2:2 + cfg.n_free_shape], dtype=float)

        c2, c3, c4 = self._base_poly_coeffs(lam, f)

        p = c2 * self.u**2 + c3 * self.u**3 + c4 * self.u**4

        if cfg.n_free_shape > 0:
            B = self._shape_basis(self.u, lam)
            p = p + B @ c

        y = cfg.b * p

        y_u = _first_derivative_uniform(y, self.du)
        y_uu = _second_derivative_uniform(y, self.du)

        return y, y_u, y_uu
    
    def arc_length_a(self, a:float, y_u: np.ndarray) -> float:
        return _arc_length_from_yu(a, y_u, self.du)

    def minimum_length(self, y_u: np.ndarray) -> float:
        return _minimum_length_from_yu(y_u, self.du)
    
    def solve_for_a(self, y_u: np.ndarray) -> float:
        length_total = self.cfg.total_length
        L_min = self.minimum_length(y_u)
        tol = self.cfg.length_shortfall_tol

        if L_min > length_total + tol:
            raise ValueError("Total length is too small for the given shape coefficients.")

        def residual(a):
            return self.arc_length_a(a, y_u) - length_total

        # Never bracket starting exactly at 0 to avoid degenerate a=0 roots.
        a_lo = 1.0e-12
        r_lo = residual(a_lo)
        if r_lo >= -tol:
            raise ValueError(
                "Arc-length root is at/below numerical lower bound for a (degenerate). "
                f"residual(a_lo)={r_lo:.12e}, a_lo={a_lo:.12e}."
            )

        a_hi = max(length_total, self.cfg.min_horizontal_scale_a, 1.0e-6)
        r_hi = residual(a_hi)
        for _ in range(10):
            if r_hi > 0.0:
                break
            a_hi *= 2.0
            r_hi = residual(a_hi)

        if r_hi <= 0.0:
            raise ValueError(
                "Failed to bracket arc-length root for a. "
                f"L_min={L_min:.12e}, L_target={length_total:.12e}, "
                f"residual(a_hi)={r_hi:.12e}, a_hi={a_hi:.12e}."
            )

        if not (r_lo < 0.0 and r_hi > 0.0):
            raise ValueError(
                "Invalid root bracket for arc-length solve. "
                f"residual(a_lo)={r_lo:.12e}, residual(a_hi)={r_hi:.12e}."
            )

        return brentq(residual, a_lo, a_hi, xtol=1e-8, rtol=1e-8, maxiter=300)
    
    def curve(self, z: np.ndarray):
        y, y_u, y_uu = self.curve_u(z)
        a = self.solve_for_a(y_u)
        x = a * self.u
        
        y_x = y_u / a
        y_xx = y_uu / a**2
        
        return a, x, y, y_u, y_uu, y_x, y_xx

    def upper_curve_from_lower(self, y_lower: np.ndarray) -> np.ndarray:
        return 2.0 * self.cfg.b + self.cfg.fiber_diameter - y_lower



    # --------------------------------------------------------
    # Geometry and energetics
    # --------------------------------------------------------

    def arc_length(self, y_u: np.ndarray) -> float:
        return _arc_length_from_yu(1.0, y_u, self.du)

    def bending_energy(self, a: float, y_u: np.ndarray, y_uu: np.ndarray) -> float:
        # exact planar curvature
        return _bending_energy_from_yu(a, y_u, y_uu, self.du)

    def peg_centers(self, a: float, z: np.ndarray, y: np.ndarray, y_u: np.ndarray):
        """
        Peg j touches the lower fiber at u_j = z[...].

        The peg center is placed on the lower fiber normal:
            center = point + side_sign * (r + d/2) * n_hat
        """
        cfg = self.cfg

        start = 2 + cfg.n_free_shape
        stop = start + self.n_pegs
        u_pegs = np.asarray(z[start:stop], dtype=float)

        x_contact = a * u_pegs
        y_contact = np.interp(u_pegs, self.u, y)
        yu_contact = np.interp(u_pegs, self.u, y_u)

        norm = np.sqrt(a**2 + yu_contact**2)
        nx = -yu_contact / norm
        ny = a / norm

        centers = np.empty((self.n_pegs, 2), dtype=float)

        for j, (r, side) in enumerate(zip(cfg.peg_radii, cfg.peg_side_signs)):
            R = r + 0.5 * cfg.fiber_diameter
            centers[j, 0] = x_contact[j] + side * R * nx[j]
            centers[j, 1] = y_contact[j] + side * R * ny[j]

        return u_pegs, x_contact, y_contact, yu_contact/a, centers

    def min_gap_to_curve(
        self,
        center: np.ndarray,
        radius_eff: float,
        x_curve: np.ndarray,
        y_curve: np.ndarray,
    ) -> float:
        return _min_gap_to_curve_numba(center[0], center[1], radius_eff, x_curve, y_curve)

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    def objective(self, z: np.ndarray) -> float:
        cfg = self.cfg

        y, y_u, y_uu = self.curve_u(z)
        f = float(z[1])
        
        J = 0.0
        

        if cfg.n_free_shape > 0:
            c = z[2:2 + cfg.n_free_shape]
            J += cfg.w_param_reg * np.dot(c, c)

        length_shortfall = self.minimum_length(y_u) - cfg.total_length
        if length_shortfall > cfg.length_shortfall_tol:
            # Hard barrier against geometrically impossible shapes.
            scaled = length_shortfall / cfg.length_shortfall_tol
            return J + cfg.w_length * (1.0 + scaled * scaled)

        try:
            a = self.solve_for_a(y_u)
        except ValueError:
            # Keep optimizer away from non-bracketable or degenerate candidates.
            return J + cfg.w_length * 1.0e3

        if a < cfg.min_horizontal_scale_a:
            shortfall = cfg.min_horizontal_scale_a - a
            return J + cfg.w_a_min * shortfall * shortfall

        x = a * self.u
        y_x = y_u / a
        y_upper = self.upper_curve_from_lower(y)
        
        J += cfg.w_f * f
        J += cfg.w_exit * y_x[-1]**2                        # Soft exit straightness
        J += cfg.w_bend * self.bending_energy(a, y_u, y_uu) # Prefer gentler shapes

        above = np.maximum(y - cfg.b, 0.0)
        J += cfg.w_above_b * _trapz_uniform(above * above, a * self.du)
        
        
        kappa = a * y_uu / np.maximum((a*a + y_u*y_u)**1.5, 1.0e-14)
        kappa_excess = np.maximum(np.abs(kappa) - 1.0 / cfg.min_bend_radius, 0.0)
        J += cfg.w_kappa_limit * _trapz_uniform(kappa_excess * kappa_excess, a * self.du)        
        # Peg topology penalty
        start = 2 + cfg.n_free_shape
        stop = start + self.n_pegs
        u_pegs = np.asarray(z[start:stop], dtype=float)
        
        topo = 0.0
        
        for i in range(self.n_pegs - 1):
            topo += np.maximum(cfg.min_peg_spacing_u - (u_pegs[i + 1] - u_pegs[i]), 0.0)**2

        topo += np.maximum(cfg.min_contact_span_u - (u_pegs[-1] - u_pegs[0]), 0.0)**2
        
        J += cfg.w_topology * topo

        # Peg geometry penalties
        _, _, _, _, centers = self.peg_centers(a, z, y, y_u)

        for center, r, touch_upper in zip(
            centers,
            cfg.peg_radii,
            cfg.peg_touch_upper,
        ):
            R = r + 0.5 * cfg.fiber_diameter

            gap_lower = self.min_gap_to_curve(center, R, x, y)
            if gap_lower < 0.0:
                J += cfg.w_lower_pen * gap_lower**2

            # Upper fiber either clears the peg or also touches it.
            gap_upper = self.min_gap_to_curve(center, R, x, y_upper)

            if touch_upper:
                if gap_upper < 0.0:
                    J += cfg.w_upper_pen * gap_upper**2
                J += cfg.w_upper_touch * gap_upper**2
            else:
                if gap_upper < 0.0:
                    J += cfg.w_upper_pen * gap_upper**2

        return J

    # --------------------------------------------------------
    # Optimization helpers
    # --------------------------------------------------------

    def bounds(self) -> Bounds:
        cfg = self.cfg

        n = 2 + cfg.n_free_shape + self.n_pegs
        lower = np.full(n, -np.inf, dtype=float)
        upper = np.full(n, +np.inf, dtype=float)

        lower[0] = cfg.lambda_min
        upper[0] = cfg.lambda_max

        lower[1] = 0.0
        upper[1] = cfg.f_max

        if cfg.n_free_shape > 0:
            lower[2:2 + cfg.n_free_shape] = -cfg.coeff_abs_max
            upper[2:2 + cfg.n_free_shape] = +cfg.coeff_abs_max

        start = 2 + cfg.n_free_shape
        lower[start:] = cfg.contact_u_min
        upper[start:] = cfg.contact_u_max

        return Bounds(lower, upper)

    def initial_guess(self, rng: np.random.Generator) -> np.ndarray:
        cfg = self.cfg
        n = 2 + cfg.n_free_shape + self.n_pegs
        z = np.zeros(n, dtype=float)

        lam = rng.uniform(0.40, 0.65)
        z[0] = lam
        z[1] = rng.uniform(0.00003, min(cfg.f_max, 0.7 * cfg.b))

        if cfg.n_free_shape > 0:
            z[2:2 + cfg.n_free_shape] = rng.normal(
                loc=0.0,
                scale=0.05,
                size=cfg.n_free_shape,
            )

        z[2 + cfg.n_free_shape:] = np.sort(rng.uniform(cfg.contact_u_min, cfg.contact_u_max, size=self.n_pegs))
        return z

    def solve(self, verbose: bool = True):
        rng = np.random.default_rng(self.cfg.random_seed)
        bounds = self.bounds()

        best_res = None

        for k in range(self.cfg.n_starts):
            z0 = self.initial_guess(rng)

            res = minimize(
                self.objective,
                z0,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": 800,
                },
            )

            if best_res is None or res.fun < best_res.fun:
                best_res = res

            if verbose:
                print(
                    f"start {k+1:02d}/{self.cfg.n_starts} | "
                    f"objective = {res.fun:.6e} | "
                    f"success = {res.success}"
                )
 
        _, best_y_u, _ = self.curve_u(best_res.x)
        best_min_length = self.minimum_length(best_y_u)
        # Allow tiny numerical margin for floating-point comparison
        if best_min_length > self.cfg.total_length + self.cfg.length_shortfall_tol:
            raise ValueError(
                "No feasible shape found. "
                f"The best candidate still needs at least {best_min_length:.6f} "
                f"of total length, but total_length is {self.cfg.total_length:.6f}."
            )

        try:
            best_a = self.solve_for_a(best_y_u)
        except ValueError as exc:
            raise ValueError(
                "Best candidate is not physically usable: could not solve for horizontal scale a. "
                f"L_min={best_min_length:.12e}, L_target={self.cfg.total_length:.12e}."
            ) from exc

        if best_a < self.cfg.min_horizontal_scale_a:
            raise ValueError(
                "Best candidate is degenerate (collapsed horizontal span). "
                f"a={best_a:.12e} < min_horizontal_scale_a={self.cfg.min_horizontal_scale_a:.12e}. "
                "Increase total_length, relax shape/contact constraints, or reduce min_horizontal_scale_a."
            )
        return self.build_report(best_res)

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------

    def build_report(self, res):
        a, x, y, y_u, y_uu, dy_dx, d2y_dx2 = self.curve(res.x)
        y_upper = self.upper_curve_from_lower(y)

        lam = float(res.x[0])
        f = float(res.x[1])

        u_pegs, x_pegs, y_pegs, slope_pegs, centers = self.peg_centers(a, res.x, y, y_u)

        arc_len = self.arc_length_a(a, y_u)
        bend = self.bending_energy(a, y_u, y_uu)

        peg_gaps_lower = []
        peg_gaps_upper = []

        for center, r in zip(centers, self.cfg.peg_radii):
            R = r + 0.5 * self.cfg.fiber_diameter
            peg_gaps_lower.append(self.min_gap_to_curve(center, R, x, y))
            peg_gaps_upper.append(self.min_gap_to_curve(center, R, x, y_upper))

        peg_gaps_lower = np.array(peg_gaps_lower, dtype=float)
        peg_gaps_upper = np.array(peg_gaps_upper, dtype=float)

        max_y_minus_b = float(np.max(y - self.cfg.b))
        exit_slope = float(dy_dx[-1])

        feasible = (
            abs(arc_len - self.cfg.total_length) <= self.cfg.length_tol
            and a >= self.cfg.min_horizontal_scale_a
            and max_y_minus_b <= self.cfg.above_b_tol
            and np.all(peg_gaps_lower >= -self.cfg.gap_tol)
            and np.all(
                np.where(
                    np.array(self.cfg.peg_touch_upper, dtype=bool),
                    np.abs(peg_gaps_upper) <= self.cfg.gap_tol,
                    peg_gaps_upper >= -self.cfg.gap_tol,
                )
            )
        )

        return {
            "optimizer_result": res,
            "objective": float(res.fun),
            "feasible": bool(feasible),

            "a": a,
            "lambda": lam,
            "x_contact": a * lam,
            "f": f,

            "shape_coeffs": res.x[2:2 + self.cfg.n_free_shape].copy(),
            "peg_u": u_pegs.copy(),
            "peg_x_contact": x_pegs.copy(),
            "peg_y_contact": y_pegs.copy(),
            "peg_slopes": slope_pegs.copy(),
            "peg_centers": centers.copy(),

            "x": x.copy(),
            "y_lower": y.copy(),
            "y_upper": y_upper.copy(),
            "dy_dx": dy_dx.copy(),
            "d2y_dx2": d2y_dx2.copy(),

            "arc_length": float(arc_len),
            "length_error": float(arc_len - self.cfg.total_length),
            "bending_energy": float(bend),
            "exit_slope": exit_slope,
            "max_y_minus_b": max_y_minus_b,

            "peg_gaps_lower": peg_gaps_lower,
            "peg_gaps_upper": peg_gaps_upper,
        }


# ============================================================
# Plotting and summary
# ============================================================

def print_report(report):
    print("\n===== Design summary =====")
    print(f"Feasible:        {report['feasible']}")
    print(f"Objective:       {report['objective']:.6e}")
    print(f"x_contact:       {report['x_contact']:.6f}")
    print(f"f:               {report['f']:.6e}")
    print(f"Arc length:      {report['arc_length']:.6f}")
    print(f"Length error:    {report['length_error']:.6e}")
    print(f"Exit slope:      {report['exit_slope']:.6e}")
    print(f"Max(y-b):        {report['max_y_minus_b']:.6e}")
    print(f"Bending energy:  {report['bending_energy']:.6e}")
    print("Peg gaps lower: ", report["peg_gaps_lower"])
    print("Peg gaps upper: ", report["peg_gaps_upper"])
    print("Peg centers:\n", report["peg_centers"])


def plot_report(cfg: FiberDesignConfig, report):
    x = report["x"]
    y_lower = report["y_lower"]
    y_upper = report["y_upper"]
    centers = report["peg_centers"]
    
    dy_dx = report["dy_dx"]

    fig, ax = plt.subplots(figsize=(13, 6))
    
    def normal_vector(dy_dx):
        norm_length = np.sqrt(1.0 + dy_dx**2)
        return np.array([dy_dx, -np.ones_like(dy_dx)]) / norm_length

    diam = 0.5 * cfg.fiber_diameter
    normal = normal_vector(dy_dx)
    normal_mirr = normal_vector(-dy_dx)
    ax.plot(x - diam * normal[0], y_lower - diam * normal[1], color="C0", alpha=0.25)
    ax.plot(x + diam * normal[0], y_lower + diam * normal[1], color="C0", alpha=0.25)
    ax.plot(x + diam * normal_mirr[0], y_upper - diam * normal_mirr[1], color="C1", alpha=0.25)
    ax.plot(x - diam * normal_mirr[0], y_upper + diam * normal_mirr[1], color="C1", alpha=0.25)

    ax.plot(x, y_lower, lw=2.5, label="lower fiber")
    ax.plot(x, y_upper, lw=2.5, label="upper mirrored fiber")

    # interior fiber contact
    xc = report["x_contact"]
    ax.scatter([xc], [cfg.b + diam], s=60, zorder=5, label="interior fiber contact")

    # pegs
    for center, r, side_sign in zip(centers, cfg.peg_radii, cfg.peg_side_signs):
        circle = plt.Circle((center[0], center[1]), r, fill=False, lw=2, color="red")
        ax.add_patch(circle)
        ax.scatter([center[0]], [center[1]], color="red", s=20)

    # lower peg contact points
    ax.scatter(
        report["peg_x_contact"],
        report["peg_y_contact"],
        s=35,
        zorder=5,
        label="peg contacts on lower fiber",
    )

    ax.axhline(cfg.b+diam, color="gray", ls="--", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    cfg = FiberDesignConfig(
        b=8.5e-3,
        total_length=50e-3,
        fiber_diameter=330e-6,

        peg_radii= (1e-3,) * 4,
        peg_side_signs=(-1, +1, -1, -1),

        # Set True for a peg that should also touch the upper fiber
        peg_touch_upper=(False, False, False, False),

        n_free_shape=2,
        n_grid=801,
        n_starts=16,
        random_seed=1,
    )

    designer = MirroredFiberDesigner(cfg)
    report = designer.solve(verbose=True)

    print_report(report)
    plot_report(cfg, report)

