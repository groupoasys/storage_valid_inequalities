"""
Scheduling_with_cuts_gurobipy_v2.py
===================================

Cutting-plane solution of the battery scheduling (arbitrage) problem with the
submodular capacity cuts, on top of gurobipy.  Same models and the same
bookkeeping as Scheduling_with_cuts_gurobipy.py, with the separation of
Section 3.6 (v6) of the paper:

  * SEPARATOR = 'new' : storage_cuts.separate  (two-pass coefficients,
                        exact window scan, dominance moves, local search,
                        robust choice of D) + strengthening stage (lifting
                        of c_t / d_t where (C3) fails) when no cut is found;
  * SEPARATOR = 'old' : the original heuristic (old_separation.py, verbatim),
                        for comparison.

The relaxation 'lp_c' contains the singleton cuts (period-dependent power
limits PC[t], PD[t]) and the reachable SoC bounds (U*)/(L*) of Prop. 5
(which dominate Pozo's SOC_up / SOC_lo); the Pozo relaxation 'lp' and the
MIP are unchanged.  Every cut is stored in physical units and added with a
single addLConstr; Gurobi re-optimizes with the dual simplex from the
previous basis.

The separation is only called when the LP solution has periods with
simultaneous charging and discharging (both branches): otherwise its power
profile is feasible for the MIP with binary u, hence optimal, and the loop
stops.  Timing: see the comment at the start of the lp_c loop (disjoint
buckets solver / read / separation / cut management; total = their sum).

Data: ESS_data_SPTP.csv and scenarios_prices.xlsx as in the original script.
If they are not found (e.g. in a sandbox), N_SYNTHETIC random instances are
generated so that the script can be run anywhere.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from storage_cuts import (Storage, Point, Cut, separate, select_cuts, strengthen_pool, SepStats,
                          max_violation_exact, max_over_P01)
import old_separation as old

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
VERBOSE = False            # log de Gurobi
PRINT_ITER = False         # resumen por iteración del bucle de cortes
SEPARATOR = 'new'          # 'new' | 'old'
LIFT = True                # etapa de fortalecimiento (solo con SEPARATOR = 'new')
CUTS_PER_PERIOD = 0        # cortes añadidos por cada periodo con carga y descarga simultáneas y por familia (carga/descarga); 0 = todos los violados
CUT_SCORE = 'violation'    # 'violation' | 'efficacy'   (criterio para ordenar/seleccionar cortes)
EPS_VIOL = 1e-6            # violación mínima para añadir un corte (unidades normalizadas: energía)
EPS_SIMULT = 1e-6          # umbral para considerar que un periodo carga y descarga a la vez: c_t, d_t > EPS_SIMULT (unidades normalizadas)
REACHABLE_BOUNDS = True    # (U*)/(L*) en lp_c en lugar de SOC_up/SOC_lo de Pozo
CERTIFY = os.environ.get('CERTIFY', '0') == '1'   # certificar con el oráculo exacto la validez de cada corte (depuración)
MAX_ITER = 1000
CPU_CLOCK = time.process_time   # reloj unico para todas las partidas de Python (CPU del proceso; Gurobi va aparte con m.Runtime)
RUN_MIP_DEFAULT = True    # MIP con presolve/cortes/heuristicas; si False, no se resuelve y la referencia para los gaps es el MIP sin ayudas
RUN_MIP_PLAIN = True       # MIP sin presolve/cortes/heuristicas; si False, la referencia para los gaps es el MIP por defecto
MIP_TIME_LIMIT = 30      # limite de tiempo (s) para los MIP; None = sin limite (con T grande, el MIP sin ayudas puede no terminar)
N_DAYS = 1                 # horizonte = 24*N_DAYS horas, concatenando los escenarios slurm, slurm+1, ... (modulo el numero de escenarios)
PRINT_PROGRESS = False     # una linea por instancia (util con horizontes largos)
SCENARIOS = 'all'         # escenarios de precios a resolver uno detras de otro: 'all' o lista 1-based, p. ej. [1, 5]
                           # (si existe SLURM_ARRAY_TASK_ID, se resuelve solo ese escenario)
N_SYNTHETIC = int(os.environ.get('N_SYNTHETIC', 100))   # instancias sintéticas si no hay ficheros de datos
SEED_SYNTHETIC = 1

ENV = gp.Env(empty=True)
ENV.setParam('OutputFlag', 1 if VERBOSE else 0)
ENV.start()


# =============================================================================
# MODELOS
# =============================================================================
def build_model(kind, T, price, t_step, S_lo, S_up, S_ini, PC_MAX, PD_MAX, nu_c, nu_d, PC, PD, min_s, max_s):
    """kind = 'mip' | 'mip_default' | 'lp' (Pozo) | 'lp_c' (relajación reforzada, base del bucle)."""
    m = gp.Model(kind, env=ENV)
    p = np.asarray(price, dtype=float)
    is_mip = kind.startswith('mip')
    c = m.addVars(T, lb=0.0, name='charge')
    d = m.addVars(T, lb=0.0, name='discharge')
    s = m.addVars(T, lb=S_lo, ub=S_up, name='s')
    u = m.addVars(T, lb=0.0, ub=1.0, vtype=GRB.BINARY if is_mip else GRB.CONTINUOUS, name='u')
    m.setObjective(gp.quicksum(p[t] * (d[t] - c[t]) for t in range(T)), GRB.MAXIMIZE)

    if kind == 'lp_c':                                   # cortes unitarios (eq:singleton_cuts)
        pc_lim = [float(PC[t][0]) for t in range(T)]
        pd_lim = [float(PD[t][0]) for t in range(T)]
    else:
        pc_lim = [float(PC_MAX)] * T
        pd_lim = [float(PD_MAX)] * T
    m.addConstrs((c[t] <= pc_lim[t] * u[t] for t in range(T)), name='max_charge_t')
    m.addConstrs((d[t] <= pd_lim[t] * (1 - u[t]) for t in range(T)), name='max_discharge_t')

    prev = lambda t: S_ini if t == 0 else s[t - 1]
    m.addConstrs((s[t] == prev(t) + t_step * (nu_c * c[t] - d[t] / nu_d) for t in range(T)), name='SOC')

    if kind == 'lp' or (kind == 'lp_c' and not REACHABLE_BOUNDS):
        # cotas de Pozo (2023)
        m.addConstrs((prev(t) + t_step * nu_c * c[t] <= S_up for t in range(T)), name='SOC_up')
        m.addConstrs((prev(t) - t_step * d[t] / nu_d >= S_lo for t in range(T)), name='SOC_lo')
    elif kind == 'lp_c':
        # cotas de SoC alcanzable (U*)/(L*), Prop. reachable_soc_bounds: sigma_hi(t) = max_s[t], sigma_hi(-1) = S_ini
        sh = lambda t: S_ini if t < 0 else float(max_s[t][0])
        sl = lambda t: S_ini if t < 0 else float(min_s[t][0])
        m.addConstrs((prev(t) + t_step * nu_c * c[t] <= sh(t - 1) * (1 - u[t]) + sh(t) * u[t] for t in range(T)), name='U_star')
        m.addConstrs((prev(t) - t_step * d[t] / nu_d >= sl(t - 1) * u[t] + sl(t) * (1 - u[t]) for t in range(T)), name='L_star')

    m.Params.Threads = 1
    if kind != 'mip_default':
        m.Params.Presolve = 0
    if kind == 'mip':
        m.Params.Cuts = 0
        m.Params.Heuristics = 0
    if kind.startswith('mip') and MIP_TIME_LIMIT is not None:
        m.Params.TimeLimit = float(MIP_TIME_LIMIT)
    return m, c, d, s, u


def values(m, var, T):
    return np.array(m.getAttr('X', [var[t] for t in range(T)]))


def to_df(vals, name, ndigits=None):
    if ndigits is not None:
        vals = np.round(vals, ndigits)
    return pd.DataFrame({f'{name}.val': vals}, index=range(len(vals)))


# =============================================================================
# CORTES: forma genérica  sum_t (ac[t] charge[t] + ad[t] discharge[t] + au[t] u[t]) <= rhs  (unidades físicas)
# =============================================================================
def add_generic_cut(m, c, d, u, a_c, a_d, a_u, rhs, name=None):
    vars_, coefs = [], []
    for t, v in a_c.items():
        if v != 0.0:
            vars_.append(c[t - 1]); coefs.append(float(v))
    for t, v in a_d.items():
        if v != 0.0:
            vars_.append(d[t - 1]); coefs.append(float(v))
    for t, v in a_u.items():
        if v != 0.0:
            vars_.append(u[t - 1]); coefs.append(float(v))
    return m.addLConstr(gp.LinExpr(coefs, vars_), GRB.LESS_EQUAL, float(rhs), name=name or '')


def old_cuts_to_generic(charge_cuts, discharge_cuts, nu_c, nu_d):
    """Cortes del separador antiguo (alpha, beta, rhs; periodos 0-indexados) -> forma genérica 1-indexada."""
    out = []
    for cut in charge_cuts:        # sum c - alpha d - beta (1-u) <= rhs
        a_c = {t + 1: 1.0 for t in cut['Omega']}
        a_d = {t + 1: -float(cut['alpha'][t]) for t in cut['Omega'] if cut['alpha'][t] != 0}
        a_u = {t + 1: float(cut['beta'][t]) for t in cut['Omega'] if cut['beta'][t] != 0}
        out.append(('charge', a_c, a_d, a_u, float(cut['rhs']) + sum(a_u.values()), float(cut['V'])))
    for cut in discharge_cuts:     # sum d - alpha c - beta u <= rhs
        a_d = {t + 1: 1.0 for t in cut['Omega']}
        a_c = {t + 1: -float(cut['alpha'][t]) for t in cut['Omega'] if cut['alpha'][t] != 0}
        a_u = {t + 1: -float(cut['beta'][t]) for t in cut['Omega'] if cut['beta'][t] != 0}
        out.append(('discharge', a_c, a_d, a_u, float(cut['rhs']), float(cut['V'])))
    return out


def efficacy(a_c, a_d, a_u, viol):
    nrm = np.sqrt(sum(v * v for v in a_c.values()) + sum(v * v for v in a_d.values()) + sum(v * v for v in a_u.values()))
    return viol / nrm if nrm > 0 else viol


# =============================================================================
# RESOLUCIÓN
# =============================================================================
def solve_storage(modelo, data):
    T, price, t_step = data['T'], data['price'], data['t_step']
    S_lo, S_up, S_ini, PC_MAX, PD_MAX, nu_c, nu_d = (data[k] for k in ('S_lo', 'S_up', 'S_ini', 'PC_MAX', 'PD_MAX', 'nu_c', 'nu_d'))
    PCe, PDe, min_s, max_s, PC, PD = (data[k] for k in ('PCe', 'PDe', 'min_s', 'max_s', 'PC', 'PD'))

    t0 = time.perf_counter()
    m, c, d, s, u = build_model(modelo, T, price, t_step, S_lo, S_up, S_ini, PC_MAX, PD_MAX, nu_c, nu_d, PC, PD, min_s, max_s)
    build_time = time.perf_counter() - t0

    if modelo in ('mip', 'mip_default', 'lp'):
        m.optimize()
        ok = m.Status == GRB.OPTIMAL or (modelo.startswith('mip') and m.Status == GRB.TIME_LIMIT and m.SolCount > 0)
        if not ok:
            raise RuntimeError(f"{modelo}: estado {m.Status}")
        info = {'build_time': build_time, 'simplex_its': int(m.IterCount)}
        if modelo.startswith('mip'):
            info['nodes'] = int(m.NodeCount)
            info['time_limit'] = m.Status == GRB.TIME_LIMIT
            info['mip_gap'] = float(m.MIPGap)
        out = (to_df(values(m, c, T), 'charge', 7), to_df(values(m, d, T), 'discharge', 7), to_df(values(m, s, T), 's', 7),
               m.ObjVal, to_df(values(m, u, T), 'u', 7), m.Runtime, info)
        m.dispose()
        return out

    # ---------------- lp_c: bucle de planos de corte en memoria ----------------
    # Contabilidad de tiempos (cada instante de la iteracion cae en UNA sola partida):
    #   solver_time : suma de m.Runtime (tiempo de Gurobi en optimize()).
    #   read_time   : lectura de la solucion del LP (getAttr, conversion al modelo
    #                 normalizado, DataFrames del separador antiguo) y comprobacion
    #                 de si hay periodos con carga y descarga simultaneas.
    #   sep_time    : solo las llamadas a las rutinas de separacion: separate() y,
    #                 si se activa, strengthen_pool() (lift_time es parte de sep_time)
    #                 en la rama 'new'; las dos llamadas del separador antiguo en 'old'.
    #   mgmt_time   : gestion de los cortes encontrados: seleccion, eliminacion de
    #                 duplicados, paso a unidades fisicas y addLConstr.
    #   total_time  = solver_time + read_time + sep_time + mgmt_time.
    # Las partidas de Python se miden con el mismo reloj (CPU_CLOCK). La certificacion
    # opcional (CERTIFY) queda fuera de todas las partidas.
    inst = Storage.from_physical(T, S_lo, S_up, S_ini, PC_MAX, PD_MAX, nu_c, nu_d, t_step)   # modelo normalizado
    solver_time = read_time = sep_time = mgmt_time = lift_time = 0.0
    simplex_its = n_cuts_c = n_cuts_d = n_lifted = iteration = sep_calls = 0
    stop_reason = ''
    pool, seen = [], set()
    stats = SepStats()
    while True:
        iteration += 1
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"lp_c iter {iteration}: estado {m.Status}")
        solver_time += m.Runtime
        simplex_its += int(m.IterCount)
        if iteration >= MAX_ITER:
            stop_reason = 'max_iter'
            break

        # ---- 1. lectura de la solucion y criterio de parada ----
        t0 = CPU_CLOCK()
        pc, pd_, uu = values(m, c, T), values(m, d, T), values(m, u, T)
        pt = Point.from_physical(pc, pd_, uu, nu_c, nu_d, t_step)
        simultaneous = pt.simultaneous_periods(EPS_SIMULT)
        if simultaneous and SEPARATOR == 'old':
            c_lp, d_lp, u_lp = to_df(pc, 'charge'), to_df(pd_, 'discharge'), to_df(uu, 'u')
            soc_lp = to_df(values(m, s, T), 's')
        read_time += CPU_CLOCK() - t0
        if not simultaneous:
            # Sin carga y descarga simultaneas, el perfil de potencias del LP es factible
            # (con u binaria) y, por tanto, optimo para el MIP: no hace falta separar.
            stop_reason = 'sin carga y descarga simultaneas'
            break

        # ---- 2. separacion ----
        sep_calls += 1
        t0 = CPU_CLOCK()
        if SEPARATOR == 'new':
            found = separate(inst, pt, eps=EPS_VIOL, stats=stats)
        else:
            cc_ = old.separation_algorithm(c_lp, d_lp, soc_lp, u_lp, min_s, PC, PD, T, t_step, nu_c, nu_d, S_up, S_lo, S_ini)
            dd_ = old.separation_algorithm_discharge(c_lp, d_lp, soc_lp, u_lp, max_s, PC, PD, T, t_step, nu_c, nu_d, S_up, S_lo, S_ini)
        sep_time += CPU_CLOCK() - t0

        # ---- 3. gestion: seleccion, duplicados, unidades fisicas ----
        t0 = CPU_CLOCK()
        new = []
        if SEPARATOR == 'new':
            if CUTS_PER_PERIOD:
                found = select_cuts(found, pt, eps=EPS_VIOL, per_period=CUTS_PER_PERIOD)
            if CUT_SCORE == 'efficacy':
                found.sort(key=lambda cc: -efficacy(*cc.coeffs()[:3], cc.violation(pt)))
            for cc in found:
                if cc.key() in seen:
                    continue
                seen.add(cc.key()); pool.append(cc)
                a_c, a_d, a_u, rhs = cc.physical(nu_c, nu_d, t_step)
                new.append((cc.kind, a_c, a_d, a_u, rhs, cc))
        else:
            cands = old_cuts_to_generic(cc_, dd_, nu_c, nu_d)
            if cands:
                score = (lambda z: z[5]) if CUT_SCORE == 'violation' else (lambda z: efficacy(z[1], z[2], z[3], z[5]))
                if CUTS_PER_PERIOD == 0:
                    chosen = cands
                else:                                # el mas violado de cada orientacion ('max_per_type')
                    best = {}
                    for z in cands:
                        if z[0] not in best or score(z) > score(best[z[0]]):
                            best[z[0]] = z
                    chosen = list(best.values())
                new = [(z[0], z[1], z[2], z[3], z[4], None) for z in chosen]
        mgmt_time += CPU_CLOCK() - t0

        # ---- 2b. fortalecimiento (solo 'new', solo si la separacion no encontro nada) ----
        if not new and SEPARATOR == 'new' and LIFT:
            t0 = CPU_CLOCK()
            lifted = strengthen_pool(inst, pool, pt, eps=EPS_VIOL)
            dt_ = CPU_CLOCK() - t0
            sep_time += dt_; lift_time += dt_
            t0 = CPU_CLOCK()
            for cc in lifted:
                if cc.key() in seen or cc.violation(pt) <= EPS_VIOL:
                    continue
                seen.add(cc.key()); pool.append(cc); n_lifted += 1
                a_c, a_d, a_u, rhs = cc.physical(nu_c, nu_d, t_step)
                new.append((cc.kind, a_c, a_d, a_u, rhs, cc))
            mgmt_time += CPU_CLOCK() - t0

        if not new:
            stop_reason = 'sin cortes violados'
            break
        if PRINT_ITER:
            print(f"--- iter {iteration}: obj {m.ObjVal:.6f}, {len(new)} cortes, simplex {int(m.IterCount)}, {m.Runtime:.5f}s")

        # ---- certificacion opcional (fuera de todas las partidas de tiempo) ----
        if CERTIFY:
            for kind, a_c, a_d, a_u, rhs, cc in new:
                if cc is not None:
                    assert max_violation_exact(cc) <= 1e-7, "corte invalido (nuevo)"
                else:
                    # comprobacion en unidades normalizadas: c = dt eta_c p^c, d = dt p^d / eta_d
                    ac_n = {t: v / (t_step * nu_c) for t, v in a_c.items()}
                    ad_n = {t: v * nu_d / t_step for t, v in a_d.items()}
                    assert max_over_P01(inst, ac_n, ad_n, dict(a_u)) - rhs <= 1e-7, "corte invalido (antiguo)"

        # ---- 4. insercion en Gurobi (parte de la gestion de cortes) ----
        t0 = CPU_CLOCK()
        for kind, a_c, a_d, a_u, rhs, cc in new:
            if kind == 'charge':
                n_cuts_c += 1
            else:
                n_cuts_d += 1
            add_generic_cut(m, c, d, u, a_c, a_d, a_u, rhs)
        mgmt_time += CPU_CLOCK() - t0

    total_time = solver_time + read_time + sep_time + mgmt_time
    info = {'build_time': build_time, 'read_time': read_time, 'mgmt_time': mgmt_time, 'lift_time': lift_time,
            'lifted': n_lifted, 'iterations': iteration, 'sep_calls': sep_calls, 'cuts_c': n_cuts_c, 'cuts_d': n_cuts_d,
            'simplex_its': simplex_its, 'stop_reason': stop_reason, 'evaluations': stats.evaluations}
    out = (to_df(values(m, c, T), 'charge', 7), to_df(values(m, d, T), 'discharge', 7), to_df(values(m, s, T), 's', 7),
           m.ObjVal, to_df(values(m, u, T), 'u', 7), solver_time, sep_time, total_time, info)
    m.dispose()
    return out


# =============================================================================
# DATOS
# =============================================================================
def make_data(T, t_step, S_lo, S_up, S_ini, PD_MAX, PC_MAX, nu_c, nu_d, price):
    PDe = min(PD_MAX, (S_up - S_lo) * nu_d / t_step)
    PCe = min(PC_MAX, (S_up - S_lo) / (nu_c * t_step))
    min_s, max_s, PC, PD = old.reachable_levels(T, PCe, PDe, S_up, S_lo, S_ini, t_step, nu_c, nu_d)
    return dict(T=T, t_step=t_step, S_lo=float(S_lo), S_up=float(S_up), S_ini=float(S_ini), PD_MAX=float(PD_MAX),
                PC_MAX=float(PC_MAX), nu_c=float(nu_c), nu_d=float(nu_d), price=np.asarray(price, dtype=float),
                PDe=PDe, PCe=PCe, min_s=min_s, max_s=max_s, PC=PC, PD=PD)


def n_price_scenarios():
    """Numero de escenarios de precios (columnas de scenarios_prices.xlsx; 10 con datos sinteticos)."""
    if os.path.exists("scenarios_prices.xlsx"):
        return pd.read_excel('scenarios_prices.xlsx', header=None).shape[1]
    return 10


def load_instances(T, t_step, slurm):
    if os.path.exists("ESS_data_SPTP.csv") and os.path.exists("scenarios_prices.xlsx"):
        data_storage = pd.read_csv("ESS_data_SPTP.csv")
        data_prices = pd.read_excel('scenarios_prices.xlsx', header=None)
        n_sc = data_prices.shape[1]
        price = np.concatenate([data_prices.loc[:, (slurm + k) % n_sc].values for k in range(T // 24)])
        return [make_data(T, t_step, r["Emin"], r["Emax"], r["E0"], r["PdMax"], r["PcMax"], r["eta_c"], r["eta_d"], price)
                for _, r in data_storage.iterrows()], 'files'
    # ---- instancias sintéticas (sin ficheros de datos) ----
    rng = np.random.default_rng(SEED_SYNTHETIC + slurm)
    out = []
    for _ in range(N_SYNTHETIC):
        Emax = rng.uniform(2.0, 10.0); Emin = rng.choice([0.0, 0.1 * Emax])
        E0 = rng.uniform(Emin + 0.05 * (Emax - Emin), Emax - 0.05 * (Emax - Emin))
        PcMax = rng.uniform(0.1, 0.6) * (Emax - Emin); PdMax = rng.uniform(0.1, 0.6) * (Emax - Emin)
        eta_c, eta_d = rng.uniform(0.85, 0.98), rng.uniform(0.85, 0.98)
        price = np.cumsum(rng.normal(0, 15, size=T)) + rng.normal(0, 10)
        price -= price.mean() * rng.uniform(0.3, 1.2)
        out.append(make_data(T, t_step, Emin, Emax, E0, PdMax, PcMax, eta_c, eta_d, price))
    return out, 'synthetic'


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def run_scenario(slurm, T, t_step):
    """Resuelve las instancias de un escenario de precios (indice slurm, 0-based) y devuelve los agregados."""
    instances, source = load_instances(T, t_step, slurm)
    print(f"=== Escenario {slurm + 1} (horizonte {T} h) === "
          f"{len(instances)} instancias ({source}), separador = {SEPARATOR}, lift = {LIFT and SEPARATOR == 'new'}, "
          f"cortes por periodo = {CUTS_PER_PERIOD}, reachable bounds = {REACHABLE_BOUNDS}")

    mat_mip, mat_pozo, mat_morales, mat_mip_def = [], [], [], []
    acc_time_mipd = 0.0; acc_nodes_mipd = acc_nodes_mip = 0; max_dobj_mipd = 0.0; n_tl_mip = n_tl_mipd = 0
    failures_mip = failures_pozo = failures_morales = 0
    acc_time_mip = acc_time_pozo = 0.0
    acc_w_time_morales = acc_sep_time_morales = acc_time_morales = acc_mgmt_time_morales = acc_read_time_morales = acc_lift_time = 0.0
    n_sep_calls_total = 0
    sum_failures_mip = sum_failures_pozo = sum_failures_morales = 0.0
    acc_gap_pozo = acc_gap_morales = 0.0
    n_lifted_total = n_iter_total = n_cuts_total = 0

    for data in instances:
        # ---------------- MIP ----------------
        if RUN_MIP_PLAIN:
            c_mip, d_mip, soc_mip, obj_mip, u_mip, work_time_mip, info_mip = solve_storage('mip', data)
            cl, dl = c_mip['charge.val'].tolist(), d_mip['discharge.val'].tolist()
            prods = [x * y for x, y in zip(cl, dl)]
            mat_mip.append([obj_mip, sum(prods)] + cl + dl + [work_time_mip, info_mip['nodes'], info_mip['simplex_its'],
                                                          info_mip['time_limit'], info_mip['mip_gap']])
            sum_failures_mip += sum(prods); failures_mip += sum(1 for v in prods if abs(v) > 1e-4); acc_time_mip += work_time_mip
            acc_nodes_mip += info_mip['nodes']; n_tl_mip += info_mip['time_limit']
        if RUN_MIP_DEFAULT or not RUN_MIP_PLAIN:
            _, _, _, obj_mipd, _, time_mipd, info_mipd = solve_storage('mip_default', data)
            mat_mip_def.append([obj_mipd, time_mipd, info_mipd['nodes'], info_mipd['simplex_its'], info_mipd['time_limit'], info_mipd['mip_gap']])
            acc_time_mipd += time_mipd; acc_nodes_mipd += info_mipd['nodes']; n_tl_mipd += info_mipd['time_limit']
            if RUN_MIP_PLAIN:
                max_dobj_mipd = max(max_dobj_mipd, abs(obj_mipd - obj_mip) / max(1.0, abs(obj_mip)))
        # referencia para los gaps: el MIP sin ayudas si se resuelve hasta el optimo; si no, el mejor de los dos
        if not RUN_MIP_PLAIN:
            obj_mip = obj_mipd
        elif (RUN_MIP_DEFAULT and info_mip['time_limit']):
            obj_mip = max(obj_mip, obj_mipd)
        # ---------------- LP de Pozo ----------------
        c_lp, d_lp, soc_lp, obj_lp, u_lp, work_time_lp, info_lp = solve_storage('lp', data)
        cl, dl = c_lp['charge.val'].tolist(), d_lp['discharge.val'].tolist()
        prods = [x * y for x, y in zip(cl, dl)]
        gap_pozo = (obj_lp - obj_mip) / obj_mip if obj_mip != 0 else 0.0
        mat_pozo.append([obj_lp, sum(prods)] + cl + dl + [work_time_lp, gap_pozo])
        sum_failures_pozo += sum(prods); failures_pozo += sum(1 for v in prods if abs(v) > 1e-4)
        acc_time_pozo += work_time_lp; acc_gap_pozo += gap_pozo
        # ---------------- LP con cortes ----------------
        (c_lp_c, d_lp_c, soc_lp_c, obj_lp_c, u_lp_c, work_time_lp_c, sep_time_lp_c, total_time_lp_c, info_c) = solve_storage('lp_c', data)
        cl, dl = c_lp_c['charge.val'].tolist(), d_lp_c['discharge.val'].tolist()
        prods = [x * y for x, y in zip(cl, dl)]
        gap_morales = (obj_lp_c - obj_mip) / obj_mip if obj_mip != 0 else 0.0
        mat_morales.append([obj_lp_c, sum(prods)] + cl + dl + [work_time_lp_c, sep_time_lp_c, total_time_lp_c, gap_morales,
                                                               info_c['iterations'], info_c['cuts_c'] + info_c['cuts_d'],
                                                               info_c['simplex_its'], info_c['read_time'], info_c['mgmt_time'],
                                                               info_c['lift_time'], info_c['sep_calls'], info_c['lifted'], info_c['stop_reason']])
        sum_failures_morales += sum(prods); failures_morales += sum(1 for v in prods if abs(v) > 1e-4)
        acc_time_morales += total_time_lp_c; acc_w_time_morales += work_time_lp_c; acc_sep_time_morales += sep_time_lp_c
        acc_gap_morales += gap_morales; acc_mgmt_time_morales += info_c['mgmt_time']; acc_read_time_morales += info_c['read_time']
        acc_lift_time += info_c['lift_time']; n_sep_calls_total += info_c['sep_calls']
        n_lifted_total += info_c['lifted']; n_iter_total += info_c['iterations']; n_cuts_total += info_c['cuts_c'] + info_c['cuts_d']
        if PRINT_PROGRESS:
            k = len(mat_morales)
            msg = f"[{k}/{len(instances)}]"
            if RUN_MIP_PLAIN:
                msg += f" MIP {1000*work_time_mip:.1f} ms{' (limite)' if info_mip['time_limit'] else ''}"
            if RUN_MIP_DEFAULT or not RUN_MIP_PLAIN:
                msg += f" | MIP-D {1000*time_mipd:.1f} ms"
            print(msg + f" | Pozo {sum(1 for v in [x * y for x, y in zip(c_lp['charge.val'], d_lp['discharge.val'])] if abs(v) > 1e-4)} h"
                  f" | cortes {1000*total_time_lp_c:.1f} ms, {info_c['sep_calls']} rondas, {info_c['cuts_c'] + info_c['cuts_d']} cortes,"
                  f" {sum(1 for v in prods if abs(v) > 1e-4)} h residuales", flush=True)

    cols = ['pc' + str(t) for t in range(T)] + ['pd' + str(t) for t in range(T)]
    pd.DataFrame(mat_mip, columns=['profit', 'dot_prod'] + cols + ['time', 'nodes', 'simplex_its', 'time_limit', 'mip_gap']).to_csv(f'results_mip_s{slurm}.csv')
    pd.DataFrame(mat_pozo, columns=['profit', 'dot_prod'] + cols + ['time', 'gap']).to_csv(f'results_pozo_s{slurm}.csv')
    pd.DataFrame(mat_morales, columns=['profit', 'dot_prod'] + cols + ['Solver time', 'Separation time', 'Total time', 'gap',
                                       'iterations', 'cuts', 'simplex_its', 'Read time', 'Cut management time',
                                       'Lifting time', 'separation calls', 'lifted', 'stop reason']).to_csv(f'results_morales_{SEPARATOR}_s{slurm}.csv')
    if mat_mip_def:
        pd.DataFrame(mat_mip_def, columns=['profit', 'time', 'nodes', 'simplex_its', 'time_limit', 'mip_gap']).to_csv(f'results_mip_default_s{slurm}.csv')

    n = len(instances)
    if RUN_MIP_PLAIN:
        print(f"MIP   : horas carga-descarga simultanea {failures_mip}, suma pc*pd {sum_failures_mip:.4f}, tiempo {acc_time_mip:.3f}s ({1000*acc_time_mip/n:.1f} ms/inst),"
              f" nodos {acc_nodes_mip}, instancias en el limite de tiempo {n_tl_mip}")
    if RUN_MIP_DEFAULT or not RUN_MIP_PLAIN:
        print(f"MIP-D : (Gurobi por defecto) tiempo {acc_time_mipd:.3f}s ({1000*acc_time_mipd/n:.1f} ms/inst), nodos {acc_nodes_mipd},"
              f" instancias en el limite de tiempo {n_tl_mipd}" + (f"; max |dif. objetivo| relativa con el MIP {max_dobj_mipd:.2e}" if RUN_MIP_PLAIN else ""))
    print(f"POZO  : horas {failures_pozo}, suma pc*pd {sum_failures_pozo:.4f}, tiempo {acc_time_pozo:.3f}s, gap medio {100*acc_gap_pozo/n:.4f}%")
    print(f"CORTES: horas {failures_morales}, suma pc*pd {sum_failures_morales:.4f}, tiempo total {acc_time_morales:.3f}s ({1000*acc_time_morales/n:.1f} ms/inst)"
          f" = solver {acc_w_time_morales:.3f}s + lectura {acc_read_time_morales:.3f}s + separacion {acc_sep_time_morales:.3f}s"
          f" (de los que lifting {acc_lift_time:.3f}s) + gestion de cortes {acc_mgmt_time_morales:.3f}s;"
          f" gap medio {100*acc_gap_morales/n:.4f}%; {n_cuts_total} cortes ({n_lifted_total} elevados);"
          f" {n_iter_total} resoluciones del LP, {n_sep_calls_total} llamadas a la separacion")
    with open(str(slurm) + f'_{SEPARATOR}.res', 'a') as f:
        f.write(','.join(str(v) for v in [slurm, failures_mip, sum_failures_mip, acc_time_mip, failures_pozo, sum_failures_pozo, acc_time_pozo,
                                          acc_gap_pozo, failures_morales, sum_failures_morales, acc_w_time_morales, acc_sep_time_morales,
                                          acc_time_morales, acc_gap_morales]) + '\n')
    return dict(n=n, failures_mip=failures_mip, time_mip=acc_time_mip, time_mipd=acc_time_mipd, tl_mip=n_tl_mip, tl_mipd=n_tl_mipd,
                failures_pozo=failures_pozo, pcpd_pozo=sum_failures_pozo, time_pozo=acc_time_pozo, gap_pozo=acc_gap_pozo,
                failures_cuts=failures_morales, pcpd_cuts=sum_failures_morales, time_cuts=acc_time_morales, solver_cuts=acc_w_time_morales,
                read_cuts=acc_read_time_morales, sep_cuts=acc_sep_time_morales, lift_cuts=acc_lift_time, mgmt_cuts=acc_mgmt_time_morales,
                gap_cuts=acc_gap_morales, cuts=n_cuts_total, lifted=n_lifted_total, lp_solves=n_iter_total, sep_calls=n_sep_calls_total)


if __name__ == '__main__':
    T = 24 * N_DAYS
    t_step = 1
    if len(sys.argv) > 1 and sys.argv[1] in ('old', 'new'):
        SEPARATOR = sys.argv[1]
    if len(sys.argv) > 2:
        LIFT = sys.argv[2] == 'lift'
    # Escenarios a resolver: con SLURM_ARRAY_TASK_ID (1-based) solo ese; si no, los de SCENARIOS, uno detras de otro
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        scen = [int(os.environ['SLURM_ARRAY_TASK_ID']) - 1]
    elif SCENARIOS == 'all':
        scen = list(range(n_price_scenarios()))
    else:
        scen = [k - 1 for k in SCENARIOS]
    tot = {}
    for sl in scen:
        r = run_scenario(sl, T, t_step)
        for key, v in r.items():
            tot[key] = tot.get(key, 0) + v
    if len(scen) > 1:
        n = tot['n']
        print(f"=== TOTAL ({len(scen)} escenarios, {n} instancias, horizonte {T} h) ===")
        if RUN_MIP_PLAIN:
            print(f"MIP   : horas {tot['failures_mip']}, tiempo {tot['time_mip']:.3f}s ({1000*tot['time_mip']/n:.2f} ms/inst), en el limite de tiempo {tot['tl_mip']}")
        if RUN_MIP_DEFAULT or not RUN_MIP_PLAIN:
            print(f"MIP-D : tiempo {tot['time_mipd']:.3f}s ({1000*tot['time_mipd']/n:.2f} ms/inst), en el limite de tiempo {tot['tl_mipd']}")
        print(f"POZO  : horas {tot['failures_pozo']}, suma pc*pd {tot['pcpd_pozo']:.4f}, tiempo {tot['time_pozo']:.3f}s, gap medio {100*tot['gap_pozo']/n:.4f}%")
        print(f"CORTES: horas {tot['failures_cuts']}, suma pc*pd {tot['pcpd_cuts']:.4f}, tiempo total {tot['time_cuts']:.3f}s ({1000*tot['time_cuts']/n:.2f} ms/inst)"
              f" = solver {tot['solver_cuts']:.3f}s + lectura {tot['read_cuts']:.3f}s + separacion {tot['sep_cuts']:.3f}s (de los que lifting {tot['lift_cuts']:.3f}s)"
              f" + gestion de cortes {tot['mgmt_cuts']:.3f}s; gap medio {100*tot['gap_cuts']/n:.4f}%; {tot['cuts']} cortes ({tot['lifted']} elevados);"
              f" {tot['lp_solves']} resoluciones del LP, {tot['sep_calls']} llamadas a la separacion")
    ENV.dispose()
