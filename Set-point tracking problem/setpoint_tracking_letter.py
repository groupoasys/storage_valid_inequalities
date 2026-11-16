# -*- coding: utf-8 -*-
"""
Set-point tracking problem (STP) with energy storage -- case study of the letter
"Second-Order Cone Valid Inequalities for Set-Point Tracking with Energy Storage".

For every battery of ESS_data_SPTP.csv and the PV profile selected by the SLURM
array index, the script solves (all in physical units, T = 24, Delta = 1 h):

  miqp          MIQP with binary mode variables u_t                    (Gurobi)
  hch_qp        quadratic relaxation on the HCH model of Pozo (2023)     (Gurobi)
  hch_soc       HCH + cylinder cut, written as the convex QP objective
                sum_t [(p^d_t + p^c_t)^2 - 2 p^s_t (p^d_t - p^c_t) + (p^s_t)^2]
                = sum_t [(p^d_t - p^c_t - p^s_t)^2 + 4 p^d_t p^c_t]       (Gurobi)
  hch_per       HCH + perspective reformulation (Prop. 1 of the letter):
                u_t in [0,1] as an auxiliary continuous variable, p^c_t <= PCe u_t,
                p^d_t <= PDe (1-u_t), z_t >= w^d_t + w^c_t - 2 p^s_t (p^d_t - p^c_t) + (p^s_t)^2,
                (p^d_t)^2 <= w^d_t (1-u_t), (p^c_t)^2 <= w^c_t u_t (rotated cones)  (MOSEK)
Optional (add them to MODELS):
  miqp_plain    MIQP with presolve, cutting planes and heuristics disabled  (Gurobi)
  hch_soc_cone  HCH + cylinder in the explicit SOC form of the old code     (MOSEK)

Timing.  The time of a model is the CPU time of the solver process reported by
AMPL (_solve_time = user + system CPU of the most recent solve command, with
threads=1).  It excludes the generation of the model by AMPL (as the timing of
the companion paper excludes model building) and includes the time the solver
needs to read the problem.  The elapsed time (_solve_elapsed_time) is stored too.

Outputs (one set per SLURM task, slurm = SLURM_ARRAY_TASK_ID - 1):
  results_stp_s<slurm>.csv   one row per (battery, model): objective, tracking
                             error of the solution, hours with p^c p^d > EPS_PROD,
                             sum_t p^c_t p^d_t, CPU and elapsed times, solve status
                             (and the power profiles if SAVE_PROFILES)
  <slurm>.res                one line with the aggregated results per model
Aggregate all tasks with read_results.py.
"""

import ast
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from amplpy import AMPL, add_to_path
sys.path.insert(0, str(Path(__file__).parent.parent))     # aÃ±ade la carpeta al path 
AMPL_DIR = os.environ.get("AMPL_DIR", "/mnt/home/soft/ampl/programs/x86_64_sles156/20260520")
os.environ.setdefault("AMPL_LICFILE", os.path.join(AMPL_DIR, "ampl.lic"))
add_to_path(AMPL_DIR)

# =============================================================================
# CONFIGURATION
# =============================================================================
MODELS = ['miqp', 'hch_qp', 'hch_soc', 'hch_per']        # optional: 'miqp_plain', 'hch_soc_cone'
SOLVER = {'miqp': 'gurobi', 'miqp_plain': 'gurobi', 'hch_qp': 'gurobi',
          'hch_soc': 'gurobi', 'hch_soc_cone': 'mosek', 'hch_per': 'mosek'}
SOLVER_OPTIONS = {'gurobi': 'threads=1 outlev=0', 'mosek': 'threads=1 outlev=0'}
EXTRA_OPTIONS = {'miqp_plain': 'presolve=0 cuts=0 heurfrac=0'}
ESS_FILE = 'ESS_data_SPTP.csv'
PV_FILE = 'PV_and_Wind_data_scenarios.csv'
DEMAND_FILE = 'demand_profile.csv'
PV_PEAK_KW = 35.0            # PV system of the household (kW)
T = 24                       # planning horizon (hours)
T_STEP = 1.0                 # time step (h)
EPS_PROD = 1e-4              # simultaneous charging and discharging: p^c_t p^d_t > EPS_PROD (kW^2), as in Pozo (2023)
ROUND_DECIMALS = None        # None: metrics on the solver's solution; 2: round the powers to 0.01 kW first (old code)
SAVE_PROFILES = False        # store p^c_t and p^d_t of every solution in the CSV
N_DEVICES = None             # None: all the batteries of ESS_FILE; an integer for quick tests

# =============================================================================
# AMPL MODELS
# =============================================================================
DATA = r'''
set T;
param signal{T};
param t_step;
param S_lo;
param S_up;
param S_ini;
param PDe;
param PCe;
param nu_c;
param nu_d;
'''

# SoC limits of the HCH model of Pozo (2023) and single-period convex hull of the power limits
HCH_CONSTRAINTS = r'''
s.t. max_charge_discharge{t in T}: charge[t]/PCe + discharge[t]/PDe <= 1;
s.t. min_stor_level{t in T}: S_lo + (1/nu_d)*t_step*discharge[t]
     <= S_ini + t_step*sum{tt in T: tt < t}(nu_c*charge[tt] - (1/nu_d)*discharge[tt]);
s.t. max_stor_level{t in T}: S_ini + t_step*sum{tt in T: tt < t}(nu_c*charge[tt] - (1/nu_d)*discharge[tt])
     <= S_up - nu_c*t_step*charge[t];
'''

MODEL = {}

MODEL['miqp'] = r'''
var charge{T} >= 0;
var discharge{T} >= 0;
var u{T} binary;                       # 1: charging mode, 0: discharging mode
minimize mismatch: sum{t in T} (discharge[t] - charge[t] - signal[t])^2;
s.t. max_charge_t{t in T}: charge[t] <= PCe*u[t];
s.t. max_discharge_t{t in T}: discharge[t] <= PDe*(1 - u[t]);
s.t. min_stor_level{t in T}: S_lo <= S_ini + t_step*sum{tt in T: tt <= t}(nu_c*charge[tt] - (1/nu_d)*discharge[tt]);
s.t. max_stor_level{t in T}: S_ini + t_step*sum{tt in T: tt <= t}(nu_c*charge[tt] - (1/nu_d)*discharge[tt]) <= S_up;
'''
MODEL['miqp_plain'] = MODEL['miqp']

MODEL['hch_qp'] = r'''
var charge{T} >= 0;
var discharge{T} >= 0;
minimize mismatch: sum{t in T} (discharge[t] - charge[t] - signal[t])^2;
''' + HCH_CONSTRAINTS

# Cylinder (Theorem 1) as a convex QP objective; r_t = p^c_t + p^d_t keeps the Hessian diagonal
MODEL['hch_soc'] = r'''
var charge{T} >= 0;
var discharge{T} >= 0;
var r{T} >= 0;
minimize mismatch: sum{t in T} (r[t]^2 - 2*signal[t]*(discharge[t] - charge[t]) + signal[t]^2);
s.t. def_r{t in T}: r[t] = charge[t] + discharge[t];
''' + HCH_CONSTRAINTS

# Cylinder in the explicit SOC form of the previous code (for verification with MOSEK)
MODEL['hch_soc_cone'] = r'''
var charge{T} >= 0;
var discharge{T} >= 0;
var z{T};
var x1{T};
var x2{T};
var y{T};
minimize mismatch: sum{t in T} z[t];
s.t. eqx1{t in T}: x1[t] = signal[t]*(charge[t] - discharge[t]) - 0.5*z[t] + 0.5*(1 + signal[t]^2);
s.t. eqx2{t in T}: x2[t] = charge[t] + discharge[t];
s.t. eqy{t in T}: y[t] = 0.5*(1 - 2*signal[t]*(charge[t] - discharge[t]) + z[t] - signal[t]^2);
s.t. soc{t in T}: sqrt(x1[t]^2 + x2[t]^2) <= y[t];
''' + HCH_CONSTRAINTS

# Perspective reformulation (Proposition 1): two rotated cones per period,
# written as standard cones ||(2x, w - y)|| <= w + y  <=>  x^2 <= w y, w, y >= 0
MODEL['hch_per'] = r'''
var charge{T} >= 0;
var discharge{T} >= 0;
var u{T} >= 0, <= 1;                   # mode variable, auxiliary and continuous
var v{T} >= 0, <= 1;                   # v = 1 - u
var z{T};
var wd{T} >= 0;
var wc{T} >= 0;
var qd1{T}; var qd2{T}; var qd3{T};
var qc1{T}; var qc2{T}; var qc3{T};
minimize mismatch: sum{t in T} z[t];
s.t. def_v{t in T}: v[t] = 1 - u[t];
s.t. charge_mode{t in T}: charge[t] <= PCe*u[t];
s.t. discharge_mode{t in T}: discharge[t] <= PDe*v[t];
s.t. epigraph{t in T}: z[t] >= wd[t] + wc[t] - 2*signal[t]*(discharge[t] - charge[t]) + signal[t]^2;
s.t. def_qd1{t in T}: qd1[t] = 2*discharge[t];
s.t. def_qd2{t in T}: qd2[t] = wd[t] - v[t];
s.t. def_qd3{t in T}: qd3[t] = wd[t] + v[t];
s.t. cone_d{t in T}: sqrt(qd1[t]^2 + qd2[t]^2) <= qd3[t];      # (p^d)^2 <= w^d (1 - u)
s.t. def_qc1{t in T}: qc1[t] = 2*charge[t];
s.t. def_qc2{t in T}: qc2[t] = wc[t] - u[t];
s.t. def_qc3{t in T}: qc3[t] = wc[t] + u[t];
s.t. cone_c{t in T}: sqrt(qc1[t]^2 + qc2[t]^2) <= qc3[t];      # (p^c)^2 <= w^c u
''' + HCH_CONSTRAINTS


# =============================================================================
# SOLUTION OF ONE MODEL
# =============================================================================
def solve_model(kind, inst):
    """Solve model `kind` for the instance `inst` and return a dict with the results."""
    ampl = AMPL()
    try:
        ampl.eval(DATA)
        ampl.set['T'] = list(range(T))
        ampl.param['signal'] = {t: float(inst['signal'][t]) for t in range(T)}
        for name in ('t_step', 'S_lo', 'S_up', 'S_ini', 'PDe', 'PCe', 'nu_c', 'nu_d'):
            ampl.param[name] = float(inst[name])
        ampl.eval(MODEL[kind])
        solver = SOLVER[kind]
        ampl.setOption('solver', solver)
        ampl.setOption('solver_msg', 0)
        ampl.setOption(f'{solver}_options', (SOLVER_OPTIONS[solver] + ' ' + EXTRA_OPTIONS.get(kind, '')).strip())
        ampl.solve()
        status = ampl.get_value('solve_result')
        cpu_time = ampl.get_value('_solve_time')
        wall_time = ampl.get_value('_solve_elapsed_time')
        obj = ampl.get_objective('mismatch').value()
        cd = ampl.get_variable('charge').get_values().to_dict()
        dd = ampl.get_variable('discharge').get_values().to_dict()
    finally:
        ampl.close()
    C = np.array([cd[t] for t in range(T)], dtype=float)
    D = np.array([dd[t] for t in range(T)], dtype=float)
    if ROUND_DECIMALS is not None:
        C, D = np.round(C, ROUND_DECIMALS), np.round(D, ROUND_DECIMALS)
    prod = C * D
    res = dict(model=kind, solver=SOLVER[kind], status=status, obj=obj,
               tracking_error=float(np.sum((D - C - inst['signal'])**2)),
               hours=int(np.sum(np.abs(prod) > EPS_PROD)), dot=float(np.sum(prod)),
               cpu_time=cpu_time, wall_time=wall_time)
    if SAVE_PROFILES:
        res.update({f'pc{t}': C[t] for t in range(T)})
        res.update({f'pd{t}': D[t] for t in range(T)})
    return res


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        slurm = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    elif len(sys.argv) > 1:
        slurm = int(sys.argv[1]) - 1                     # python setpoint_tracking_letter.py 5
    else:
        slurm = 0

    data_storage = pd.read_csv(ESS_FILE)
    data_pv = pd.read_csv(PV_FILE)
    data_load = pd.read_csv(DEMAND_FILE).loc[:, 'value'].to_numpy(dtype=float)
    pv_profile = PV_PEAK_KW * np.array(ast.literal_eval(data_pv.loc[2*slurm, 'Power']), dtype=float)  # PV rows are the even rows
    signal = data_load - pv_profile                       # net demand to be tracked (kW)
    assert len(signal) == T, "the demand and PV profiles must have T values"

    n_dev = len(data_storage) if N_DEVICES is None else min(N_DEVICES, len(data_storage))
    rows = []
    t_start = time.time()
    for row in range(n_dev):
        S_lo = data_storage.loc[row, 'Emin']
        S_up = data_storage.loc[row, 'Emax']
        nu_c = data_storage.loc[row, 'eta_c']
        nu_d = data_storage.loc[row, 'eta_d']
        inst = dict(signal=signal, t_step=T_STEP, S_lo=S_lo, S_up=S_up, S_ini=data_storage.loc[row, 'E0'],
                    nu_c=nu_c, nu_d=nu_d,
                    PDe=min(data_storage.loc[row, 'PdMax'], (S_up - S_lo)*nu_d/T_STEP),     # effective limits
                    PCe=min(data_storage.loc[row, 'PcMax'], (S_up - S_lo)/(nu_c*T_STEP)))
        ref = None
        for kind in MODELS:
            res = solve_model(kind, inst)
            if res['status'] != 'solved':
                print(f"WARNING: slurm {slurm}, battery {row}, model {kind}: solve_result = {res['status']}", flush=True)
            if kind == 'miqp':
                ref = res['obj']
            res.update(slurm=slurm, device=row,
                       gap=(res['obj'] - ref)/abs(ref) if ref not in (None, 0) else np.nan)
            rows.append(res)

    df = pd.DataFrame(rows)
    first = ['slurm', 'device', 'model', 'solver', 'status', 'obj', 'tracking_error', 'gap', 'hours', 'dot', 'cpu_time', 'wall_time']
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(f'results_stp_s{slurm}.csv', index=False)

    # ---------------- summary of this task ----------------
    print(f"PV profile {slurm + 1}: {n_dev} batteries, {time.time() - t_start:.1f} s in total")
    line = [str(slurm)]
    for kind in MODELS:
        g = df[df.model == kind]
        print(f"{kind:13s} ({SOLVER[kind]:6s}): hours with p^c p^d > {EPS_PROD:g}: {int(g['hours'].sum()):4d}, "
              f"sum p^c p^d: {g['dot'].sum():10.4f}, CPU time: {g['cpu_time'].sum():.4f} s ({1000*g['cpu_time'].mean():.2f} ms/inst), "
              f"mean gap: {100*g['gap'].mean():+.4f}%, not solved: {int((g['status'] != 'solved').sum())}")
        line += [str(int(g['hours'].sum())), repr(float(g['dot'].sum())), repr(float(g['cpu_time'].sum()))]
    with open(f'{slurm}.res', 'w') as f:                  # columns: slurm, then (hours, dot, cpu_time) per model in MODELS order
        f.write(','.join(line) + '\n')
