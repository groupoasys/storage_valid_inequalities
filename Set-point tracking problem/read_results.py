"""
Aggregate the results of setpoint_tracking_letter.py over all the SLURM tasks
(files results_stp_s*.csv) and print the table of the letter.

Columns of the table (as in Pozo (2023)):
  hours (%)     percentage of hours with p^c_t p^d_t > 1e-4 kW^2
  pcpd (kW^2)   average value of p^c . p^d per instance
  gap (%)       average relative difference between the objective of the model and that of the MIQP
                (for the relaxations: the lower bound they provide)
  time (ms)     average CPU time of the solver per instance
  dTime (%)     reduction in total CPU time with respect to the MIQP: 100 (1 - time_model / time_MIQP)
"""
import glob
import sys

import numpy as np
import pandas as pd

files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'results_stp_s*.csv'))
if not files:
    sys.exit('no results_stp_s*.csv files found')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
T = 24
models = list(dict.fromkeys(df.model))                 # in the order in which they were solved
n = df[df.model == models[0]].shape[0]
print(f"{len(files)} PV profiles, {n} instances, {n*T} hours")

not_solved = df[df.status != 'solved']
if len(not_solved):
    print(f"WARNING: {len(not_solved)} solves without solve_result = 'solved':")
    print(not_solved.groupby(['model', 'status']).size())

ref_time = df[df.model == 'miqp'].cpu_time.sum() if 'miqp' in models else np.nan
rows = []
for m in models:
    g = df[df.model == m]
    t = 1000*g['cpu_time']
    rows.append(dict(model=m, solver=g['solver'].iloc[0], instances=len(g),
                     hours_pct=100*g['hours'].sum()/(len(g)*T), hours=int(g['hours'].sum()),
                     inst_with_simult=int((g['hours'] > 0).sum()), pcpd=g['dot'].mean(),
                     gap_pct=100*g['gap'].mean(), time_ms=t.mean(), time_median_ms=t.median(), time_max_ms=t.max(),
                     dTime_pct=100*(1 - g['cpu_time'].sum()/ref_time)))
tab = pd.DataFrame(rows).set_index('model')
pd.set_option('display.width', 200)
print(tab.round(4).to_string())

# relaxations: equal bounds?  (the perspective constraints can only improve on the cylinder where
# p^d + p^c > min(PDe, PCe), and not at all if PDe = PCe)
P = df.pivot_table(index=['slurm', 'device'], columns='model', values='obj')
scale = P['miqp'].abs().clip(lower=1) if 'miqp' in P else 1
for a, b in (('hch_soc', 'hch_per'), ('hch_soc', 'hch_soc_cone')):
    if a in P and b in P:
        d = (P[b] - P[a])/scale
        print(f"{b} - {a}: relative difference of the objectives: max {d.max():+.2e}, min {d.min():+.2e}; "
              f"instances with {b} tighter by more than 1e-6: {int((d > 1e-6).sum())}")

print("\nLaTeX rows (Formulation & hours % & p^c.p^d & dTime %):")
names = {'hch_qp': 'HCH-QP', 'hch_soc': 'HCH-SOC', 'hch_per': 'HCH-PER', 'hch_soc_cone': 'HCH-SOC (cone)', 'miqp_plain': 'MIQP$^{-}$'}
for m in models:
    if m == 'miqp':
        continue
    r = tab.loc[m]
    print(f"{names.get(m, m)} & {r.hours_pct:.2f} & {r.pcpd:.2f} & {r.dTime_pct:.2f} \\\\")
