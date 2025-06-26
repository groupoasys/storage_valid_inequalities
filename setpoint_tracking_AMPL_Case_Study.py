# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:45:50 2024

@author: Juan Miguel Morales González
"""

import numpy as np
import pandas as pd
import pdb
import ast
from amplpy import AMPL 
import time

###############################################################################
###############################################################################

# DATA AND MODELS DEFINITION IN AMPL

data = '''

# Indexing sets

set T;

# Parameters

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

model_mip = '''

# Variables

var charge{T} >= 0;
var discharge{T} >= 0;
var u{T} binary;

minimize mismatch: sum{t in T} (discharge[t] - charge[t] - signal[t])*(discharge[t] - charge[t] - signal[t]);
s.t. max_charge_t{t in T}: charge[t] <= PCe * u[t];
s.t. max_discharge_t{t in T}: discharge[t] <= PDe * (1-u[t]);
s.t. min_stor_level{t in T}: S_lo <= S_ini + t_step * sum{tt in T: tt<=t}(nu_c*charge[tt]-(1/nu_d)*discharge[tt]);
s.t. max_stor_level{t in T}: S_ini + t_step*sum{tt in T:tt<=t}(nu_c*charge[tt]-(1/nu_d)*discharge[tt]) <= S_up;
'''

model_POZO = '''

# Variables

var charge{T} >= 0;
var discharge{T} >= 0;

minimize mismatch: sum{t in T} (discharge[t] - charge[t] - signal[t])*(discharge[t] - charge[t] - signal[t]);
s.t. max_charge_discharge{t in T}: charge[t]/PCe + discharge[t]/PDe <= 1;
s.t. min_stor_level{t in T}: S_lo + (1/nu_d)*t_step * discharge[t] <= S_ini + t_step * sum{tt in T: tt<t}(nu_c*charge[tt]-(1/nu_d)*discharge[tt]);
s.t. max_stor_level{t in T}: S_ini + t_step*sum{tt in T:tt<t}(nu_c*charge[tt]-(1/nu_d)*discharge[tt]) <= S_up-nu_c *t_step*charge[t];
'''

model_MORALES = '''

# Parameters

# COEFFICIENTS OF LINEAR VALID INEQUALITIES

param c{T,T}; 
param bar_c{T};
param d{T,T}; 
param bar_d{T};
param PC{T};
param PD{T};
param bar_ro_c{T,T,T};
param bar_ro_d{T,T,T};


# Variables

var charge{t in T} >= 0, <= PC[t];
var discharge{t in T} >= 0, <= PD[t];
var z{T};
var x1{T};
var x2{T};
var y{T};

minimize mismatch: sum{t in T} z[t];
#s.t. max_charge_discharge{t in T}: charge[t]/PC[t] + discharge[t]/PD[t] <= 1;
#s.t. min_stor_level{t in T}: S_lo + (1/nu_d)*t_step * discharge[t] <= S_ini + t_step * sum{tt in T: tt<t}(nu_c*charge[tt]-(1/nu_d)*discharge[tt]);
#s.t. max_stor_level{t in T}: S_ini + t_step*sum{tt in T:tt<t}(nu_c*charge[tt]-(1/nu_d)*discharge[tt]) <= S_up-nu_c *t_step*charge[t];

# LINEAR VALID INEQUALITIES

# Complete set of proposed facet-defining inequalities
#s.t. val_ineq_c{t in T, bar_tau in T: bar_tau <= card(T)-t-1}:  sum{tau in T: tau <= bar_tau}(charge[t+tau]+bar_ro_c[t,tau,bar_tau] * discharge[t+tau]) <= sum{tau in T: tau <= bar_tau}(c[t,tau]);
#s.t. val_ineq_d{t in T, bar_tau in T: bar_tau <= card(T)-t-1}:  sum{tau in T: tau <= bar_tau}(discharge[t+tau]+bar_ro_d[t,tau,bar_tau] * charge[t+tau]) <= sum{tau in T: tau <= bar_tau}(d[t,tau]);

# Set of proposed facet-defining inequalities for t = 0 only and for bar_tau > 0 (to avoid redundancy with variable bounds)
s.t. val_ineq_c{t in T, bar_tau in T: t == 0 and bar_tau <= card(T)-t-1 and bar_tau > 0}:  sum{tau in T: tau <= bar_tau}(charge[t+tau]+bar_ro_c[t,tau,bar_tau] * discharge[t+tau]) <= sum{tau in T: tau <= bar_tau}(c[t,tau]);
s.t. val_ineq_d{t in T, bar_tau in T: t == 0 and bar_tau <= card(T)-t-1 and bar_tau > 0}:  sum{tau in T: tau <= bar_tau}(discharge[t+tau]+bar_ro_d[t,tau,bar_tau] * charge[t+tau]) <= sum{tau in T: tau <= bar_tau}(d[t,tau]);

# Set of proposed facet-defining inequalities for all t > 0, but for bar_tau = 0 only
#s.t. val_ineq_c_bar_tau_0{t in T, bar_tau in T: t > 0 and bar_tau == 0}:  sum{tau in T: tau <= bar_tau}(charge[t+tau]+bar_ro_c[t,tau,bar_tau] * discharge[t+tau]) <= sum{tau in T: tau <= bar_tau}(c[t,tau]);
#s.t. val_ineq_d_bar_tau_0{t in T, bar_tau in T: t > 0 and bar_tau == 0}:  sum{tau in T: tau <= bar_tau}(discharge[t+tau]+bar_ro_d[t,tau,bar_tau] * charge[t+tau]) <= sum{tau in T: tau <= bar_tau}(d[t,tau]);

# SOC CONSTRAINTS (parabolyc cylinder) TO MODEL THE DISJUNCTION IN THE OBJECTIVE FUNCTION

subject to eqx1{t in T}: x1[t] == signal[t]*(charge[t]-discharge[t]) - 0.5*z[t] + 0.5*(1+signal[t]**2);
subject to eqx2{t in T}: x2[t] == charge[t] + discharge[t];
subject to eqy{t in T}: y[t] == 0.5*(1-2*signal[t]*(charge[t]-discharge[t])+z[t]-signal[t]**2);
subject to soc{t in T}: sqrt(x1[t]**2 + x2[t]**2) <= y[t];
'''

###############################################################################
###############################################################################

# FUNCTIONS TO CONSTRUCT THE COEFFICIENTS OF VALID INEQUALITIES

#------------------------------------------------------------------------------

# Define the maximum and minimum energy storage level that can be reached at the beginning of each time period t in T

def reachable_levels(T, PCe, PDe, S_up, S_lo, S_ini, t_step, nu_c, nu_d):
    min_s = S_lo * np.ones((T,1))
    max_s = S_up * np.ones((T,1))
    min_s[0] = S_ini
    max_s[0] = S_ini

    for t in range(1,T):
        min_s[t] = max(S_lo, min_s[t-1]-PDe*t_step/nu_d)
        max_s[t] = min(S_up, max_s[t-1]+ nu_c* t_step * PCe)
    return min_s, max_s

# Define coefficients:


def VI_coefficients(T, PCe, PDe, S_up, S_lo, S_ini, t_step, nu_c, nu_d, min_s, max_s):
    bar_c = np.zeros((T,1))
    bar_d = np.zeros((T,1))
    PC = PCe * np.ones((T,1))
    PD = PDe * np.ones((T,1))
    c = np.zeros((T, T))
    d = np.zeros((T, T))
    ro_c = np.zeros((T, T, T))
    ro_d = np.zeros((T, T, T))
    bar_ro_c = np.zeros((T, T, T))
    bar_ro_d = np.zeros((T, T, T))     

    for t in range(T):
        bar_c[t] = min(PCe, max((S_up-S_lo)/(t_step*nu_c) - t*PCe, 0))
        bar_d[t] = min(PDe, max((S_up-S_lo)*nu_d/t_step - t*PDe, 0))

        for bar_tau in range(T-t):
            c[t,bar_tau] = min(PCe, max((S_up-min_s[t].item())/(t_step*nu_c) - bar_tau*PCe, 0))
            d[t,bar_tau] = min(PDe, max((max_s[t].item()-S_lo)*nu_d/t_step - bar_tau*PDe, 0))
        
        PC[t] = c[t,0]
        PD[t] = d[t,0]
        
   
    for t in range(T):
        for bar_tau in range(T-t):        
            for tau in range(bar_tau+1):
                ro_c[t,tau,bar_tau] = max(-PD[t+tau].item()/(nu_c*nu_d), sum(c[t,j].item() for j in range(tau, bar_tau+1))-sum(bar_c[j].item() for j in range(bar_tau-tau)))
                ro_d[t,tau,bar_tau] = max(-PC[t+tau].item()*nu_c*nu_d, sum(d[t,j].item() for j in range(tau, bar_tau+1))-sum(bar_d[j].item() for j in range(bar_tau-tau)))              
                if ro_c[t,tau,bar_tau] < 0:
                    bar_ro_c[t,tau,bar_tau] = -1/(nu_c*nu_d)
                else:
                    bar_ro_c[t,tau,bar_tau] = ro_c[t,tau,bar_tau]/(PD[t+tau].item())
                if ro_d[t,tau,bar_tau] < 0:
                    bar_ro_d[t,tau,bar_tau] = -nu_c*nu_d
                else:
                    bar_ro_d[t,tau,bar_tau] = ro_d[t,tau,bar_tau]/(PC[t+tau].item())
                
    bar_ro_c_dict = {(t,tau,bar_tau): bar_ro_c[t,tau,bar_tau] for t in range(T) for tau in range(T) for bar_tau in range(T)}
    bar_ro_d_dict = {(t,tau,bar_tau): bar_ro_d[t,tau,bar_tau] for t in range(T) for tau in range(T) for bar_tau in range(T)}

    return bar_c, bar_d, PC, PD, c, d, bar_ro_c_dict, bar_ro_d_dict
        
###############################################################################
###############################################################################

# MODELS RESOLUTION

def solve_storage(modelo):
    ampl = AMPL()
    ampl.reset()
    ampl.eval(data)
    ampl.set['T'] = range(T)
    ampl.param['signal'] = signal
    ampl.param['t_step'] = t_step
    ampl.param['S_lo'] = S_lo
    ampl.param['S_up'] = S_up
    ampl.param['S_ini'] = S_ini
    ampl.param['PDe'] = PDe
    ampl.param['PCe'] = PCe
    ampl.param['nu_c'] = nu_c
    ampl.param['nu_d'] = nu_d
#
    if modelo == 'mip':
        ampl.eval(model_mip)
        ampl.setOption('solver', 'gurobi')
        start_time = time.time()
        ampl.solve(gurobi_options="threads=1")
        end_time = time.time()
        work_time = end_time - start_time
    elif modelo == 'pozo':
        ampl.eval(model_POZO)
        ampl.setOption('solver', 'gurobi')
        start_time = time.time()
        ampl.solve(gurobi_options="threads=1")
        end_time = time.time()
        work_time = end_time - start_time
    else:
        # Define maximum and minimum energy levels that can be reached in the storage at the beginning of each time period 
    
        min_s, max_s = reachable_levels(T, PCe, PDe, S_up, S_lo, S_ini, t_step, nu_c, nu_d)

        # COMPUTATION OF VALID INEQUALITIES' COEFFICIENTS.
 
        bar_c, bar_d, PC, PD, c, d, bar_ro_c, bar_ro_d = VI_coefficients(T, PCe, PDe, S_up, S_lo, S_ini, t_step, nu_c, nu_d, min_s, max_s)

        ampl.eval(model_MORALES)
        ampl.param['PD'] = PD
        ampl.param['PC'] = PC
        ampl.param['c'] = c
        ampl.param['d'] = d
        ampl.param['bar_c'] = bar_c
        ampl.param['bar_d'] = bar_d
        ampl.param['bar_ro_c'] = bar_ro_c
        ampl.param['bar_ro_d'] = bar_ro_d
             
        ampl.setOption('solver', 'mosek')
        start_time = time.time()
        ampl.solve(mosek_options="threads=1")
        end_time = time.time()
        work_time = end_time - start_time
    carga = np.round(ampl.get_variable('charge').to_pandas(),2)
    descarga = np.round(ampl.get_variable('discharge').to_pandas(),2)
    valor_objetivo = ampl.get_value('mismatch')
    return carga, descarga, valor_objetivo, work_time

###############################################################################
###############################################################################

# WE READ POZO'S CSV FILES

data_storage = pd.read_csv("ESS_data_SPTP.csv") # Storage technical parameters and initial condition
data_pv = pd.read_csv("PV_and_Wind_data_scenarios.csv") # PV profiles
data_demand = pd.read_csv("demand_profile.csv")
data_load = data_demand.loc[:,"value"].tolist()

T = 24 # Planning horizon (number of time periods)
t_step = 1 # Time resolution - step

mat_mip = []
mat_pozo = []
mat_morales = []
failures_mip = 0
failures_pozo = 0
failures_morales = 0
acc_time_mip = 0
acc_time_pozo = 0
acc_time_morales = 0

sum_failures_mip = 0
sum_failures_pozo = 0
sum_failures_morales = 0


for row in range(len(data_storage)):
    S_lo = data_storage.loc[row, "Emin"] # Minimum energy level
    S_up = data_storage.loc[row, "Emax"] # Maximum energy level
    S_ini = data_storage.loc[row, "E0"] # Initial energy level
    PD_MAX = data_storage.loc[row, "PdMax"] # Maximum rate of power discharge
    PC_MAX = data_storage.loc[row, "PcMax"] # Maximum rate of power charge
    nu_c = data_storage.loc[row, "eta_c"] # Charging efficiency
    nu_d = data_storage.loc[row, "eta_d"] # Discharging efficiency
    n_pv_list = ast.literal_eval(data_pv.loc[2*row, "Power"]) # normalized pv production as a list
    pv_list = [x * 35 for x in n_pv_list] # pv production in kW as a list
    signal =  np.subtract(data_load, pv_list).tolist()
    PDe = min(PD_MAX, (S_up-S_lo)*nu_d/t_step) # Effective maximum rate of power discharge
    PCe = min(PC_MAX, (S_up-S_lo)/(nu_c*t_step)) # Effective maximum rate of power charge


#------------------------------------------------------------------------------

    c_mip,d_mip,obj_mip,work_time_mip = solve_storage('mip')
   
    charge_list_mip = c_mip['charge.val'].tolist()
    discharge_list_mip = d_mip['discharge.val'].tolist()
    dot_mip = np.dot(charge_list_mip,discharge_list_mip)
    dot_mip_elementwise = [x * y for x, y in zip(charge_list_mip, discharge_list_mip)]     
    mat_mip.append([obj_mip, dot_mip] + charge_list_mip + discharge_list_mip +[work_time_mip])
    sum_failures_mip = sum_failures_mip + dot_mip
    positive_count_mip = sum(1 for num in dot_mip_elementwise if abs(num) > 1e-4)
    failures_mip += positive_count_mip
    acc_time_mip += work_time_mip          
    
#------------------------------------------------------------------------------

    c_pozo,d_pozo,obj_pozo,work_time_pozo = solve_storage('pozo')
   
    charge_list_pozo = c_pozo['charge.val'].tolist()
    discharge_list_pozo = d_pozo['discharge.val'].tolist()
    dot_pozo = np.dot(charge_list_pozo,discharge_list_pozo) 
    dot_pozo_elementwise = [x * y for x, y in zip(charge_list_pozo, discharge_list_pozo)]    
    mat_pozo.append([obj_pozo, dot_pozo] + charge_list_pozo + discharge_list_pozo + [work_time_pozo])
    sum_failures_pozo = sum_failures_pozo + dot_pozo
    positive_count_pozo = sum(1 for num in dot_pozo_elementwise if abs(num) > 1e-4)
    failures_pozo += positive_count_pozo
    acc_time_pozo += work_time_pozo

#------------------------------------------------------------------------------

    c_morales,d_morales,obj_morales,work_time_morales = solve_storage('morales')
    
    charge_list_morales = c_morales['charge.val'].tolist()
    discharge_list_morales = d_morales['discharge.val'].tolist()
    dot_morales = np.dot(charge_list_morales,discharge_list_morales) 
    dot_morales_elementwise = [x * y for x, y in zip(charge_list_morales, discharge_list_morales)]    
    mat_morales.append([obj_morales, dot_morales] + charge_list_morales + discharge_list_morales + [work_time_morales])
    sum_failures_morales = sum_failures_morales + dot_morales
    positive_count_morales = sum(1 for num in dot_morales_elementwise if abs(num) > 1e-4)
    failures_morales += positive_count_morales
    acc_time_morales += work_time_morales
  

df_mip = pd.DataFrame(mat_mip)
df_mip.columns = ['mismatch','dot_prod'] + ['pc'+str(t) for t in range(24)] + ['pd'+str(t) for t in range(24)] + ['time']
df_mip.to_csv('results_mip.csv')

df_pozo = pd.DataFrame(mat_pozo)
df_pozo.columns = ['mismatch','dot_prod'] + ['pc'+str(t) for t in range(24)] + ['pd'+str(t) for t in range(24)] + ['time']
df_pozo.to_csv('results_pozo.csv')

df_morales = pd.DataFrame(mat_morales)
df_morales.columns = ['mismatch','dot_prod'] + ['pc'+str(t) for t in range(24)] + ['pd'+str(t) for t in range(24)] + ['time']
df_morales.to_csv('results_morales.csv')

print("MIP number of hours with simultaneous charging and discharging:", failures_mip)

print("MIP cumulative scalar product of charging and discharging variable vectors:", sum_failures_mip)

print("MIP cumulative elapsed time:", acc_time_mip)
    
print("POZO number of hours with simultaneous charging and discharging:", failures_pozo)

print("POZO cumulative scalar product of charging and discharging variable vectors:", sum_failures_pozo)

print("POZO cumulative elapsed time:", acc_time_pozo)

print("MORALES number of hours with simultaneous charging and discharging:", failures_morales)

print("MORALES cumulative scalar product of charging and discharging variable vectors:", sum_failures_morales)

print("MORALES cumulative elapsed time:", acc_time_morales)
    