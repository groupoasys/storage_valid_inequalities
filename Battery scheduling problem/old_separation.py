"""Verbatim port of the separation routines of Scheduling_with_cuts.py
(functions reachable_levels ... separation_algorithm_discharge), used only
for the numerical comparison with the new separator.  Periods are 0-indexed
here, as in the original script; the LP point is passed as pandas DataFrames
with the original column names."""
import numpy as np

def reachable_levels(T, PCe, PDe, S_up, S_lo, S_ini, t_step, nu_c, nu_d):
    min_s = S_lo * np.ones((T,1))
    max_s = S_up * np.ones((T,1))
    min_s[0] = max(S_lo, S_ini-PDe*t_step/nu_d)
    max_s[0] = min(S_up, S_ini+ nu_c* t_step * PCe)
    PC = PCe * np.ones((T,1))
    PD = PDe * np.ones((T,1))
    PC[0] = (max_s[0]-S_ini)/(t_step*nu_c)
    PD[0] = (S_ini-min_s[0])*nu_d/t_step

    for t in range(1,T):
        min_s[t] = max(S_lo, min_s[t-1]-PDe*t_step/nu_d)
        max_s[t] = min(S_up, max_s[t-1]+ nu_c* t_step * PCe)
        PC[t] = min(PCe, (S_up-min_s[t-1])/(t_step*nu_c))
        PD[t] = min(PDe, (max_s[t-1]-S_lo)*nu_d/t_step)
    return min_s, max_s, PC, PD

def evaluate_greedy(Omega, l, k, s_start, PC, PD, t_step, nu_c, nu_d, S_up, S_lo):
    """
    Simula la trayectoria greedy (Lema 1) para un conjunto Omega en la ventana [l, k].
    Retorna el valor de f(Omega), la trayectoria s_t y los flujos x_t.
    """
    s_traj = {l-1: s_start}
    x_traj = {}
    f_val = 0.0

    for t in range(l, k + 1):
        if t in Omega:
            # Intentar cargar a máxima potencia posible en este periodo
            max_charge_energy = PC[t][0] * t_step * nu_c
            x = min(max_charge_energy, S_up - s_traj[t-1])
            f_val += x / (t_step * nu_c)
        else:
            # Intentar descargar a máxima potencia
            max_discharge_energy = PD[t][0] * t_step / nu_d
            x = max(-max_discharge_energy, S_lo - s_traj[t-1])

        x_traj[t] = x
        s_traj[t] = s_traj[t-1] + x

    return f_val, s_traj, x_traj

def check_conditions_and_decompose(Omega, l, k, s_traj, PC, t_step, nu_c, S_up, S_lo):
    """
    Verifica las Condiciones 1 y 2 de INSEPARABILIDAD para los periodos
    intermedios t in [l, k-1] fuera de Omega en orden cronologico.

    1. Condicion 2 (Memory Preservation): s_t > S_lo.
       Si se viola (s_t <= S_lo): Reset de memoria en t.
       Accion: Dividir en Omega_left (t' < t) y Omega_right (t' > t).
       AMBOS bloques se conservan para seguir explorandose.

    2. Condicion 1 (Headroom Limitation): S_up - s_t < Carga_Futura_En_Omega.
       Si se viola (S_up - s_t >= Carga_Futura): Margen de almacenamiento holgado.
       Accion: Conservar SOLO Omega_left (t' < t). El bloque derecho SE DISPORTA.
    """
    if not Omega:
        return False, []

    t_max_omega = max(Omega)
    # Periodos intermedios t en [l, t_max_omega - 1] que NO pertenecen a Omega
    check_periods = [t for t in range(l, t_max_omega) if t not in Omega]

    for t in check_periods:
        # 1. Verificar primero la Condicion 2 (Memory Preservation / Reset de memoria)
        cond2_holds = (s_traj[t] > S_lo + 1e-6)

        if not cond2_holds:
            # Violacion de Condicion 2: Nos quedamos con el bloque izquierdo Y el derecho
            omega_left = {tp for tp in Omega if tp < t}
            omega_right = {tp for tp in Omega if tp > t}

            sub_intervals = []
            if omega_left:
                sub_intervals.append(omega_left)
            if omega_right:
                sub_intervals.append(omega_right)

            return True, sub_intervals

        # 2. Verificar la Condicion 1 (Upper-bound headroom limitation)
        future_charge_capacity = sum(PC[tp][0] * t_step * nu_c for tp in Omega if tp > t)
        cond1_holds = (S_up - s_traj[t]) < (future_charge_capacity - 1e-6)

        if not cond1_holds:
            # Violacion de Condicion 1: Margen holgado
            # Nos quedamos UNICAMENTE con el bloque izquierdo (el derecho se descarta)
            omega_left = {tp for tp in Omega if tp < t}

            sub_intervals = []
            if omega_left:
                sub_intervals.append(omega_left)

            return True, sub_intervals

    # Si ambas condiciones se cumplieron para todos los periodos t intermedios, NO hay ruptura
    return False, []

def evaluate_and_prune(Omega, l, k, s_start, Gamma, epsilon, C_cuts, Q, c_star, d_star, u_star, PC, PD, t_step, nu_c, nu_d, S_up, S_lo):
    """
    Subrutina de Evaluación y Poda (Algorithm 2)
    Ejecuta evaluate_greedy UNA SOLA VEZ y gestiona tanto la separabilidad
    como la generación de cortes y la poda individual.
    """
    # 1. EVALUACIÓN GREEDY ÚNICA
    f_base, s_base_traj, _ = evaluate_greedy(Omega, l, k, s_start, PC, PD, t_step, nu_c, nu_d, S_up, S_lo)

    # 2. VERIFICACIÓN DE CONDICIONES DE SEPARABILIDAD
    violated_cond, sub_intervals = check_conditions_and_decompose(Omega, l, k, s_base_traj, PC, t_step, nu_c, S_up, S_lo)

    if violated_cond:
        # Si la desigualdad es separable, añadimos los sub-intervalos resultantes a Q y terminamos
        for sub_omega in sub_intervals:
            if (sub_omega & set(Gamma)) and (sub_omega not in Q):
                Q.append(sub_omega)
        return # Salimos de la subrutina sin hacer cálculos de pi_t ni cortes

    # 3. CÁLCULO DE COEFFICIENTES Y FORTALECIMIENTO (Solo si NO es separable)
    pi = {}
    alpha = {}
    beta = {}

    for t in sorted(list(Omega), reverse=True):
        Omega_minus_t = set(Omega) - {t}
        f_minus, _, _ = evaluate_greedy(Omega_minus_t, t, k, s_base_traj[t-1], PC, PD, t_step, nu_c, nu_d, S_up, S_lo)

        prefix_f = sum((s_base_traj[tau] - s_base_traj[tau-1]) / (t_step * nu_c) for tau in Omega if tau < t)
        f_minus_total = prefix_f + f_minus

        pi[t] = f_minus_total - f_base

        tol = 1e-6
        max_d_coef = PD[t][0] / (nu_c * nu_d)

        if (abs(pi[t] - max_d_coef) < tol) or (pi[t] > tol and (d_star[t] / (nu_c * nu_d)) < pi[t] * (1 - c_star[t]/PC[t][0])):
            alpha[t] = 1.0 / (nu_c * nu_d)
            beta[t] = 0.0
        else:
            alpha[t] = 0.0
            beta[t] = pi[t]

    # 4. EVALUACIÓN DE LA VIOLACIÓN Y GENERACIÓN DE CORTE
    lhs = sum(c_star[t] - alpha[t] * d_star[t] - beta[t] * (1 - u_star[t]) for t in Omega)
    V = lhs - f_base

    if V > epsilon:
        C_cuts.append({'Omega': set(Omega), 'alpha': alpha, 'beta': beta, 'rhs': f_base, 'V': V, 'l': l, 'k': k})
    else:
        # 5. PODA INDIVIDUAL SI NO HAY VIOLACIÓN SUFICIENTE
        max_val = -float('inf')
        M_candidates = []
        for t in Omega:
            val = (1.0 / nu_d) * d_star[t] - nu_c * c_star[t]
            if val > max_val + 1e-6:
                max_val = val
                M_candidates = [t]
            elif abs(val - max_val) <= 1e-6:
                M_candidates.append(t)

        t_star = max(M_candidates)
        new_Omega = set(Omega) - {t_star}
        if new_Omega not in Q and new_Omega:
            Q.append(new_Omega)

def separation_algorithm(c_lp, d_lp, soc_lp, u_lp, min_s, PC, PD, T, t_step, nu_c, nu_d, S_up, S_lo, S_ini, epsilon=1e-3):
    c_star = c_lp['charge.val'].values
    d_star = d_lp['discharge.val'].values
    s_star = soc_lp['s.val'].values
    u_star = u_lp['u.val'].values

    Gamma = [t for t in range(T) if c_star[t] > epsilon and d_star[t] > epsilon]
    if not Gamma:
        return []

    C_cuts = []
    Q = []

    # Inicialización por ventanas de cuello de botella
    for tau in Gamma:
        l = tau
        while l > 0 and abs(s_star[l-1] - S_lo) > 1e-4:
            l -= 1

        k = tau
        while k < T - 1 and abs(s_star[k] - S_up) > 1e-4:
            k += 1

        Omega = set(range(l, k + 1))
        if Omega not in Q:
            Q.append(Omega)

    Q.sort(key=lambda x: min(x))

    while Q:
        Omega = Q.pop(0)

        if not (Omega & set(Gamma)) or len(Omega) <= 1:
            continue

        l = min(Omega)
        k = max(Omega)
        s_start = S_ini if l == 0 else min_s[l-1][0]

        # Pre-poda de descarga máxima
        Omega = Omega - {t for t in Omega if abs(d_star[t] - PD[t][0]) < 1e-4}

        # Delegamos toda la evaluación, separabilidad y poda a la subrutina
        evaluate_and_prune(Omega, l, k, s_start, Gamma, epsilon, C_cuts, Q, c_star, d_star, u_star, PC, PD, t_step, nu_c, nu_d, S_up, S_lo)

    return C_cuts
# Llama a la función usando las salidas de tu LP:
# cortes_encontrados = separation_algorithm(c_lp, d_lp, soc_lp, u_lp, min_s, PC, PD, T, t_step, nu_c, nu_d, S_up, S_lo, S_ini)

def evaluate_greedy_discharge(Omega, l, k, s_start, PC, PD, t_step, nu_c, nu_d, S_up, S_lo):
    """
    Simula la trayectoria greedy para descarga (Lema 1 especular).
    """
    s_traj = {l-1: s_start}
    f_val = 0.0

    for t in range(l, k + 1):
        if t in Omega:
            # Intentar descargar a máxima potencia
            max_discharge_energy = PD[t][0] * t_step / nu_d
            x = max(-max_discharge_energy, S_lo - s_traj[t-1])
            f_val += -x * nu_d / t_step
        else:
            # Intentar cargar a máxima potencia
            max_charge_energy = PC[t][0] * t_step * nu_c
            x = min(max_charge_energy, S_up - s_traj[t-1])

        s_traj[t] = s_traj[t-1] + x

    return f_val, s_traj


def check_conditions_and_decompose_discharge(Omega, l, k, s_traj, PD, t_step, nu_d, S_up, S_lo):
    """
    Verifica las Condiciones 1 y 2 de INSEPARABILIDAD para descarga.
    """
    if not Omega:
        return False, []

    t_max_omega = max(Omega)
    check_periods = [t for t in range(l, t_max_omega) if t not in Omega]

    for t in check_periods:
        # 1. Condicion 2: Memory Preservation para descarga (s_t < S_up)
        # Si se viola (s_t >= S_up), la batería se llenó y resetea la memoria de descarga.
        cond2_holds = (s_traj[t] < S_up - 1e-6)

        if not cond2_holds:
            omega_left = {tp for tp in Omega if tp < t}
            omega_right = {tp for tp in Omega if tp > t}
            return True, [comp for comp in [omega_left, omega_right] if comp]

        # 2. Condicion 1: Headroom / Reserva (Energía disponible < Descarga futura)
        # Si se viola, hay energía de sobra y el bloque derecho no limita.
        future_discharge = sum(PD[tp][0] * t_step / nu_d for tp in Omega if tp > t)
        cond1_holds = (s_traj[t] - S_lo) < (future_discharge - 1e-6)

        if not cond1_holds:
            omega_left = {tp for tp in Omega if tp < t}
            return True, [omega_left] if omega_left else []

    return False, []


def evaluate_and_prune_discharge(Omega, l, k, s_start, Gamma, epsilon, C_cuts, Q, c_star, d_star, u_star, PC, PD, t_step, nu_c, nu_d, S_up, S_lo):
    """
    Subrutina de Evaluación y Poda para Descargas.
    """
    f_base, s_base_traj = evaluate_greedy_discharge(Omega, l, k, s_start, PC, PD, t_step, nu_c, nu_d, S_up, S_lo)

    violated_cond, sub_intervals = check_conditions_and_decompose_discharge(Omega, l, k, s_base_traj, PD, t_step, nu_d, S_up, S_lo)

    if violated_cond:
        for sub_omega in sub_intervals:
            if (sub_omega & set(Gamma)) and (sub_omega not in Q):
                Q.append(sub_omega)
        return

    pi = {}
    alpha = {}
    beta = {}

    for t in sorted(list(Omega), reverse=True):
        Omega_minus_t = set(Omega) - {t}
        f_minus, _ = evaluate_greedy_discharge(Omega_minus_t, t, k, s_base_traj[t-1], PC, PD, t_step, nu_c, nu_d, S_up, S_lo)

        # Calcular el prefijo acumulado para descargas
        prefix_f = sum(-(s_base_traj[tau] - s_base_traj[tau-1]) * nu_d / t_step for tau in Omega if tau < t)
        f_minus_total = prefix_f + f_minus

        pi[t] = f_minus_total - f_base

        tol = 1e-6
        max_c_coef = PC[t][0] * nu_c * nu_d  # Ajuste simétrico del coeficiente máximo

        if (abs(pi[t] - max_c_coef) < tol) or (pi[t] > tol and (c_star[t] * nu_c * nu_d) < pi[t] * (1 - d_star[t]/PD[t][0])):
            alpha[t] = nu_c * nu_d
            beta[t] = 0.0
        else:
            alpha[t] = 0.0
            beta[t] = pi[t]

    # LHS: Suma de descargas - Suma de cargas penalizadas - penalización por estado (u_t = 1 es carga)
    lhs = sum(d_star[t] - alpha[t] * c_star[t] - beta[t] * u_star[t] for t in Omega)
    V = lhs - f_base

    if V > epsilon:
        C_cuts.append({'Omega': set(Omega), 'alpha': alpha, 'beta': beta, 'rhs': f_base, 'V': V, 'l': l, 'k': k})
    else:

        max_val = -float('inf')
        M_candidates = []
        for t in Omega:
            # Criterio invertido: maximizar la carga neta
            val = nu_c * c_star[t] - (1.0 / nu_d) * d_star[t]
            if val > max_val + 1e-6:
                max_val = val
                M_candidates = [t]
            elif abs(val - max_val) <= 1e-6:
                M_candidates.append(t)

        t_star = max(M_candidates)
        new_Omega = set(Omega) - {t_star}
        if new_Omega not in Q and new_Omega:
            Q.append(new_Omega)


def separation_algorithm_discharge(c_lp, d_lp, soc_lp, u_lp, max_s, PC, PD, T, t_step, nu_c, nu_d, S_up, S_lo, S_ini, epsilon=1e-3):
    """
    Main Loop especular para desigualdades de descarga.
    """
    c_star = c_lp['charge.val'].values
    d_star = d_lp['discharge.val'].values
    s_star = soc_lp['s.val'].values
    u_star = u_lp['u.val'].values

    Gamma = [t for t in range(T) if c_star[t] > epsilon and d_star[t] > epsilon]
    if not Gamma:
        return []

    C_cuts = []
    Q = []

    for tau in Gamma:
        # Buscar l hacia atrás hasta que la batería esté completamente LLENA
        l = tau
        while l > 0 and abs(s_star[l-1] - S_up) > 1e-4:
            l -= 1

        # Buscar k hacia adelante hasta que la batería esté completamente VACÍA
        k = tau
        while k < T - 1 and abs(s_star[k] - S_lo) > 1e-4:
            k += 1

        Omega = set(range(l, k + 1))
        if Omega not in Q:
            Q.append(Omega)

    Q.sort(key=lambda x: min(x))

    while Q:
        Omega = Q.pop(0)

        if not (Omega & set(Gamma)) or len(Omega) <= 1:
            continue

        l = min(Omega)
        k = max(Omega)

        # Para descargas, el peor caso es arrancar desde el máximo SOC acumulable
        s_start = S_ini if l == 0 else max_s[l-1][0]

        # Pre-poda de carga máxima
        Omega = Omega - {t for t in Omega if abs(c_star[t] - PC[t][0]) < 1e-4}
        if not (Omega & set(Gamma)):
            continue

        evaluate_and_prune_discharge(Omega, l, k, s_start, Gamma, epsilon, C_cuts, Q, c_star, d_star, u_star, PC, PD, t_step, nu_c, nu_d, S_up, S_lo)

    return C_cuts
