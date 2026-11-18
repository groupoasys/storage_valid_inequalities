#!/usr/bin/env bash
# =============================================================================
# Experimentos de scheduling (Seccion 4): un trabajo por escenario de precios.
#
# Uso (desde la carpeta que contiene el codigo y los datos):
#     sbatch envia_scheduling.sh                    # CP,  T = 24
#     sbatch envia_scheduling.sh 1 1                # CP1, T = 24
#     sbatch envia_scheduling.sh 1 0 efficacy       # CP,  T = 24, cortes ordenados por eficacia
#     sbatch --time=0-03:00:00 envia_scheduling.sh 3 0     # CP, T = 72
#
# Argumentos:  N_DAYS (1 = 24 h, 3 = 72 h)   CUTS_PER_PERIOD (0 = CP, 1 = CP1)   CUT_SCORE (violation | efficacy)
#
# Cada configuracion escribe en su propia carpeta (run_T24_CP, run_T24_CP1, ...),
# asi que CP y CP1 no se pisan los resultados. Los valores de N_DAYS,
# CUTS_PER_PERIOD y CUT_SCORE se fijan aqui, no en el fichero .py: lo que haya
# escrito en el .py para esas tres opciones se ignora.
# =============================================================================

# The name to show in queue lists for this job:
#SBATCH -J scheduling_cuts

# Number of desired cores (all in same node). El script fija Threads = 1 en Gurobi:
#SBATCH --cpus-per-task=1

# Amount of RAM needed for this job (8 GB, como indica el articulo):
#SBATCH --mem=8gb

# The time the job will be running. Con T = 24 sobran 30 minutos; con T = 72 y
# RUN_MIP_PLAIN = True hay que pedir mas (ver el ejemplo de arriba):
#SBATCH --time=0-00:30:00

# If you need nodes with special features uncomment the desired constraint line.
# IMPORTANTE: usar el mismo tipo de maquina que en las ejecuciones de la v11,
# para que los tiempos sean comparables con los de MILP y HCH-LP que no se repiten.
# * to request only the machines with 80 cores and 2TB of RAM
##SBATCH --constraint=bigmem
# * to request only machines with 16 cores and 64GB with InfiniBand network
#SBATCH --constraint=cal
# * to request only machines with 24 cores and Gigabit network
##SBATCH --constraint=slim

# Set output and error files
#SBATCH --error=zzz.%A.%a.err
#SBATCH --output=zzz.%A.%a.out

# ARRAY JOB: un trabajo por escenario de precios (SLURM_ARRAY_TASK_ID de 1 a 10)
#SBATCH --array=1-10

# ----------------------------------------------------------------------------
# Configuracion de esta ejecucion
# ----------------------------------------------------------------------------
N_DAYS=${1:-1}
CUTS_PER_PERIOD=${2:-0}
CUT_SCORE=${3:-violation}

SCRIPT=Scheduling_with_cuts_gurobipy_v2.py
SRC_DIR=${SLURM_SUBMIT_DIR:-$PWD}

case "$N_DAYS" in 1|3) ;; *) echo "ERROR: N_DAYS debe ser 1 o 3 (recibido: $N_DAYS)"; exit 1 ;; esac
case "$CUTS_PER_PERIOD" in 0) VARIANT=CP ;; 1) VARIANT=CP1 ;; *) echo "ERROR: CUTS_PER_PERIOD debe ser 0 o 1 (recibido: $CUTS_PER_PERIOD)"; exit 1 ;; esac
case "$CUT_SCORE" in violation) ;; efficacy) VARIANT=${VARIANT}_efficacy ;; *) echo "ERROR: CUT_SCORE debe ser violation o efficacy (recibido: $CUT_SCORE)"; exit 1 ;; esac
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then echo "ERROR: este script se lanza con sbatch (trabajo en array)"; exit 1; fi

RUN_DIR=$SRC_DIR/run_T$((24 * N_DAYS))_${VARIANT}

# Sin los ficheros de datos, el .py resuelve instancias sinteticas sin dar error: mejor parar aqui.
for f in $SCRIPT storage_cuts.py old_separation.py ESS_data_SPTP.csv scenarios_prices.xlsx; do
    if [ ! -f "$SRC_DIR/$f" ]; then echo "ERROR: falta $f en $SRC_DIR"; exit 1; fi
done

# Escalonar el arranque de los trabajos (licencias de Gurobi, sistema de ficheros):
sleep $SLURM_ARRAY_TASK_ID

# To load some software (you can show the list with 'module avail').
# Si para este script se cargaba ademas algun modulo de Python, hay que anadirlo aqui.
module purge
module load gurobi/10.0.3

# ----------------------------------------------------------------------------
# Carpeta de la ejecucion: copia propia del .py con las tres opciones fijadas;
# los datos se enlazan y los modulos se importan desde la carpeta de origen.
# ----------------------------------------------------------------------------
mkdir -p "$RUN_DIR"
cd "$RUN_DIR" || exit 1
ln -sf "$SRC_DIR/ESS_data_SPTP.csv" ESS_data_SPTP.csv
ln -sf "$SRC_DIR/scenarios_prices.xlsx" scenarios_prices.xlsx
export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"

MY_SCRIPT=scheduling_s${SLURM_ARRAY_TASK_ID}.py
sed -e "s/^N_DAYS = .*/N_DAYS = ${N_DAYS}/" \
    -e "s/^CUTS_PER_PERIOD = .*/CUTS_PER_PERIOD = ${CUTS_PER_PERIOD}/" \
    -e "s/^CUT_SCORE = .*/CUT_SCORE = '${CUT_SCORE}'/" \
    "$SRC_DIR/$SCRIPT" > "$MY_SCRIPT"

# Comprobar que las tres opciones han quedado como se pide (por si cambia el formato del .py):
if ! grep -q "^N_DAYS = ${N_DAYS}\$" "$MY_SCRIPT" || ! grep -q "^CUTS_PER_PERIOD = ${CUTS_PER_PERIOD}\$" "$MY_SCRIPT" \
   || ! grep -q "^CUT_SCORE = '${CUT_SCORE}'\$" "$MY_SCRIPT"; then
    echo "ERROR: no se han podido fijar N_DAYS, CUTS_PER_PERIOD y CUT_SCORE en $MY_SCRIPT"; exit 1
fi

# Dejar constancia de la configuracion efectiva en el fichero de salida:
hostname
echo "Escenario de precios: $SLURM_ARRAY_TASK_ID   Carpeta: $RUN_DIR"
echo "storage_cuts.py: $(md5sum "$SRC_DIR/storage_cuts.py" | cut -d' ' -f1)"
grep -E "^(SEPARATOR|LIFT|CUTS_PER_PERIOD|CUT_SCORE|EPS_VIOL|EPS_SIMULT|REACHABLE_BOUNDS|MAX_ITER|RUN_MIP_DEFAULT|RUN_MIP_PLAIN|MIP_TIME_LIMIT|N_DAYS) = " "$MY_SCRIPT" | sed 's/ *#.*//'

# the program to execute (sin argumentos: SEPARATOR y LIFT se toman del .py):
unset CERTIFY
time python "$MY_SCRIPT"
