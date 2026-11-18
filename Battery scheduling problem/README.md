# Battery Scheduling Problem

This folder contains the data sets and code used in the battery scheduling (energy arbitrage) case study of the related research titled *The Polyhedral Structure of Energy Storage Operations: Convex Hull and Valid Inequalities*. A single battery buys and sells energy at given hourly prices to maximize its profit. The problem is solved with three approaches:

* **MIP**: the exact mixed-integer model, in which a binary variable per period prevents simultaneous charging and discharging. It is solved with Gurobi both with default settings (MIP-D) and without presolve, cuts and heuristics (MIP).
* **Pozo LP**: the linear relaxation of the convex hull formulation of [1].
* **LP with cuts**: a cutting-plane algorithm that starts from a strengthened LP relaxation and iteratively adds submodular capacity cuts until the LP solution no longer charges and discharges in the same period.

## Data and Code Description

The following files are included in this folder:

* `ESS_data_SPTP.csv`: contains the 100 battery parameter configurations obtained from [1]. Each row gives the maximum charging and discharging power (`PcMax`, `PdMax`), the charging and discharging efficiencies (`eta_c`, `eta_d`), the energy bounds (`Emin`, `Emax`) and the initial state of charge (`E0`).
* `scenarios_prices.xlsx`: contains 10 daily electricity price scenarios with hourly resolution (one column per scenario, 24 rows). For longer horizons (72 h), consecutive scenarios are concatenated.
* `Scheduling_with_cuts_gurobipy_v2.py`: main script of the case study, written with gurobipy. For each price scenario and each battery configuration, it solves the MIP, the Pozo LP and the LP with cuts, and stores the results. The options of the experiment are set at the top of the file:
  * `N_DAYS`: length of the horizon (1 = 24 h, 3 = 72 h).
  * `CUTS_PER_PERIOD`: number of cuts added per period with simultaneous charging and discharging (0 = all violated cuts, CP; 1 = only the best one, CP1).
  * `CUT_SCORE`: criterion used to rank the cuts (`violation` or `efficacy`).
  * `SEPARATOR`: separation routine (`new` = `storage_cuts.py`, `old` = `old_separation.py`).
  * `LIFT`, `REACHABLE_BOUNDS`, `MIP_TIME_LIMIT`, etc.

  If the data files are not found, the script generates random synthetic instances, so that it can run anywhere.
* `storage_cuts.py`: module with the separation, strengthening (lifting) and verification of the submodular capacity cuts for the operational polytope of an energy storage unit. It also includes an exact dynamic programming oracle to check the validity of the cuts.
* `old_separation.py`: original heuristic separation routine, kept only for comparison with the new separator.
* `envia_scheduling.sh`: SLURM script used to run the experiments on a computing cluster. It launches an array of 10 tasks, one per price scenario. Each configuration writes its results in its own folder (`run_T<T>_<variant>`). Usage examples:
  ```
  sbatch envia_scheduling.sh                          # CP,  T = 24
  sbatch envia_scheduling.sh 1 1                      # CP1, T = 24
  sbatch envia_scheduling.sh 1 0 efficacy             # CP,  T = 24, cuts ranked by efficacy
  sbatch --time=0-03:00:00 envia_scheduling.sh 3 0    # CP,  T = 72
  ```
* `run_T24_CP.zip`, `run_T24_CP1.zip`, `run_T24_CP_efficacy.zip` and `run_T72_CP.tgz`: results of the experiments for each configuration. For each price scenario `s<k>` (k = 0, ..., 9) they contain:
  * `results_mip_s<k>.csv`: MIP without presolve, cuts and heuristics (profit, charging and discharging profiles, time, nodes, ...).
  * `results_mip_default_s<k>.csv`: MIP with Gurobi default settings.
  * `results_pozo_s<k>.csv`: Pozo LP relaxation (profit, profiles, time and gap with respect to the MIP).
  * `results_morales_new_s<k>.csv`: LP with cuts (profit, profiles, solver/separation/total times, gap, number of iterations and cuts, ...).
  * `<k>_new.res`: one-line summary of the scenario (only in the 24 h runs).

  The 24 h archives also include the copy of the script used in each task (`scheduling_s<k>.py`) and the data files.

## How to run

Requirements: Python 3 with `numpy`, `pandas`, `openpyxl` and `gurobipy` (a Gurobi license is needed).

* Locally, from this folder: `python Scheduling_with_cuts_gurobipy_v2.py` solves all the price scenarios one after the other.
* On a cluster with SLURM: `sbatch envia_scheduling.sh [N_DAYS] [CUTS_PER_PERIOD] [CUT_SCORE]`. The values passed to the script override those written in the `.py` file.

## References

[1] D. Pozo (2023). Convex hull formulations for linear modeling of energy storage systems, IEEE Transactions on Power Systems 38 (6), 5934-5936.

## Developed by

* Juan M. Morales ([juan.morales@uma.es](mailto:juan.morales@uma.es)) - [GitHub: Juanmi82mg](https://github.com/Juanmi82mg)

## Funding

This work was supported by the Consejería de Universidad, Investigación e Innovación de la Junta de Andalucía through project DGP-PIDI-2024-00851. Juan M. Morales is with the research group OASYS, University of Malaga, Malaga 29071, Spain: juan.morales@uma.es. The author thankfully acknowledges the computer resources, technical expertise, and assistance provided by the SCBI (Supercomputing and Bioinformatics) center of the University of Málaga.

## How to cite the repo and the paper?

If you want to cite the related paper or this repository, please use the following bib entries:

* Article:
```
@article{{ {{ Article_Citation_Key }},
title = {The Polyhedral Structure of Energy Storage Operations: Convex Hull and Valid Inequalities},
journal = {{ {{ Journal_Name }} }},
volume = {{ {{ Volume }} }},
pages = {{ {{ Pages }} }},
year = {{ {{ Year }} }},
author = {{ {{ Authors }} }}
}
```
* Repository:
```
@misc{mygithub,
author = {OASYS},
journal = {GitHub repository},
title = {Data and Code for The Polyhedral Structure of Energy Storage Operations: Convex Hull and Valid Inequalities},
year = 2026,
url = "https://github.com/groupoasys/storage\_valid\_inequalities",
}
```

## Do you want to contribute?

Please, do it. Any feedback is welcome, so feel free to ask or comment anything you want via a Pull Request in this repo.
If you need extra help, you can contact us.

## License

Licensed under the GNU General Public License, Version 3 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

   http://www.gnu.org/licenses/gpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the License governing permissions and
limitations under the License.
