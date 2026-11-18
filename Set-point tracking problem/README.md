# Set-Point Tracking Problem

The aim of this repository is to provide details of the data sets and code used in the related research titled *Second-Order Cone Valid Inequalities for Set-Point Tracking with Energy Storage* (preprint: [arXiv:2611.21526](https://arxiv.org/abs/2611.21526)).

## Data and Code Description

The following files are included in this repository:
* `ESS_data_SPTP.csv`: contains the 100 battery parameter configurations obtained from [1].
* `PV_and_Wind_data_scenarios.csv`: includes 725 daily PV power profiles with hourly resolution, also obtained from [1].
* `demand_profile.csv`: provides the household demand [1].
* `setpoint_tracking_letter.py`: contains the code to run the case study based on the set-point tracking problem. For each battery configuration in `ESS_data_SPTP.csv` and the PV profile selected by the SLURM array index, it solves the MIQP benchmark and the relaxations based on HCH model of [1], namely, HCH-QP, HCH-SOC and HCH-PER, using AMPL with Gurobi and MOSEK, and stores the results in `results_stp_s<k>.csv`.
* `envia.sh`: SLURM script used to parallelize the experiments on a computing cluster. It launches an array of 200 tasks, each one running `setpoint_tracking_letter.py` for a different PV production profile (and hence net demand profile) over all the battery configurations.
* `read_results.py`: aggregates the results of all the SLURM tasks (files `results_stp_s*.csv`) and prints the summary table of the paper.
* `resultados_stp.zip`: contains the results of the experiment, that is, the 200 files `results_stp_s<k>.csv` (one per PV profile) produced by `setpoint_tracking_letter.py`.

## References

[1] D. Pozo (2023). Convex hull formulations for linear modeling of energy storage systems, IEEE Transactions on Power Systems 38 (6), 5934-5936.

## Developed by

* Juan M. Morales ([juan.morales@uma.es](mailto:juan.morales@uma.es)) - [GitHub: Juanmi82mg](https://github.com/Juanmi82mg)  


## Funding

This work was supported by the Consejería de Universidad, Investigación e Innovación de la Junta de Andalucía through project DGP-PIDI-2024-00851. Juan M. Morales is with the research group OASYS, University of Malaga, Malaga 29071, Spain: juan.morales@uma.es. The author thankfully acknowledges the computer resources, technical expertise, and assistance provided by the SCBI (Supercomputing and Bioinformatics) center of the University of Málaga.

## How to cite the repo and the paper?

If you want to cite the related paper or this repository, please use the following bib entries:

* Article (preprint available at [arXiv:2611.21526](https://arxiv.org/abs/2611.21526)):
```
@misc{morales2026secondorderconevalidinequalities,
title = {Second-Order Cone Valid Inequalities for Set-Point Tracking with Energy Storage},
author = {Juan M. Morales},
year = {2026},
eprint = {2611.21526},
archivePrefix = {arXiv},
primaryClass = {math.OC},
url = {https://arxiv.org/abs/2611.21526}
}
```
* Repository:
```
@misc{mygithub,
author = {OASYS},
journal = {GitHub repository},
title = {Data and Code for Second-Order Cone Valid Inequalities for Set-Point Tracking with Energy Storage},
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


