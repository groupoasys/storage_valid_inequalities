# storage_valid_inequalities

The aim of this repository is to provide details of the data sets and code used in the related research titled *Linear and Second-order-cone Valid Inequalities for Problems with Storage*.

## Data and Code Description

The following files are included in this repository:
* `ESS_data_SPTP.csv`: contains the 100 battery parameter configurations obtained from [1].
* `PV_and_Wind_data_scenarios.csv`: includes 725 daily PV power profiles with hourly resolution, also obtained from [1].
* `demand_profile.csv`: provides the household demand [1].
* `scenarios_prices.xlsx`: contains the 10 price vectors from the Danish day-ahead electricity market (zone DK1), each representing a day with negative electricity prices. These days are: July 2, 2023; January 1, 2024; and June 2, 8, 9, 15, 16, 28, as well as July 4 and 7, 2024. The data is publicly available from the ENTSO-e Transparency Platform.
* `setpoint_tracking_AMPL_Case_Study.py`: includes the basic code to run the case study based on the setpoint tracking problem.
* `scheduling_AMPL_Case_Study.py`: contains the basic code to run the case study based on the storage scheduling problem.

## References

[1] D. Pozo (2023). Convex hull formulations for linear modeling of energy storage systems, IEEE Transactions on Power Systems 38 (6), 5934-5936.

## Developed by

* Juan M. Morales ([juan.morales@uma.es](mailto:juan.morales@uma.es)) - [GitHub: Juanmi82mg](https://github.com/Juanmi82mg)  


## Funding

This work was supported by the Spanish Ministry of Science and Innovation (AEI/10.13039/501100011033) through project PID2023-148291NB-I00

## How to cite the repo and the paper?

If you want to cite the related paper or this repository, please use the following bib entries:

* Article:
```
@article{{ {{ Article_Citation_Key }},
title = {{ {{ Article_Title }} }},
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
title = {Data and Code for Linear and Second-order-cone Valid Inequalities for Problems with Storage},
year = 2025,
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


