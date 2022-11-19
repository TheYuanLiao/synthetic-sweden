# Impacts of charging behaviors on BEV charging infrastructure needs and energy use
Yuan Liao, Caglar Tozluoglu, Frances Sprei, Sonia Yeh, Swapnil Dhamal

## 1. Data preparation
Prepare simulation data described in Section 2.1 Simulating mobility trajectories
and Section 2.2 BEV simulation.

| Step | Script                                     | Objective                                                                                       |
|------|--------------------------------------------|-------------------------------------------------------------------------------------------------|
| 1    | `src/1-agents-process.ipynb`               | Select agents who live in the study area, Västra Götalands region.                              |
| 2    | `src/2-matsim-data-preparation.ipynb`      | Prepare road network, car agents, and create MATSim input files.                                |
| 3    | `src/3-bev-charging-dynamics-data-prep.py` | Prepare look-up tables for charging time considering different power ratings and battery sizes. |

## 2. Simulating mobility trajectories
Simulate agents' car trajectories using prepared input files including road network, activity plans, 
and configuration (`src/matsim_simulation/scenario_vg_car/config.xml`).

After the simulation, the MATSim output (`output_experienced_plans.xml.gz`) is summarized together with the raw input to have the same data format: trip/activity records. 
Some basic characteristics are also calculated. See `src/4-matsim-output-summary.py`, and called functions in `lib/dataworkers.py`.

## 3. Additional data preparation for BEV simulation
| Step | Script                                                              | Objective                                                                                                                                                                                                     |
|------|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | `src/5-stratified-sample-agents.py`                                 | Remove agents with infeasible activity plans according to MATSim,  create a representative subset of car agents according to their home's statistical zone, attach income information,  and assign car fleet. |
| 2    | `src/6-bev-sim-preparation.ipynb` `src/qgis_model/add_slope.model3` | Prepare road network with slope,  create batches of trajectories,  prepare home charger access data, and prepare charging opportunity data using experienced trip records from MATSim.                        |

## 4. BEV simulation - baseline day
Simulate one day's activities of car agents. The class and called functions can be found in `lib/ev.py`.

| Step | Script                                   | Objective                                                                                                                                                                     |
|------|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | `src/7-bev-sim-baseline.py`              | Call the class `EVSimulation` in `lib/ev.py` over 20 batches of car trajectories, simulate their discharging and charging time history, and store the output in the database. |
| 2    | `8-bev-sim-baseline-individual-stats.py` | Summarize the individual statistics of charging records e.g., SOC in the end, and intermediate charging usage.                                                                |

## 5. BEV simulation and results summary
Simulate multiple days of car agents using the same activity plans but with continuous SOC. The final results are based on the 5th day.
The class and called functions can be found in `lib/ev.py`.

| Step | Script                                   | Objective                                                                                                                                                                                                           |
|------|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | `src/9-bev-sim.py`                       | Call the class `EVSimulationMulti(EVSimulation)` in `lib/ev.py` over 20 batches of car trajectories, simulate their discharging and charging time history over multiple days, and store the output in the database. |
| 2    | `src/10-bev-sim-stats-individual.py`     | Summarize the individual statistics of charging records e.g.,  SOC in the end, and intermediate charging usage.                                                                                                     |
| 3    | `src/11-bev-sim-stats-spatiotemporal.py` | Summarize the charging points statistics, e.g., number of fast charging points per statistical zone.                                                                                                                |
| 4    | `src/12-charge-points-comparison.ipynb`  | Compare simulated charging points with existing charging infrastructure in the study area.                                                                                                                          |

## 7. A summary of applied agents' mobility patterns (Appendix A)
This is done in the notebook `src/a1-agents-mobility-patterns.ipynb`, including preparing data for visualization.

## 8. Sensitivity analysis (Appendix B)
This is done in the notebook `src/b1-sensitivity-stats-aggregation.ipynb` and `src/b2-sensitivity-results-summary.ipynb`.

## 9. Results illustration
The below shows a summary of figures produced by the scripts under `src/visualisation/`.

| Script                               | Objective                                                                                  | Output                                                | No. in the manuscript |
|--------------------------------------|--------------------------------------------------------------------------------------------|-------------------------------------------------------|-----------------------|
| `fig2-sim-input.R`                   | Simulation inputs: agents, road network, and BEVs.                                         | `figures/scenario_vg_car_inputs.png`                  | 2                     |
| `fig3-initial-socs-example-agents.R` | An example of SOC time history of a selected agent and initial socs distribution.          | `figures/scenario_vg_car_socs_examples.png`           | 3                     |
| `fig4-indi-charging.R`               | Charging demand of individual agents.                                                      | `figures/scenario_vg_car_indi_stats.png`              | 4                     |
| `fig5-inf-chargers-spatial.R`        | Spatial distributions of charging points for daytime charging by charging strategy.        | `figures/scenario_vg_car_inf_spatial.png`             | 5                     |
| `fig6-inf-chargers-density.R`        | Daytime charging point density by charging strategy and occasion (work or other).          | `figures/scenario_vg_car_inf_density.png`             | 6                     |
| `fig7-power-tempo.R`                 | Hourly power demand of all daytime charging points by charging strategy.                   | `figures/scenario_vg_car_inf_tempo.png`               | 7                     |
| `fig8-inf-comparison.R`              | Spatial disparity in charging points between simulated results and today's infrastructure. | `figures/scenario_vg_car_inf_comp.png`                | 8                     |
| `figa1-act-tempo-duration.R`         | Activity patterns.                                                                         | `figures/scenario_vg_car_mobi_act.png`                | A.1                   |
| `figa2-daily-distance.R`             | Distribution of daily travel distance by car.                                              | `figures/scenario_vg_car_mobi_distance.png`           | A.2                   |
| `figb1-power-tempo-sensitivity.R`    | Hourly power demand for daytime charging by scenario.                                      | `figures/scenario_vg_car_power_tempo_sensitivity.png` | B.1                   |