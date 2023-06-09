# Measurement-change-point-evaluation
This repository contains the experimental toolbench used to evaluate the impact of process drifts on specification measures, presented in the paper “Measuring Rule-based LTLf Process Specifications: A Probabilistic Data-driven Approach” (pre-print: https://doi.org/10.48550/arXiv.2305.05418). For other tests and tools mentioned in the paper, please visit https://oneiroe.github.io/DeclarativeSpecificationMeasurements-Journal-static/.
# Requirements
To run our Python scripts, the following libraries are required: `pm4py`, `os`, `re`, `pandas`, `numpy`, `matplotlib`.
# Overview
The first step to take is the discovery of the process specification. We do this through Janus https://github.com/Oneiroe/Janus. Other algorithms can also be empoloyed.
This is the used command to launch the Janus miner (further details on the parameters can be found in the paper):

`Java -cp Janus.jar minerful.JanusOfflineMinerStarter -iLF ${INPUT_LOG} -s 0 -c 0.95 -oJSON ${OUTPUT_FILE_JSON}`

We then extract sequences of 50 consecutive, non-overlapping trace sets from the event log with a tumbling window approach. In order to achieve this, we use the dedicated pre-processing feature of MINERful https://github.com/cdc08x/MINERful.

`./run-MINERfulSlideLogExtractor-unstable.sh -iLF ${INPUT_LOG}  -sSLoXOutDir ${OUTPUT_PATH} -iLSubLen 50`

We analyze the trend of the specification measurements for each generated sublog. This is the used command to launch the Janus measurement framework:

`java -cp Janus.jar minerful.JanusMeasurementsStarter -iLF ${INPUT_LOG}  -iLE xes -iMF ${INPUT_MODEL}  -iME json -oCSV ${OUTPUT_FILE_CSV}`

At this point we merge all the log measures generated by Janus into a single .csv file (to ease the process the bash executable file is available in the folder) and launch `measurement_changepoint.py`.



