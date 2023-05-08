# Measurement-change-point-evaluation
This repository contains the experimental settings used to evaluate the impact of process drift on specification measures presented in the paper “Measuring Rule-based LTLf Process Specifications: A Probabilistic Data-driven Approach”.
# Requirements
In order to run the system, the following libraries must be installed: python3, pm4py, os, re, pandas, matplotlib.
# Overview
The first step to take is the discovery of the process model. In this case we do this through Janus https://github.com/Oneiroe/Janus. Other algorithms can also be used.
This is the used command to launch the Janus mining (further details on the parameters can be found in the paper):

`Java -cp Janus.jar minerful.JanusOfflineMinerStarter -iLF ${INPUT_LOG} -s 0 -c 0.95 -oJSON ${OUTPUT_FILE_JSON}`

We then extracted sequences of 50 consecutive, non-overlapping trace sets from the event log using a tumbling window approach. In order to achieve this, we used the dedicated pre-processing feature of MINERful https://github.com/cdc08x/MINERful.

`./run-MINERfulSlideLogExtractor-unstable.sh -iLF ${INPUT_LOG}  -sSLoXOutDir ${OUTPUT_PATH} -iLSubLen 50`

We analyzed the trend of the specification measurements for each generated sublog. This is the used command to launch the Janus measurement framework:

`java -cp Janus.jar minerful.JanusMeasurementsStarter -iLF ${INPUT_LOG}  -iLE xes -iMF ${INPUT_MODEL}  -iME json -oCSV ${OUTPUT_FILE_CSV}`

