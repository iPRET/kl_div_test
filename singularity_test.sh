#!/bin/bash
# Singularity wrapper: launches the container and runs the inner script.
singularity exec --nv -B /e:/e ../../nemo_sandbox ./run_test.sh
