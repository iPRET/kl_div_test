#!/bin/bash
# Inner script: runs inside the Singularity container.
torchrun --nproc_per_node 4 test_kl_loss.py
