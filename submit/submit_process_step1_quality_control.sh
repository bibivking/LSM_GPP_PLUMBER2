#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q normalsr
#PBS -l walltime=4:00:00
#PBS -l mem=300GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+scratch/w97+gdata/zz93

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable

cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2/
python process_step1_quality_control.py
