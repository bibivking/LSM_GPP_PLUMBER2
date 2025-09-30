#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q normalsr
#PBS -l walltime=2:00:00
#PBS -l mem=500GB
#PBS -l ncpus=104
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+scratch/w97

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable

cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2/
python save_monthly_value.py
