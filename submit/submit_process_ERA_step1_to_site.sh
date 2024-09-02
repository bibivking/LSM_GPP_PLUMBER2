#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q hugemembw
#PBS -l walltime=24:00:00
#PBS -l mem=1000GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+scratch/w97+gdata/zz93

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable

cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2/
python process_ERA_step1_to_site.py
