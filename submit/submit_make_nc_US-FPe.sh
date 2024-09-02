#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q normalbw
#PBS -l walltime=6:00:00
#PBS -l mem=50GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+scratch/w97

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable

cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2

python make_nc_file_US-FPe.py
rm make_nc_file_US-FPe.py

