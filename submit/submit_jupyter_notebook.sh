#!/bin/bash
#PBS -m ae
#PBS -l walltime=2:00:00
#PBS -l ncpus=2
#PBS -l mem=50GB
#PBS -l ngpus=1
#PBS -j oe

# 设置工作目录
cd /srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_GPP_PLUMBER2

# 模块加载（根据你的系统调整）
source /srv/ccrc/LandAP/z5218916/miniconda3/etc/profile.d/conda.sh
conda activate science

# 固定端口
PORT=8888

# 当前时间戳
datetime=$(date +%Y%m%d%H%M)

# 日志文件
JUPYTER_LOG="jupyter_${datetime}.log"

# 启动 Jupyter Notebook
jupyter notebook --no-browser --port=$PORT --ip=0.0.0.0 > $JUPYTER_LOG 2>&1 &
echo "Jupyter Notebook started on port $PORT. Logs: $JUPYTER_LOG"
