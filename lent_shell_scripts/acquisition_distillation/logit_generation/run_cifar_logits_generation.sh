#!/bin/tcsh

set ALLARGS = ($*)

touch $1
set LOG=$1
set TRAIN='/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/acquisition_distillation/cifar_logits_generation.sh'

set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass='*' -l osrel='*' -l hostname='\\!air206.eng.cam.ac.uk' -l hostname='\\!air209.eng.cam.ac.uk' $TRAIN $2 $3 $4 $5`

#set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass=kepler -l osrel='*' -l hostname='*' $TRAIN`
#set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass=pascal -l osrel='*' -l hostname='*' $TRAIN`
#set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass=pascal -l osrel='*' -l hostname=air209.eng.cam.ac.uk  $TRAIN`
#set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass=volta -l osrel='*' -l hostname='*' $TRAIN`
#set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass=volta -l osrel='*' -l hostname=air213.eng.cam.ac.uk $TRAIN`

echo $CMD
