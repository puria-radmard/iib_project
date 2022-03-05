#!/bin/tcsh

set ALLARGS = ($*)

touch $1
set LOG=$1
set TRAIN='/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/las_multitask/train_multitask_speech.sh'

set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass='*' -l osrel='*' $TRAIN $2 $3 $4 $5 $6 $7 $8`
# set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass=volta -l osrel='*' $TRAIN $2 $3 $4 $5 $6`

#set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass=kepler -l osrel='*' -l hostname='*' $TRAIN`
#set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass=pascal -l osrel='*' -l hostname='*' $TRAIN`
#set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass=pascal -l osrel='*' -l hostname=air209.eng.cam.ac.uk  $TRAIN`
#set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass=volta -l osrel='*' -l hostname='*' $TRAIN`
#set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass=volta -l osrel='*' -l hostname=air213.eng.cam.ac.uk $TRAIN`

echo $CMD

# train.sh
# run-train.sh

# When you run the run-train.sh file it is called as:
# ./run-train.sh /path/to/log.txt

# Kepler is air206.eng.cam.ac.uk 
# Kepler is air207.eng.cam.ac.uk 

# Pascal is air208.eng.cam.ac.uk 
# Pascal is air209.eng.cam.ac.uk 

# Volta is air210.eng.cam.ac.uk and above
