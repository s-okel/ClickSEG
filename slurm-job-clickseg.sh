#!/bin/bash
#
#SBATCH -o /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.out
#SBATCH -e /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.err
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=sanne.okel@philips.com

python train.py ./models/focalclick/segformerB3_S2_aorta.py --exp-name=segformerB3_S2_aorta_test
