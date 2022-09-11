#!/bin/bash
#
#SBATCH -o /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.out
#SBATCH -e /home/014118_emtic_oncology/Pancreas/interactivity/slurm/logs/%j.%x.%N.err
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=sanne.okel@philips.com

python train.py ./models/cdnet/cdnet_res34_aorta.py --exp-name=cdnet_res34_aorta_radius_1
python train.py ./models/cdnet/cdnet_res34_arteria_mesenterica_superior.py --exp-name=cdnet_res34_arteria_mesenterica_superior_radius_1
python train.py ./models/cdnet/cdnet_res34_common_bile_duct.py --exp-name=cdnet_res34_common_bile_duct_radius_1
python train.py ./models/cdnet/cdnet_res34_gastroduodenalis.py --exp-name=cdnet_res34_gastroduodenalis_radius_1
python train.py ./models/cdnet/cdnet_res34_pancreas.py --exp-name=cdnet_res34_pancreas_radius_1
python train.py ./models/cdnet/cdnet_res34_pancreatic_duct.py --exp-name=cdnet_res34_pancreatic_duct_radius_1
python train.py ./models/cdnet/cdnet_res34_tumour.py --exp-name=cdnet_res34_tumour_radius_1

