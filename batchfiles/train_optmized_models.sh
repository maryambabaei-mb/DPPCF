#!/bin/bash
#SBATCH --time=10:01:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --mail-user=maryam.babaei.1@ens.etsmtl.ca
#SBATCH --array=0,1,2,3,4
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-ebrahimi

datasets=(adult compas default_credit)
models=(NN RF SVM)
epsilons=(0.01 0.1 1 5 10)

echo "#!/bin/bash"
echo "#SBATCH --time=05:01:00"
echo "#SBATCH --cpus-per-task=8"
echo "#SBATCH --mem-per-cpu=6G"
echo "#SBATCH --array=0,1,2,3,4"
echo "export TMPDIR=/tmp"
echo "cd ../.."

for dataset in "${datasets[@]}" ### except for synth
    do
        for model in "${models[@]}" 
            do	
                echo "dataset" $dataset "rseed" $SLURM_ARRAY_TASK_ID "model" $model  
                python train_models_optimized.py --dataset $dataset --rseed $SLURM_ARRAY_TASK_ID --model $model 

            done
        
    done
	 
