#!/bin/bash
#SBATCH --time=24:00:00
##SBATCH --cpus-per-task=2
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=6G
#SBATCH --array=0,1,2,3,4
#SBATCH --mail-user=maryam.babaei.1@ens.etsmtl.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-ebrahimi

datasets=(adult compas default_credit)
models=(NN RF SVM) 
CF_methods=(NICE synth_dp_cf ldp_ds_cf   LDP_SRR  inline_LDP zerocost_DP_CF)
nice_pre_mthods=(NICE synth_dp_cf ldp_ds_cf)
inline=(inline_LDP)
exp_prox=(zerocost_DP_CF)
post_methods=(LDP_SRR  LDP_CF LDP_Noisy_max)
n_counts=(3 5 10 20)
epsilons=(0.01 0.1 1 5 10) 


echo "#!/bin/bash"
echo "#SBATCH --time=12:00:00"
echo "#SBATCH --cpus-per-task=2"
echo "#SBATCH --mem-per-cpu=12G"
echo "export TMPDIR=/tmp"
echo "cd ../.."



for dataset in "${datasets[@]}"
    do
        for model in "${models[@]}"
            do 
                for method in "${nice_pre_mthods[@]}"
                    do
                        for epsilon in "${epsilons[@]}"
                            do
                                for n_count in "${n_counts[@]}"
                                    do
                                        echo "--dataset" $dataset "--rseed" $SLURM_ARRAY_TASK_ID "--model" $model "--CF_method" $method "--epsilon" $epsilon "--n_count" $n_count 
                                        srun python CF_generation_optimized.py --dataset $dataset --rseed $SLURM_ARRAY_TASK_ID --model $model --CF_method $method --epsilon $epsilon --n_count $n_count 
                                    done
                            done
                    done
            done
    done	

echo "NICE and preprocessing methods completed\n"


for dataset in "${datasets[@]}"
    do
        for model in "${models[@]}"
            do 
                for method in "${inline[@]}"
                    do
                        for epsilon in "${epsilons[@]}"
                            do
                                for n_count in "${n_counts[@]}"
                                    do
                                        echo "--dataset" $dataset "--rseed" $SLURM_ARRAY_TASK_ID "--model" $model "--CF_method" $method "--epsilon" $epsilon "--n_count" $n_count 
                                        srun python CF_generation_optimized.py --dataset $dataset --rseed $SLURM_ARRAY_TASK_ID --model $model --CF_method $method --epsilon $epsilon --n_count $n_count 
                                    done
                            done
                    done
            done
    done	

echo "inline_dp completed\n"

for dataset in "${datasets[@]}"
    do
        for model in "${models[@]}"
            do 
                for method in "${exp_prox[@]}"
                    do
                        for epsilon in "${epsilons[@]}"
                            do
                                for n_count in "${n_counts[@]}"
                                    do
                                        echo "--dataset" $dataset "--rseed" $SLURM_ARRAY_TASK_ID "--model" $model "--CF_method" $method "--epsilon" $epsilon "--n_count" $n_count 
                                        srun python CF_generation_optimized.py --dataset $dataset --rseed $SLURM_ARRAY_TASK_ID --model $model --CF_method $method --epsilon $epsilon --n_count $n_count 
                                    done
                            done
                    done
            done
    done	
echo "post_proc_methods completed\n"
for dataset in "${datasets[@]}"
    do
        for model in "${models[@]}"
            do 
                for method in "${post_methods[@]}"
                    do
                        for epsilon in "${epsilons[@]}"
                            do
                                for n_count in "${n_counts[@]}"
                                    do
                                        echo "--dataset" $dataset "--rseed" $SLURM_ARRAY_TASK_ID "--model" $model "--CF_method" $method "--epsilon" $epsilon "--n_count" $n_count 
                                        srun python CF_generation_optimized.py --dataset $dataset --rseed $SLURM_ARRAY_TASK_ID --model $model --CF_method $method --epsilon $epsilon --n_count $n_count 
                                    done
                            done
                    done
            done
    done	