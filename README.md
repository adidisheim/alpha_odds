# alpha_odds
Usually to replicate the full results you need to run all the codes in the _XX_ files in order, so first all the code in _01_ in order, then all the one in _02_ etc. 
For this first phase you just need to replicate on your side _01_process_files and _03_feature_engineering only up to _01_feature_engineering_para.

IMPORTANT NOTE: I like to organize these folders on my local machine but when code are run on the server all the code are copied to the base of the directory, it’s just to organize my thought locally and debug. To get an idea, check the code in scripts/sh/code_to_spartan.sh it’s all the scp that push the code to the server. You’ll see that I grab the code in each folder and paste it to the base. 

To get the features you’ll need to replicate what is done in: 
## _01_process_files
- _01_untar_files.py: simply untar all the base file
- _02_process_all_files_para.py: First messy code :). As you’ll see it’s designed to be run in parallel on the server through one version of it per year-months. You can find the code that launches it on the server (if that helps) in scripts/slurm/_02_process_all_files_para.slurm. The line #SBATCH --array=0-323 means you launch 324 different computers each with a different input to the code (the id of the year-month) to process.
- _03_merge_files.py: only launch once all the _02_ have finished running. It’s simply merging the output. This is standard trivial parallelization on HPC. Just launch X instances to process something, and when finished a simple script to merge. 

## _03_feature_engineering
- _00_feature_engineering_explore.py: this is solely a local code for me to debug. Ignore. 
- _01_feature_engineering_para.py: this is the big one, it loads all the files you processed in _01_process_files and computes all the momentum/std/etc features. It’s also parallelized but only in 10 random chunks. I run two versions of it controlled by the arguments args.b. Again if you want to check how it’s run, in the folder scripts/slurm/ you’ll find _01_feature_engineering_para_v1.slurm and _01_feature_engineering_para_v2.slurm.

In any of those parts, I wouldn't be suprised if you do find bug. So please be critical while you replicate! 