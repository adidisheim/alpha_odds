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


## A notes on other codes: 
- paramaters.py: it's my standard way to keep track of all hyper-parameters and easily run grids with clear names. Some find it confusing, so don't hesitate, although I don't think you'll need it, it's mostly for my exploraiton phase. 
- utils_locals/*.py: Basically where I store most functions. Somewhat standard but if anything is unclear let me know. 




(Pdb) df
                         index  best_lay  best_back  best_lay_cum_qty  best_back_cum_qty  total_lay_qty  total_back_qty  best_lay_q_100  best_back_q_100  best_lay_q_200  best_back_q_200  best_lay_q_1000  best_back_q_1000     qty   prc      order_type          id  runner_position        file_name
0      2017-10-15 01:53:49.828      18.5       6.40              6.27              20.33          15.71           69.83     1001.000000         1.000000     1001.000000         1.000000       1001.00000          1.000000    0.00  0.00            None        -1.0                1  1.135550390.bz2
1      2017-10-15 01:58:28.668      18.5       6.40              6.27              20.33           0.00            0.00     1001.000000         1.000000     1001.000000         1.000000       1001.00000          1.000000    0.00  0.00            None        -1.0                1  1.135550390.bz2
2      2017-10-15 01:58:29.164      16.0       6.40              7.32              20.33          18.81            0.00     1001.000000         1.000000     1001.000000         1.000000       1001.00000          1.000000    0.00  0.00            None        -1.0                1  1.135550390.bz2
3      2017-10-15 01:58:29.553      16.0       6.00              7.32              21.96          18.81           73.81     1001.000000         1.000000     1001.000000         1.000000       1001.00000          1.000000    0.00  0.00            None        -1.0                1  1.135550390.bz2
4      2017-10-15 02:03:04.854      16.0       6.00              7.32              21.96           0.00            0.00     1001.000000         1.000000     1001.000000         1.000000       1001.00000          1.000000    0.00  0.00            None        -1.0                1  1.135550390.bz2
...                        ...       ...        ...               ...                ...            ...             ...             ...              ...             ...              ...              ...               ...     ...   ...             ...         ...              ...              ...
874962 2017-10-15 09:05:50.031       1.6       1.52              6.57              11.20        1270.00         7964.24        1.609343         1.504702        1.640514         1.502351          1.86605          1.414677    0.00  0.00            None  10560701.0                8  1.135552112.bz2
874963 2017-10-15 09:05:50.304       1.6       1.52              6.57              11.20        1270.00         7967.33        1.609343         1.504702        1.640514         1.502351          1.86605          1.414677    0.00  0.00            None  10560701.0                8  1.135552112.bz2
874964 2017-10-15 09:05:50.416       1.6       1.52              6.57               9.88        1161.00         7966.01        1.609343         1.504438        1.640514         1.502219          1.86605          1.414479   18.04  1.53  cross_matching  10560701.0                8  1.135552112.bz2
874965 2017-10-15 09:05:50.754       1.6       1.50              6.57             112.78        1161.00         7844.29        1.609343         1.500000        1.640514         1.475227          1.86605          1.398212  246.07  1.51  cross_matching  10560701.0                8  1.135552112.bz2
874966 2017-10-15 09:05:50.812       1.6       1.50              6.57             112.78           0.00            0.00        1.609343         1.500000        1.640514         1.475227          1.86605          1.398212    0.00  0.00            None  10560701.0                8  1.135552112.bz2


(Pdb) df
                           time  best_lay  best_back  best_lay_cum_qty  best_back_cum_qty  total_lay_qty  total_back_qty  best_lay_q_100  best_back_q_100  best_lay_q_200  best_back_q_200  best_lay_q_1000  best_back_q_1000     qty   prc order_type        id  runner_position        file_name
0       2017-10-16 00:19:52.672    100.00      19.00              2.96               6.09           8.88           21.59     1001.000000         1.000000     1001.000000         1.000000           1001.0          1.000000    0.00  0.00       None  11291584                1  1.135596789.bz2
1       2017-10-16 00:21:53.763    100.00      19.00              2.96               6.09           0.00            0.00     1001.000000         1.000000     1001.000000         1.000000           1001.0          1.000000    0.00  0.00       None  11291584                1  1.135596789.bz2
2       2017-10-16 00:21:54.567    100.00      19.00              2.96               6.09           0.00           21.59     1001.000000         1.000000     1001.000000         1.000000           1001.0          1.000000    0.00  0.00       None  11291584                1  1.135596789.bz2
3       2017-10-16 00:21:54.675    100.00      19.00              2.96               6.09           8.88           21.59     1001.000000         1.000000     1001.000000         1.000000           1001.0          1.000000    0.00  0.00       None  11291584                1  1.135596789.bz2
4       2017-10-16 00:23:54.717    100.00      19.00              2.96               6.09           0.00            0.00     1001.000000         1.000000     1001.000000         1.000000           1001.0          1.000000    0.00  0.00       None  11291584                1  1.135596789.bz2
...                         ...       ...        ...               ...                ...            ...             ...             ...              ...             ...              ...              ...               ...     ...   ...        ...       ...              ...              ...
1561686 2017-10-16 10:35:35.044      2.66       2.64              7.50             296.10         770.13        15636.12        2.709092         2.640000        2.783367         2.640000           1001.0          2.467302    0.00  0.00       None  10804734                8  1.135596995.bz2
1561687 2017-10-16 10:35:35.259      2.68       2.66             30.55               3.29         733.53        15639.41        2.767838         2.640658        2.829369         2.640329           1001.0          2.468618   94.61  2.66       back  10804734                8  1.135596995.bz2
1561688 2017-10-16 10:35:35.576      2.68       2.64             30.55             296.10         716.83        15632.18        2.767838         2.640000        2.829369         2.640000           1001.0          2.465802    0.00  0.00       None  10804734                8  1.135596995.bz2
1561689 2017-10-16 10:35:35.811      2.68       2.64             18.12             163.61         607.34        15008.58        2.832500         2.640000        2.990274         2.628324           1001.0          2.409406  760.62  2.64       back  10804734                8  1.135596995.bz2
1561690 2017-10-16 10:35:35.907      2.68       2.64             18.12             163.61           0.00            0.00        2.832500         2.640000        2.990274         2.628324           1001.0          2.409406    0.00  0.00       None  10804734                8  1.135596995.bz2

