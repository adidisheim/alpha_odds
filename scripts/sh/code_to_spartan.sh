#!/usr/bin/env bash
scp *.py adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/alpha_odds/
scp utils_locals/*.py adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/alpha_odds/utils_locals/
scp _01_process_files/*.py adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/alpha_odds/
scp _02_summary_stats/*.py adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/alpha_odds/
scp _03_feature_engeneering/*.py adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/alpha_odds/
scp _04_first_run_and_analysis/*.py adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/alpha_odds/
scp scripts/slurm/* adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/alpha_odds/
scp _05_paper_trading/scripts/save_model_artifacts.py adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/alpha_odds/

