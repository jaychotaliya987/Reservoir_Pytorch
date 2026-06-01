
### All weights upfront
======================================================================
Processing PP (points per period): 90.0
======================================================================
Input shape: (9000, 3)
Input/Output dim: 3
[Data preparation] elapsed: 0.00s
Total parameter combinations: 1000
Device: cuda

[Building all weights (1000 configs)] elapsed: 429.91s

============================================================
Batch 1 | Configs 1-80/1000
============================================================
[Batch init (80 configs)] elapsed: 0.82s
Readout training complete.
[Batch training (80 configs)] elapsed: 29.64s
[Batch prediction (80 configs)] elapsed: 30.66s
[Result extraction (80 configs)] elapsed: 0.02s
Batch time: 61.14s (80 configs)

============================================================
Batch 2 | Configs 81-160/1000
============================================================
[Batch init (80 configs)] elapsed: 0.00s
Readout training complete.
[Batch training (80 configs)] elapsed: 31.61s
[Batch prediction (80 configs)] elapsed: 26.05s
[Result extraction (80 configs)] elapsed: 0.24s
Batch time: 57.91s (80 configs)

============================================================
Batch 3 | Configs 161-240/1000
============================================================
[Batch init (80 configs)] elapsed: 0.00s
Readout training complete.
[Batch training (80 configs)] elapsed: 25.10s
[Batch prediction (80 configs)] elapsed: 23.39s
[Result extraction (80 configs)] elapsed: 0.24s
Batch time: 48.72s (80 configs)

============================================================
Batch 4 | Configs 241-320/1000
============================================================
[Batch init (80 configs)] elapsed: 0.01s
Readout training complete.
[Batch training (80 configs)] elapsed: 25.63s
[Batch prediction (80 configs)] elapsed: 22.85s
[Result extraction (80 configs)] elapsed: 0.23s
Batch time: 48.71s (80 configs)

============================================================
Batch 5 | Configs 321-400/1000
============================================================
[Batch init (80 configs)] elapsed: 0.00s
Readout training complete.
[Batch training (80 configs)] elapsed: 24.78s
[Batch prediction (80 configs)] elapsed: 22.89s
[Result extraction (80 configs)] elapsed: 0.23s
Batch time: 47.91s (80 configs)

============================================================
Batch 6 | Configs 401-480/1000
============================================================
[Batch init (80 configs)] elapsed: 0.01s
Readout training complete.
[Batch training (80 configs)] elapsed: 24.65s
[Batch prediction (80 configs)] elapsed: 22.84s
[Result extraction (80 configs)] elapsed: 0.23s
Batch time: 47.73s (80 configs)

============================================================
Batch 7 | Configs 481-560/1000
============================================================
[Batch init (80 configs)] elapsed: 0.00s
Readout training complete.
[Batch training (80 configs)] elapsed: 24.44s
[Batch prediction (80 configs)] elapsed: 22.96s
[Result extraction (80 configs)] elapsed: 0.23s
Batch time: 47.63s (80 configs)

============================================================
Batch 8 | Configs 561-640/1000
============================================================
[Batch init (80 configs)] elapsed: 0.00s
Readout training complete.
[Batch training (80 configs)] elapsed: 24.73s
[Batch prediction (80 configs)] elapsed: 22.79s
[Result extraction (80 configs)] elapsed: 0.24s
Batch time: 47.76s (80 configs)

============================================================
Batch 9 | Configs 641-720/1000
============================================================
[Batch init (80 configs)] elapsed: 0.00s
Readout training complete.
[Batch training (80 configs)] elapsed: 24.68s
[Batch prediction (80 configs)] elapsed: 22.89s
[Result extraction (80 configs)] elapsed: 0.23s
Batch time: 47.81s (80 configs)

============================================================
Batch 10 | Configs 721-800/1000
============================================================
[Batch init (80 configs)] elapsed: 0.00s
Readout training complete.
[Batch training (80 configs)] elapsed: 24.64s
[Batch prediction (80 configs)] elapsed: 22.77s
[Result extraction (80 configs)] elapsed: 0.24s
Batch time: 47.65s (80 configs)

============================================================
Batch 11 | Configs 801-880/1000
============================================================
[Batch init (80 configs)] elapsed: 0.00s
Readout training complete.
[Batch training (80 configs)] elapsed: 25.18s
[Batch prediction (80 configs)] elapsed: 22.89s
[Result extraction (80 configs)] elapsed: 0.22s
Batch time: 48.30s (80 configs)

============================================================
Batch 12 | Configs 881-960/1000
============================================================
[Batch init (80 configs)] elapsed: 0.00s
Readout training complete.
[Batch training (80 configs)] elapsed: 24.30s
[Batch prediction (80 configs)] elapsed: 26.48s
[Result extraction (80 configs)] elapsed: 0.35s
Batch time: 51.14s (80 configs)

============================================================
Batch 13 | Configs 961-1000/1000
============================================================
[Batch init (40 configs)] elapsed: 0.00s
Readout training complete.
[Batch training (40 configs)] elapsed: 15.38s
[Batch prediction (40 configs)] elapsed: 11.72s
[Result extraction (40 configs)] elapsed: 0.11s
Batch time: 27.22s (40 configs)

============================================================
Total combinations processed: 1000/1000
============================================================


Results saved to: results/Chaotic/LorenzLHS\90.0.pkl
Memory released: 0.11 MB
Total time for PP 90.0: 1064.31 seconds
======================================================================

~ 17 minutes total time for a single PP value.
This is a clear step up from the Serial version which took 72 minutes for single PP.


## Batched Weight generation


Selected system: /Chaotic/Lorenz
Loading system from datasets...
System loaded.


======================================================================
Processing PP (points per period): 95.0
======================================================================
Input shape: (9500, 3)
Input/Output dim: 3
[Data preparation] elapsed: 0.00s
Total parameter combinations: 1000
Device: cuda


============================================================
Batch 1 | Configs 1-64/1000
============================================================
[Building batch weights (64 configs)] elapsed: 20.67s
[Batch init (64 configs)] elapsed: 0.26s
Readout training complete.
[Batch training (64 configs)] elapsed: 13.86s
[Batch prediction (64 configs)] elapsed: 17.66s
[Result extraction (64 configs)] elapsed: 0.05s
Batch time: 52.49s (64 configs)

============================================================
Batch 2 | Configs 65-128/1000
============================================================
[Building batch weights (64 configs)] elapsed: 21.52s
[Batch init (64 configs)] elapsed: 0.00s
Readout training complete.
[Batch training (64 configs)] elapsed: 14.29s
[Batch prediction (64 configs)] elapsed: 17.66s
[Result extraction (64 configs)] elapsed: 0.22s
Batch time: 53.70s (64 configs)

============================================================
Batch 3 | Configs 129-192/1000
============================================================
[Building batch weights (64 configs)] elapsed: 25.36s
[Batch init (64 configs)] elapsed: 0.21s
Readout training complete.
[Batch training (64 configs)] elapsed: 14.06s
[Batch prediction (64 configs)] elapsed: 17.80s
[Result extraction (64 configs)] elapsed: 0.20s
Batch time: 57.63s (64 configs)

============================================================
Batch 4 | Configs 193-256/1000
============================================================
[Building batch weights (64 configs)] elapsed: 24.54s
[Batch init (64 configs)] elapsed: 0.22s
Readout training complete.
[Batch training (64 configs)] elapsed: 14.41s
[Batch prediction (64 configs)] elapsed: 17.66s
[Result extraction (64 configs)] elapsed: 0.21s
Batch time: 57.04s (64 configs)

============================================================
Batch 5 | Configs 257-320/1000
============================================================
[Building batch weights (64 configs)] elapsed: 24.91s
[Batch init (64 configs)] elapsed: 0.01s
Readout training complete.
[Batch training (64 configs)] elapsed: 14.09s
[Batch prediction (64 configs)] elapsed: 18.40s
[Result extraction (64 configs)] elapsed: 0.20s
Batch time: 57.61s (64 configs)

============================================================
Batch 6 | Configs 321-384/1000
============================================================
[Building batch weights (64 configs)] elapsed: 24.01s
[Batch init (64 configs)] elapsed: 0.01s
Readout training complete.
[Batch training (64 configs)] elapsed: 14.20s
[Batch prediction (64 configs)] elapsed: 17.73s
[Result extraction (64 configs)] elapsed: 0.21s
Batch time: 56.16s (64 configs)

============================================================
Batch 7 | Configs 385-448/1000
============================================================
[Building batch weights (64 configs)] elapsed: 24.25s
[Batch init (64 configs)] elapsed: 0.18s
Readout training complete.
[Batch training (64 configs)] elapsed: 14.12s
[Batch prediction (64 configs)] elapsed: 17.62s
[Result extraction (64 configs)] elapsed: 0.21s
Batch time: 56.38s (64 configs)

============================================================
Batch 8 | Configs 449-512/1000
============================================================
[Building batch weights (64 configs)] elapsed: 24.84s
[Batch init (64 configs)] elapsed: 0.18s
Readout training complete.
[Batch training (64 configs)] elapsed: 14.14s
[Batch prediction (64 configs)] elapsed: 17.83s
[Result extraction (64 configs)] elapsed: 0.21s
Batch time: 57.19s (64 configs)

============================================================
Batch 9 | Configs 513-576/1000
============================================================
[Building batch weights (64 configs)] elapsed: 25.76s
[Batch init (64 configs)] elapsed: 0.21s
Readout training complete.
[Batch training (64 configs)] elapsed: 14.16s
[Batch prediction (64 configs)] elapsed: 17.57s
[Result extraction (64 configs)] elapsed: 0.22s
Batch time: 57.92s (64 configs)

============================================================
Batch 10 | Configs 577-640/1000
============================================================
[Building batch weights (64 configs)] elapsed: 31.49s
[Batch init (64 configs)] elapsed: 0.12s
Readout training complete.
[Batch training (64 configs)] elapsed: 14.45s
[Batch prediction (64 configs)] elapsed: 17.73s
[Result extraction (64 configs)] elapsed: 0.22s
Batch time: 64.00s (64 configs)

============================================================
Batch 11 | Configs 641-704/1000
============================================================
[Building batch weights (64 configs)] elapsed: 31.53s
[Batch init (64 configs)] elapsed: 0.24s
Readout training complete.
[Batch training (64 configs)] elapsed: 13.98s
[Batch prediction (64 configs)] elapsed: 17.67s
[Result extraction (64 configs)] elapsed: 0.22s
Batch time: 63.64s (64 configs)

============================================================
Batch 12 | Configs 705-768/1000
============================================================
[Building batch weights (64 configs)] elapsed: 30.73s
[Batch init (64 configs)] elapsed: 0.28s
Readout training complete.
[Batch training (64 configs)] elapsed: 343.30s
[Batch prediction (64 configs)] elapsed: 439.61s
[Result extraction (64 configs)] elapsed: 4.25s
Batch time: 818.16s (64 configs)

============================================================
Batch 13 | Configs 769-832/1000
============================================================
[Building batch weights (64 configs)] elapsed: 21.31s
[Batch init (64 configs)] elapsed: 0.20s
Readout training complete.
[Batch training (64 configs)] elapsed: 13.38s
[Batch prediction (64 configs)] elapsed: 16.77s
[Result extraction (64 configs)] elapsed: 0.20s
Batch time: 51.86s (64 configs)

============================================================
Batch 14 | Configs 833-896/1000
============================================================
[Building batch weights (64 configs)] elapsed: 21.36s
[Batch init (64 configs)] elapsed: 0.17s
Readout training complete.
[Batch training (64 configs)] elapsed: 13.53s
[Batch prediction (64 configs)] elapsed: 17.05s
[Result extraction (64 configs)] elapsed: 0.20s
Batch time: 52.30s (64 configs)

============================================================
Batch 15 | Configs 897-960/1000
============================================================
[Building batch weights (64 configs)] elapsed: 22.25s
[Batch init (64 configs)] elapsed: 0.17s
Readout training complete.
[Batch training (64 configs)] elapsed: 304.59s
[Batch prediction (64 configs)] elapsed: 412.03s
[Result extraction (64 configs)] elapsed: 4.39s
Batch time: 743.43s (64 configs)

============================================================
Batch 16 | Configs 961-1000/1000
============================================================
[Building batch weights (40 configs)] elapsed: 14.18s
[Batch init (40 configs)] elapsed: 0.07s
Readout training complete.
[Batch training (40 configs)] elapsed: 8.73s
[Batch prediction (40 configs)] elapsed: 11.05s
[Result extraction (40 configs)] elapsed: 0.13s
Batch time: 34.16s (40 configs)

============================================================
Total combinations processed: 1000/1000
============================================================


Results saved to: results/Chaotic/LorenzLHS\95.0.pkl
Memory released: 0.11 MB
Total time for PP 95.0: 2337.18 seconds
======================================================================


This is massive difference(38 Minutes). But the real reason behind the time increase is that the 
memory of the GPU is not enough to hold everything. it does lots of calculation in the
system shared memory. To confirm this I will run again the same code but now with system shared
memory turned off. You may see in batch 15 the time taken in training is 304s and batch prediction also 
took 412 sec.

## Batched Weight Generation with Shared Memory Turned Off

The Code crashed with 64 batch size, rerunning with 50 batch size.


======================================================================
Processing PP (points per period): 100.0
======================================================================
Input shape: (10000, 3)
Input/Output dim: 3
[Data preparation] elapsed: 0.00s
Total parameter combinations: 1000
Device: cuda


============================================================
Batch 1 | Configs 1-50/1000
============================================================
[Building batch weights (50 configs)] elapsed: 18.30s
[Batch init (50 configs)] elapsed: 0.01s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.54s
[Batch prediction (50 configs)] elapsed: 14.50s
[Result extraction (50 configs)] elapsed: 0.04s
Batch time: 44.39s (50 configs)

============================================================
Batch 2 | Configs 51-100/1000
============================================================
[Building batch weights (50 configs)] elapsed: 22.78s
[Batch init (50 configs)] elapsed: 0.21s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.34s
[Batch prediction (50 configs)] elapsed: 14.51s
[Result extraction (50 configs)] elapsed: 0.18s
Batch time: 49.01s (50 configs)

============================================================
Batch 3 | Configs 101-150/1000
============================================================
[Building batch weights (50 configs)] elapsed: 18.82s
[Batch init (50 configs)] elapsed: 0.08s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.44s
[Batch prediction (50 configs)] elapsed: 14.53s
[Result extraction (50 configs)] elapsed: 0.18s
Batch time: 45.04s (50 configs)

============================================================
Batch 4 | Configs 151-200/1000
============================================================
[Building batch weights (50 configs)] elapsed: 22.88s
[Batch init (50 configs)] elapsed: 0.26s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.62s
[Batch prediction (50 configs)] elapsed: 14.93s
[Result extraction (50 configs)] elapsed: 0.19s
Batch time: 49.89s (50 configs)

============================================================
Batch 5 | Configs 201-250/1000
============================================================
[Building batch weights (50 configs)] elapsed: 23.38s
[Batch init (50 configs)] elapsed: 0.26s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.07s
[Batch prediction (50 configs)] elapsed: 14.35s
[Result extraction (50 configs)] elapsed: 0.18s
Batch time: 49.24s (50 configs)

============================================================
Batch 6 | Configs 251-300/1000
============================================================
[Building batch weights (50 configs)] elapsed: 21.36s
[Batch init (50 configs)] elapsed: 0.01s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.26s
[Batch prediction (50 configs)] elapsed: 14.33s
[Result extraction (50 configs)] elapsed: 0.18s
Batch time: 47.14s (50 configs)

============================================================
Batch 7 | Configs 301-350/1000
============================================================
[Building batch weights (50 configs)] elapsed: 23.03s
[Batch init (50 configs)] elapsed: 0.19s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.09s
[Batch prediction (50 configs)] elapsed: 14.57s
[Result extraction (50 configs)] elapsed: 0.19s
Batch time: 49.08s (50 configs)

============================================================
Batch 8 | Configs 351-400/1000
============================================================
[Building batch weights (50 configs)] elapsed: 18.57s
[Batch init (50 configs)] elapsed: 0.06s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.53s
[Batch prediction (50 configs)] elapsed: 14.40s
[Result extraction (50 configs)] elapsed: 0.20s
Batch time: 44.75s (50 configs)

============================================================
Batch 9 | Configs 401-450/1000
============================================================
[Building batch weights (50 configs)] elapsed: 22.16s
[Batch init (50 configs)] elapsed: 0.06s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.18s
[Batch prediction (50 configs)] elapsed: 14.32s
[Result extraction (50 configs)] elapsed: 0.19s
Batch time: 47.91s (50 configs)

============================================================
Batch 10 | Configs 451-500/1000
============================================================
[Building batch weights (50 configs)] elapsed: 22.74s
[Batch init (50 configs)] elapsed: 0.24s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.10s
[Batch prediction (50 configs)] elapsed: 14.30s
[Result extraction (50 configs)] elapsed: 0.19s
Batch time: 48.57s (50 configs)

============================================================
Batch 11 | Configs 501-550/1000
============================================================
[Building batch weights (50 configs)] elapsed: 20.84s
[Batch init (50 configs)] elapsed: 0.20s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.06s
[Batch prediction (50 configs)] elapsed: 14.29s
[Result extraction (50 configs)] elapsed: 0.17s
Batch time: 46.57s (50 configs)

============================================================
Batch 12 | Configs 551-600/1000
============================================================
[Building batch weights (50 configs)] elapsed: 18.52s
[Batch init (50 configs)] elapsed: 0.25s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.14s
[Batch prediction (50 configs)] elapsed: 14.35s
[Result extraction (50 configs)] elapsed: 0.16s
Batch time: 44.43s (50 configs)

============================================================
Batch 13 | Configs 601-650/1000
============================================================
[Building batch weights (50 configs)] elapsed: 18.90s
[Batch init (50 configs)] elapsed: 0.16s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.26s
[Batch prediction (50 configs)] elapsed: 14.35s
[Result extraction (50 configs)] elapsed: 0.18s
Batch time: 44.86s (50 configs)

============================================================
Batch 14 | Configs 651-700/1000
============================================================
[Building batch weights (50 configs)] elapsed: 18.92s
[Batch init (50 configs)] elapsed: 0.22s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.21s
[Batch prediction (50 configs)] elapsed: 14.35s
[Result extraction (50 configs)] elapsed: 0.19s
Batch time: 44.89s (50 configs)

============================================================
Batch 15 | Configs 701-750/1000
============================================================
[Building batch weights (50 configs)] elapsed: 19.04s
[Batch init (50 configs)] elapsed: 0.06s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.34s
[Batch prediction (50 configs)] elapsed: 14.40s
[Result extraction (50 configs)] elapsed: 0.18s
Batch time: 45.02s (50 configs)

============================================================
Batch 16 | Configs 751-800/1000
============================================================
[Building batch weights (50 configs)] elapsed: 20.37s
[Batch init (50 configs)] elapsed: 0.19s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.25s
[Batch prediction (50 configs)] elapsed: 14.34s
[Result extraction (50 configs)] elapsed: 0.18s
Batch time: 46.33s (50 configs)

============================================================
Batch 17 | Configs 801-850/1000
============================================================
[Building batch weights (50 configs)] elapsed: 23.57s
[Batch init (50 configs)] elapsed: 0.06s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.33s
[Batch prediction (50 configs)] elapsed: 14.37s
[Result extraction (50 configs)] elapsed: 0.17s
Batch time: 49.50s (50 configs)

============================================================
Batch 18 | Configs 851-900/1000
============================================================
[Building batch weights (50 configs)] elapsed: 16.20s
[Batch init (50 configs)] elapsed: 0.07s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.31s
[Batch prediction (50 configs)] elapsed: 14.51s
[Result extraction (50 configs)] elapsed: 0.18s
Batch time: 42.27s (50 configs)

============================================================
Batch 19 | Configs 901-950/1000
============================================================
[Building batch weights (50 configs)] elapsed: 16.15s
[Batch init (50 configs)] elapsed: 0.00s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.17s
[Batch prediction (50 configs)] elapsed: 14.33s
[Result extraction (50 configs)] elapsed: 0.18s
Batch time: 41.82s (50 configs)

============================================================
Batch 20 | Configs 951-1000/1000
============================================================
[Building batch weights (50 configs)] elapsed: 15.46s
[Batch init (50 configs)] elapsed: 0.07s
Readout training complete.
[Batch training (50 configs)] elapsed: 11.31s
[Batch prediction (50 configs)] elapsed: 14.36s
[Result extraction (50 configs)] elapsed: 0.18s
Batch time: 41.38s (50 configs)

============================================================
Total combinations processed: 1000/1000
============================================================


Results saved to: results/Chaotic/LorenzLHS\100.0.pkl
Memory released: 0.12 MB
Total time for PP 100.0: 922.41 seconds
======================================================================



Very nice!! 15 mins.