# PCPG benchmark summary

| config | algo | n | final (mean±std) | best (mean±std) | AUC (mean±std) | collapse | walltime |
|---|---|---|---|---|---|---|---|
| halfcheetah_pc_actor_critic_adam_tanh_ts10_bench_mt10 | pc_actor_critic | 3 | 649 ± 440 | 959 ± 63 | 407 ± 206 | 1/3 | 404s |
| halfcheetah_pc_actor_critic_adam_tanh_ts10_bench_mt20 | pc_actor_critic | 3 | 649 ± 644 | 1000 ± 148 | 423 ± 239 | 1/3 | 406s |
| halfcheetah_pc_actor_critic_adam_tanh_ts10_bench_mt40 | pc_actor_critic | 3 | 391 ± 547 | 910 ± 59 | 468 ± 72 | 1/3 | 411s |
| halfcheetah_pc_actor_critic_adam_tanh_ts10_bench_mt80 | pc_actor_critic | 3 | 699 ± 441 | 962 ± 133 | 545 ± 139 | 0/3 | 416s |
| halfcheetah_pc_actor_critic_sgd_tanh_ts10_bench_lr003_mt10 | pc_actor_critic | 3 | 406 ± 475 | 678 ± 378 | 298 ± 295 | 1/3 | 399s |
| halfcheetah_pc_actor_critic_sgd_tanh_ts10_bench_lr003_mt20 | pc_actor_critic | 3 | 279 ± 466 | 656 ± 225 | 224 ± 311 | 2/3 | 400s |
| halfcheetah_pc_actor_critic_sgd_tanh_ts10_bench_lr003_mt40 | pc_actor_critic | 3 | -62 ± 535 | 539 ± 332 | 99 ± 276 | 1/3 | 398s |
| halfcheetah_pc_actor_critic_sgd_tanh_ts10_bench_lr003_mt80 | pc_actor_critic | 3 | -99 ± 405 | 591 ± 200 | 64 ± 338 | 2/3 | 413s |
| halfcheetah_pc_reinforce_adam_tanh_ts10_bench_mt10 | pc_reinforce | 3 | 66 ± 116 | 71 ± 112 | 26 ± 55 | 0/3 | 268s |
| halfcheetah_pc_reinforce_adam_tanh_ts10_bench_mt20 | pc_reinforce | 3 | 37 ± 75 | 42 ± 71 | 9 ± 30 | 0/3 | 269s |
| halfcheetah_pc_reinforce_adam_tanh_ts10_bench_mt40 | pc_reinforce | 3 | 36 ± 71 | 42 ± 67 | 11 ± 32 | 0/3 | 266s |
| halfcheetah_pc_reinforce_adam_tanh_ts10_bench_mt80 | pc_reinforce | 3 | 59 ± 109 | 66 ± 104 | 25 ± 54 | 0/3 | 268s |
| halfcheetah_pc_reinforce_sgd_tanh_ts10_bench_lr003_mt10 | pc_reinforce | 3 | 12 ± 29 | 13 ± 28 | 3 ± 17 | 0/3 | 265s |
| halfcheetah_pc_reinforce_sgd_tanh_ts10_bench_lr003_mt20 | pc_reinforce | 3 | 8 ± 23 | 12 ± 28 | 2 ± 15 | 0/3 | 267s |
| halfcheetah_pc_reinforce_sgd_tanh_ts10_bench_lr003_mt40 | pc_reinforce | 3 | 10 ± 27 | 11 ± 27 | 1 ± 15 | 0/3 | 266s |
| halfcheetah_pc_reinforce_sgd_tanh_ts10_bench_lr003_mt80 | pc_reinforce | 3 | 12 ± 29 | 13 ± 28 | 2 ± 16 | 0/3 | 265s |
