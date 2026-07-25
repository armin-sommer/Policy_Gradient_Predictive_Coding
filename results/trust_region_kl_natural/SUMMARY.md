# PCPG benchmark summary

| config | algo | n | final (mean±std) | best (mean±std) | AUC (mean±std) | collapse | walltime |
|---|---|---|---|---|---|---|---|
| halfcheetah_pc_actor_critic_adam_tanh_ts10_bench_mt20_nat | pc_actor_critic | 3 | 425 ± 491 | 753 ± 28 | 284 ± 116 | 1/3 | 421s |
| halfcheetah_pc_actor_critic_adam_tanh_ts10_bench_mt80_nat | pc_actor_critic | 3 | 369 ± 480 | 812 ± 116 | 341 ± 118 | 1/3 | 427s |
| halfcheetah_pc_actor_critic_sgd_tanh_ts10_bench_lr003_mt20_nat | pc_actor_critic | 3 | 781 ± 47 | 862 ± 52 | 441 ± 18 | 0/3 | 416s |
| halfcheetah_pc_actor_critic_sgd_tanh_ts10_bench_lr003_mt80_nat | pc_actor_critic | 3 | 357 ± 433 | 802 ± 114 | 384 ± 71 | 0/3 | 414s |
