# PCPG benchmark summary

| config | algo | n | final (mean±std) | best (mean±std) | AUC (mean±std) | collapse | walltime |
|---|---|---|---|---|---|---|---|
| halfcheetah_pc_actor_critic_adam_relu | pc_actor_critic | 3 | 381 ± 429 | 1226 ± 476 | 214 ± 104 | 2/3 | 408s |
| halfcheetah_pc_actor_critic_adam_tanh | pc_actor_critic | 3 | 649 ± 644 | 1000 ± 148 | 423 ± 239 | 1/3 | 412s |
| halfcheetah_pc_actor_critic_sgd_relu | pc_actor_critic | 3 | -233 ± 216 | 281 ± 409 | -169 ± 195 | 1/3 | 394s |
| halfcheetah_pc_actor_critic_sgd_tanh | pc_actor_critic | 3 | 45 ± 577 | 707 ± 225 | 184 ± 339 | 2/3 | 400s |
| halfcheetah_pc_actor_critic_ts01 | pc_actor_critic | 3 | -25 ± 17 | 1 ± 12 | -12 ± 3 | 0/3 | 377s |
| halfcheetah_pc_actor_critic_ts03 | pc_actor_critic | 3 | 477 ± 366 | 488 ± 353 | 60 ± 69 | 0/3 | 382s |
| halfcheetah_pc_actor_critic_ts10 | pc_actor_critic | 3 | 279 ± 466 | 656 ± 225 | 224 ± 311 | 2/3 | 405s |
