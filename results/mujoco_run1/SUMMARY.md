# MuJoCo backprop benchmark (PPO / TRPO / REINFORCE)

| Environment | Algo | Seeds | Best (mean ± std) | Final (mean ± std) |
|---|---|---|---|---|
| halfcheetah | ppo | 3 | 2903 ± 1085 | 2632 ± 1142 |
|  | trpo | 3 | 1338 ± 55 | 1004 ± 323 |
|  | reinforce | 2 | -215 ± 51 | -598 ± 1 |
| hopper | ppo | 3 | 1676 ± 1168 | 1545 ± 1221 |
|  | trpo | 3 | 331 ± 145 | 331 ± 145 |
| inverted_pendulum | ppo | 3 | 1000 ± 0 | 1000 ± 0 |
|  | trpo | 3 | 46 ± 7 | 29 ± 7 |
|  | reinforce | 3 | 6 ± 3 | 2 ± 0 |
| walker2d | ppo | 3 | 3105 ± 793 | 2650 ± 872 |
|  | trpo | 3 | 486 ± 30 | 446 ± 33 |

Learning curves: `halfcheetah_curve.png`, `hopper_curve.png`, `inverted_pendulum_curve.png`, `walker2d_curve.png`
