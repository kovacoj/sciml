TPFM reproduction
=================
classification: TPFM_TOPOLOGIES_ONLY
domain: local 64x64 DAFoam/HFDIB fallback
ROI: 64x64, h=0.002 m
velocity error sample 0: 0.158901
velocity error sample 274: 0.185142
velocity error sample 549: 0.185171
pressure error: [0.201518, 0.154115, 0.176826]
pressure range ratios: [1.211679, 1.215378, 1.160896]

Training
========
training topology count: 256
K: 20
uses converged CFD labels for training: NO
model: SimpleFlowNet with independent phi
parameter count: 3153264
seeds: 11, 22, 33

## Teacher
mean rel_U W20 vs converged: 0.403393
mean rel_p W20 vs converged: 0.562839

## Network
mean rel_U: 0.607318
std rel_U: 0.119773
mean rel_p: 0.991826
std rel_p: 0.001979

Warm start
==========
cold iterations: not measured to a residual stopping criterion
teacher warm iterations: not measured
neural iterations: 0, 1, and 5 correction trajectories measured
T(neural) iterations: mean rel_U 0.602562
T5(neural) iterations: mean rel_U 0.567405
iteration saving: not claimed
wall-clock speedup: not claimed

Transfer
========
scratch residual after 5: NOT RUN
transfer residual after 5: NOT RUN
scratch residual after 20: NOT RUN
transfer residual after 20: NOT RUN
