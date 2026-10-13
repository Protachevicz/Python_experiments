# Intermittency Properties in a Temporal Lobe Epilepsy Model

Simulation code accompanying the peer-reviewed publication:

> Borges, F.S.; Gabrick, E.C.; **Protachevicz, P.R.**; Higa, G.S.V.; Lameu, E.L.; Rodriguez, P.X.R.; Ferraz, M.S.A.; Szezech Jr, J.D.; Batista, A.M.; Kihara, A.H. **"Intermittency properties in a temporal lobe epilepsy model."** *Epilepsy & Behavior*, 139, 109072, 2023.

Originally published as part of the author's research data collection at [henon.if.usp.br/OscilControlData](http://henon.if.usp.br/OscilControlData/Neuron_Dynamics/Synchronization/Intermittent/).

## Research summary

Neuronal synchronization underlies communication between brain regions and plays a key role in learning, but abnormal connectivity can drive hyper-synchronized states associated with epileptic seizures, which alternate intermittently with asynchronous states. The activity-regulated cytoskeleton-associated protein (ARC) is linked to synaptic alterations that can lead to epilepsy: induction of status epilepticus in rodent models produces intense ARC-immunoreactive neurons (IAINs) with a higher number of connections and conductance intensity than non-IAINs.

This model investigates how IAINs connectivity shapes firing patterns and synchronization in a network of adaptive exponential integrate-and-fire (AdEx) neurons:

1. Shows the emergence of synchronized burst patterns as IAINs appear in the network
2. Characterizes how increasing IAINs connectivity produces intermittent "up" (synchronous burst) and "down" (asynchronous spike) activity regimes
3. Applies simulated optogenetic control (1% of neurons made photosensitive) to suppress the highly synchronous activity, and shows this control is effective **only** when IAINs, not non-IAINs, are targeted

The result suggests IAINs play a pivotal role in both generating and suppressing highly synchronized epileptic-like activity, informing potential optogenetic intervention strategies.

## What the code does

`intermittent_dynamics_cv_r.c` simulates a network of `NN=1000` AdEx neurons coupled through excitatory/inhibitory synapses, with a subset of excitatory connections carrying elevated ARC-like conductance (`gARC`). It:
- Builds a random connectivity structure with a configurable mean degree (`KM`)
- Integrates the coupled neuronal dynamics with 4th-order Runge-Kutta
- Computes the **Kuramoto order parameter R(t)** to quantify synchronization over time
- Computes the **coefficient of variation (CV) of inter-spike intervals** to classify "up" (synchronous, low CV) vs. "down" (asynchronous, high CV) regimes, and tallies how much time the network spends in each
- Writes raster plots, CV time series, and order-parameter time series to `.dat` files for downstream analysis/plotting

## Requirements

A C compiler (GCC or ICC) and `libm`. No external dependencies.

## Build & run

```
gcc intermittent_dynamics_cv_r.c -lm -O2 -o intermittency_sim
./intermittency_sim
```

Output files (`raster_*.dat`, `CV_*.dat`, `R_*.dat`, `Vt_*.dat`) are written to the working directory and can be plotted with Gnuplot, Python/Matplotlib, or any standard plotting tool.
