# Analytical Solution to Short-Term Plasticity

Simulation code accompanying the peer-reviewed publication:

> **Protachevicz, P.R.**; Batista, A.M.; Caldas, I.L.; Baptista, M.S. **"Analytical solution to short-term plasticity."** *Chaos, Solitons & Fractals*, 181, 114678, 2024.

Originally released as a bioRxiv pre-print ("Analytical solutions for the short-term plasticity") and published as part of the author's research data collection at [henon.if.usp.br/OscilControlData](http://henon.if.usp.br/OscilControlData/).

## Research summary

Synaptic dynamics is central to neuronal communication, but due to its high dimensionality, the fundamental mechanisms driving different short-term plasticity (STP) regimes (facilitation, depression, and biphasic behavior) and their relation to neurotransmitter release are still not fully understood.

This work derives, from the standard Tsodyks-Markram-style STP differential equations, an **approximated discrete map with closed-form analytical solutions**:
- For a presynaptic neuron spiking **periodically**, it derives the stable equilibria and maximal release amplitude as a function of the fraction of neurotransmitter released per spike (`U`) and the presynaptic firing frequency
- For a presynaptic neuron spiking **stochastically** (Poisson process, emulating input from a complex network), it shows the time-average of the stochastic trajectory obeys the **same equations and equilibrium points** as the periodic case

In other words, synapses driven by realistic, noisy spike trains wander around equilibrium points that can be fully calculated analytically from the periodic case, a closed-form result that previously required purely numerical simulation.

## What the code does

`short_term_plasticity_analytical.c` simultaneously evolves three parallel representations of the same synapse model for direct comparison:
1. The full **numerical ODE system** (4th-order Runge-Kutta) for the four STP state variables (available, active, inactive neurotransmitter fractions, and utilization `u`)
2. The **exact analytical solution** between spikes, reset by the same spike rule
3. The **discrete map** derived in the paper, iterated spike-to-spike

It classifies the resulting dynamics into regimes (facilitation, depression, pseudo-linear, biphasic) based on how the release amplitude evolves across consecutive spikes, and writes out the map trajectory, the parameter space of regimes, and the analytical-vs-numerical discrepancy for validation.

## Requirements

A C compiler (GCC or ICC) and `libm`. No external dependencies.

## Build & run

```bash
gcc short_term_plasticity_analytical.c -lm -O2 -o stp_analytical
./stp_analytical
```

Output `.dat` files (`Dinamics*.dat`, `Parameter_space.dat`, `Solucao.dat`) can be plotted with Gnuplot, Python/Matplotlib, or any standard plotting tool.
