# Unpredictability in Seasonal Infectious Disease Spread

Simulation code accompanying the peer-reviewed publication:

> Gabrick, E.C.; Sayari, E.; **Protachevicz, P.R.**; Szezech Jr, J.D.; Iarosz, K.C.; Souza, S.L.T.; Almeida, A.C.L.; Viana, R.L.; Caldas, I.L.; Batista, A.M. **"Unpredictability in seasonal infectious diseases spread."** *Chaos, Solitons & Fractals*, 166, 113001, 2023.

Originally published as part of the author's research data collection at [henon.if.usp.br/OscilControlData](http://henon.if.usp.br/OscilControlData/).

## Research summary

This work studies the unpredictability of seasonal infectious diseases using a **SEIRS model with seasonal forcing** (a sinusoidal transmission rate representing seasonal variation in contact/transmissibility). Bifurcation diagrams (of the hysteresis type) and their corresponding Lyapunov exponents are computed to characterize the dynamical regimes.

Key findings:
- **Bistable dynamics** occur for all parameters tested: periodic and chaotic attractors can coexist for the same parameter values
- Using the inverse latent period as the control parameter, **over 70% of the interval** shows this periodic/chaotic coexistence
- Despite the competition, **chaotic attractors are preferred** in the bistable regions
- The boundaries of the bistable regions can be interpreted as **tipping points**: depending on the latent period, a predictable periodic attractor can tip into an unpredictable chaotic one

The result frames epidemic unpredictability as a consequence of bistable, chaos-preferring dynamics rather than simple parameter uncertainty, with direct implications for how confidently seasonal outbreaks (e.g. influenza, RSV) can be forecast.

## What the code does

`seir_seasonal_bistability.c` integrates a SEIRS compartmental model (Susceptible-Exposed-Infected-Recovered-Susceptible) with a seasonally forced transmission rate `beta(t) = beta0 * (1 + beta1 * cos(T))`, where `T` evolves as an additional phase variable. It uses 4th-order Runge-Kutta integration and records a **stroboscopic Poincaré section** (sampling the state once per forcing period, when `cos(T)` is near its maximum) after a long transient, which is the standard technique for revealing chaotic vs. periodic attractors in a periodically-forced system.

`splot.plt` is a Gnuplot script that renders the 2D and 3D attractor plots (periodic vs. chaotic) from the simulation output, as published in the paper.

## Requirements

A C compiler (GCC or ICC) and `libm`. Gnuplot for plotting.

## Build & run

```bash
gcc seir_seasonal_bistability.c -lm -O2 -o seir_sim
./seir_sim
gnuplot splot.plt
```
