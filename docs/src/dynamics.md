# Dynamics

## Available dynamics

Routines to evolve a given state for a number of timesteps.

All routines take a `state::Population` as their first argument, and
further parameters like mutation rate as keyword arguments.

```@docs
LatticeDynamics.moran!
LatticeDynamics.eden_with_density!
LatticeDynamics.exponential!
```
