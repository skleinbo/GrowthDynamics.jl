# Available dynamics

Routines to evolve a given state for a number of timesteps.

All routines take a `state::TumorConfiguration` as their first argument, and
further parameters like mutation rate as keyword arguments.

```@docs
LatticeTumorDynamics.moran!
LatticeTumorDynamics.eden_with_density!
LatticeTumorDynamics.exponential!
```
