# Available dynamics

Routines to evolve a given state a number of timesteps.

All routines take a `state::TumorConfiguration` as their first argument, and
further parameters like mutation rate as keyword arguments.

```@docs
LatticeTumorDynamics.moran!
LatticeTumorDynamics.die_or_proliferate!
LatticeTumorDynamics.exponential!
```
