# Meta Data

```@meta
CurrentModule = GrowthDynamics.TumorConfigurations
```

Every [`TumorConfiguration`](@ref) has [`MetaData`](@ref) attached. They contain for every genotype

* A population count. While in principle redundant, it is significantly cheaper than enumerating the lattice.
* A fitness value
* TODO: A death rate
* Mutations: Either `nothing`, or a vector of `Int` if mutations are present.
* A timestamp when the genotype entered the population.

Additionally, `meta.misc` is a dictionary for storing arbitrary user-defined key-value pairs.

Convenient getter and setter routines are provided. Let's demonstrate with an example similar to the one from the [Quick Start](@ref) section, but label genotypes with strings of fixed length (from [InlineStrings.jl](https://github.com/JuliaStrings/InlineStrings.jl))

```@setup 1
import Random
Random.seed!(1234)

import GrowthDynamics.TumorConfigurations: spheref
import GrowthDynamics.Lattices: HexagonalLattice
using GrowthDynamics.LatticeTumorDynamics
using GrowthDynamics.TumorObservables
import DataFrames: first # hide
```

```@repl 1
using InlineStrings
import Base: zero
zero(::Type{String7}) = String7("00-00")

state, _ = spheref(HexagonalLattice, 128, f=1/10, g1=String7("00-00"), g2=String7("00-AA"))

eden_with_density!(state;
  label=(state,g)->String7( join(rand('0':'9',2))*"-"*join(rand('A':'Z',2)) ),
  T=1024, # timesteps
  mu=1e0,  # mutation rate per genome (not site!)
  d=1/100, # death rate
  fitness=(s,g_old,g_new)->1.0 + 0.1*randn() # function to assign fitness to new genotypes
)
```

```@repl 1
state.meta
```

We can query according to index

```@repl 1
state.meta[2, :genotype]
```

```@repl 1
state.meta[5:10, :npop]
```

or genotype

```@repl 1
state.meta[g="46-YQ"]
```

```@repl 1
state.meta[g="46-YQ", :fitness] = 1.1
```

!!! note
    In places where performance is paramount, getter and setter should be called like
    `state.meta[2, Val(:genotype)]` to circumvent dynamic dispatch.

    Alternatively, `getgenotype(state, id)`, `setgenotype!(state, id)`, etc. are provided.

Because `MetaData` implements Julia's [array interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array), iterating is supported, and because each `MetaDatum` is a `NamedTuple`, we can for example convert to a `DataFrame`

```@repl 1
using DataFrames

filter(v->v.npop>10, state.meta) |> DataFrame
```

## API

```@docs
MetaData
MetaDatum
index(::MetaData{T}, ::T) where {T}
index!
```
