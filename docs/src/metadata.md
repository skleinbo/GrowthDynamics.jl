# Metadata

Every [`Population`](@ref) has [`MetaData`](@ref) attached. They store for every genotype

* population count. While in principle redundant, it is significantly cheaper to keep a tally during a simulation than to iterate the entire lattice.
* fitness value
* __TODO:__ death rate
* mutations: either `nothing`, or a vector of `Int` if mutations are present
* the time when the genotype entered the population

Additionally, the field `misc` is a dictionary for storing arbitrary user-defined key-value pairs.

!!! note "Implementation detail"

    Aside from using the value of `genotype` to index into metadata, one can use the linear index of the underlying array in which they are stored.
    The function [`index`](@ref) queries the linear index of a given genotype. An example is given below.

!!! warning

    While insertions only happen at the end and leave linear indices unchanged, deletions __will__ shift them. You should not rely on a particular index mapping
    to a given genotype when deletions are performed.

Convenient getter and setter are provided. Let's demonstrate on an example similar to the one from the [Quick Start](@ref) section, but label genotypes with strings of fixed length (from [InlineStrings.jl](https://github.com/JuliaStrings/InlineStrings.jl))

```@setup 1
import Random
Random.seed!(1234)

import GrowthDynamics.Populations: spheref
import GrowthDynamics.Lattices: HexagonalLattice
using GrowthDynamics.LatticeDynamics
using GrowthDynamics.Observables
import DataFrames: first # hide
```

```@repl 1
using InlineStrings
import Base: zero
zero(::Type{String7}) = String7("00-00")

state, _ = spheref(HexagonalLattice, 128, f=1/10, g1=String7("00-00"), g2=String7("00-AA"))

function newlabel(state,g)
  l = String7( join(rand('0':'9',2))*"-"*join(rand('A':'Z',2)) )
  while l in state.meta[:, :genotype]
    l = String7( join(rand('0':'9',2))*"-"*join(rand('A':'Z',2)) )
  end
  return l
end

eden_with_density!(state;
  label=newlabel,
  T=1024, # timesteps
  mu=1e0,  # mutation rate per genome (not site!)
  d=1/100, # death rate
  fitness=(s,g_old,g_new)->1.0 + 0.1*randn() # function to assign fitness to new genotypes
)

state.meta
```

Query according to index

```@repl 1
state.meta[2, :genotype]

state.meta[5:10, :npop]
```

or genotype

```@repl 1
state.meta[g="46-YQ"]

state.meta[g="46-YQ", :fitness] = 1.1
```

!!! note
    In places where performance is paramount, getter and setter should be called like
    `state.meta[2, Val(:genotype)]` to circumvent dynamic dispatch.

    Alternatively, `getgenotype(state, id)`, `setgenotype!(state, id)`, etc. are provided.

Because `MetaData` implements Julia's [array interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array), iterating is supported, and because each `MetaDatum` is a `NamedTuple`, conversion to e.g. a `DataFrame` is straightforward

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
rename!
hassnps
lastgenotype
length
```
