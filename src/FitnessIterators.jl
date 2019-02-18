module FitnessIterators

    export  ConstantFitness,
            LinearFitness


    abstract type AbstractFitness end
    Base.eltype(I::AbstractFitness) = Float64

    struct ConstantFitness <: AbstractFitness
        a::Float64
    end
    Base.iterate(I::ConstantFitness, state=1) = (I.a, state)
    Base.IteratorSize(I::ConstantFitness) = Base.IsInfinite()
    Base.getindex(I::ConstantFitness, j::Integer) = I.a
    Base.length(I::ConstantFitness) = Inf


end
