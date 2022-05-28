module RoughInterfaces

import Base: firstindex, getindex, lastindex, length, setindex!, size, view
import Distributions: Normal, truncated
import Interpolations: LinearInterpolation, Periodic, Extrapolation
import CoordinateTransformations: Spherical, SphericalFromCartesian

export isinside, PolarRandomWalkGenerator, PolarRandomWalk

struct PolarRandomWalkGenerator
    r::Float64
    σr::Float64
    R::Float64
end

struct PolarRandomWalk{T}<:AbstractVector{T}
    R
    X
    Y::AbstractVector{T}
    function PolarRandomWalk(R, X, Y::AbstractVector{T}) where T
        @assert length(X)==length(Y) "Arguments must have the same length."
        new{T}(R, X, Y)
    end
end
firstindex(P::PolarRandomWalk) = firstindex(P.X)
lastindex(P::PolarRandomWalk) = lastindex(P.X)
length(P::PolarRandomWalk) = length(P.X)
getindex(P::PolarRandomWalk, i...) = (getindex(P.X, i...), getindex(P.Y, i...))
setindex!(P::PolarRandomWalk, v, i...) = PolarRandomWalk(P.R, P.X, setindex!(P.Y, v, i...))
size(P::PolarRandomWalk) = size(P.X)
view(P::PolarRandomWalk, i...) = PolarRandomWalk(P.R, view(P.X, i...), view(P.Y, i...))

function (p::PolarRandomWalkGenerator)(n=100)
    T = 2π
    ϕ = range(0, T, length=n)
    Δ = diff(ϕ)
    W = cumsum([0; sqrt.(Δ).*randn(length(ϕ)-1)])
    # Absolute value of a Brownian Bridge at r along phi
    B = similar(W)
    B[1] = 0
    for i in 2:length(B)-1
        B[i] = p.σr*( W[i] - ϕ[i]/T*W[end])
    end
    B[end] = 0
    B .+= p.r
    B ./= p.R
    PolarRandomWalk(p.R, ϕ, clamp.(B, -pi/2+1/p.R^2, pi/2-1/p.R^2))
end 

LinearInterpolation(P::PolarRandomWalk) = LinearInterpolation(P.X, P.Y; extrapolation_bc=Periodic())

Spherical(P::PolarRandomWalk) = Spherical.(P.R, P.X, P.Y)

isinside(p, P::PolarRandomWalk) = isinside(SphericalFromCartesian()(p), LinearInterpolation(P))
isinside(p::Spherical, P::Extrapolation) = P(p.θ) <= p.ϕ

# -- END MODULE -- #
end
