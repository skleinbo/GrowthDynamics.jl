## Defines a plane in three dimensional Euclidean space
## given by the origin `p`, and three vectors `u,v` in the plane
## and `w` perpendicular to it.
## `u,v,w` are not independent, but filled in by the constructors.

import Base: ==
import LinearAlgebra: cross, dot, norm, norm2, normalize
import GeometryBasics: Pointf0

struct Plane
    p
    u
    v
    w
end

"""
    Plane(p, w)

Construct a plane with origin `p` and normal `w`.
`w` will be normalized automatically.
"""
function Plane(p, w)
    if length(p)!=3 || length(w)!=3
        throw(ArgumentError("Arguments must be vectors of length 3."))
    end
    p = SVector{3}(p)
    w = SVector{3}(w)
    if iszero(w)
        throw(ArgumentError("Plane normal cannot be zero."))
    end
    # w is the plane normal
    w = normalize(w)
    if w[1]!=0.0 && w[2]!=0.0 && w[3]!=0.0
        u = normalize(@SVector [1.0, 1.0, -(w[1]+w[2])/w[3]])
    elseif w[1]!=0.0 && w[2]!=0.0
        u = normalize(SVector{3}(1.0, -w[1]/w[2], 0.0))
    elseif w[2]!=0.0 && w[3]!=0.0
        u = normalize(SVector{3}(0.0, 1.0, -w[2]/w[3],))
    elseif w[3]!=0.0 && w[1]!=0.0
        u = normalize(SVector{3}(-w[3]/w[1], 0.0, 1.0))
    elseif w[1]!=0.0
        u = SVector{3}(0.0, 1.0, 0.0)
    elseif w[2]!=0.0
        u = SVector{3}(0.0, 0.0, 1.0)
    elseif w[3]!=0.0
        u = SVector{3}(1.0, 0.0, 0.0)
    end
    v = -normalize(cross(u, w))
    return Plane(p, u,v,w)
end
"""
    Plane(p, u,v)

Construct a plane with origin `p` spanned by `u,v`.
Vectors will be normalized automatically.
"""
function Plane(p, u,v)
    if length(p)!=3 || length(u)!=3 || length(v)!=3
        throw(ArgumentError("Arguments must be of length 3."))
    end
    p = SVector{3}(p)
    u = SVector{3}(u)
    v = SVector{3}(v)

    u = normalize(u)
    v = v - dot(u,v)*u
    v = normalize(v)

    if iszero(u) || iszero(v)
        throw(ArgumentError("Tangent vectors cannot be zero."))
    end

    w = normalize(cross(u,v))
    return Plane(p, u,v,w)
end

function isvalid(P::Plane)
    isvalid = true
    if norm(P.u)≉ 1.0 || norm(P.v)≉ 1.0 || norm(P.w)≉ 1.0 || dot(P.u,P.w)!=0 || dot(P.v,P.w)!=0
        isvalid = false
    end
    if P.u == P.v || P.u == -P.v
        isvalid = false
    end
    isvalid
end

function ==(P1::Plane, P2::Plane)
    if !isvalid(P1) || !isvalid(P2)
        throw(ArgumentError("Invalid plane(s)."))
    end
    if P1.p ≉  P2.p
        return false
    end
    if !(P1.w ≈ P2.w || P1.w ≈ -P2.w)
        return false
    end
    return true
end

"""
    euclidean_dist(q, P::Plane)

Orthogonal Euclidean distance `| (q-p).w |` of point `q` from plane `P`.
"""
function euclidean_dist(q, P::Plane)
    abs(dot((q - P.p), P.w))
end

## 

"""
    volume(r, dim)

Volume of ball of radius `r` in `dim` dimensions.
"""
function volume(r, dim::Int) 
    if dim == 1
        return 2*r
    elseif dim == 2
        return π*r^2
    elseif dim == 3
        return 4π/3*r^3
    else
        throw(ArgumentError("dim >=4 not supported"))
    end
end
"""
    surface(r, dim)

Surface of ball of radius `r` in `1<=dim<=3` dimensions.
"""
function surface(r, dim::Int) 
    if dim == 1
        return 2
    elseif dim == 2
        return 2π*r
    elseif dim == 3
        return 4π*r^2
    else
        throw(ArgumentError("dim >=4 not supported"))
    end
end

"""
    radius(v, dim)

Radius of a ball of volume `v` in `dim` dimensions.
"""
function radius(v, dim::Int)
    if dim == 1
        return v/2
    elseif dim == 2
        return sqrt(v/π)
    elseif dim == 3
        return (3/4/π*v)^(1//3)
    else
        throw(ArgumentError("dim >=4 not supported"))
    end
end
