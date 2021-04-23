## Defines a plane in three dimensional Euclidean space
## given by the origin `p`, and three vectors `u,v` in the plane
## and `w` perpendicular to it.
## `u,v,w` are not independent, but filled in by the constructors.

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
    if iszero(w)
        throw(ArgumentError("Plane normal cannot be zero."))
    end
    # w is the plane normal
    w = normalize(w)
    if w[2]!=0.0
        u = normalize([1.0, -w[1]/w[2], 0.0])
    elseif w[3]!=0.0
        u = normalize([1.0, -w[1]/w[3], 0.0])
    else
        u = [0.0, 1.0, 0.0]
    end
    v = cross(u, w)
    return Plane(SVector{3}(p), SVector{3}(u), SVector{3}(v), SVector{3}(w))
end
"""
    Plane(p, u,v)

Construct a plane with origin `p` spanned by `u,v`.
Vectors will be normalized automatically.
"""
function Plane(p, u,v)
    if length(p)!=3 || length(u)!=3 || length(v)!=3
        throw(ArgumentError("Arguments must be vectors of length 3."))
    end
    if iszero(u) || iszero(v)
        throw(ArgumentError("Tangent vectors cannot be zero."))
    end
    u = normalize(u)
    v = normalize(v)
    w = cross(u,v)
    return Plane(SVector{3}(p), SVector{3}(u), SVector{3}(v), SVector{3}(w))
end

"""
    euclidean_dist(q, P::Plane)

Euclidean distance `| (q-p).w |` of point `q` from plane `P`.
"""
function euclidean_dist(q, P::Plane)
    abs(dot((q - P.p), P.w))
end
