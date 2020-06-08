struct Plane
    p
    u
    v
    w
end
function Plane(p, w)
    if length(p)!=3 || length(w)!=3
        throw(ArgumentError("Arguments must be vectors of length 3."))
    end
    if iszero(w)
        throw(ArgumentError("Plane normal cannot not be zero."))
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
function Plane(p, u,v)
    if length(p)!=3 || length(u)!=3 || length(v)!=3
        throw(ArgumentError("Arguments must be vectors of length 3."))
    end
    if iszero(u) || iszero(v)
        throw(ArgumentError("Tangent vectors cannot not be zero."))
    end
    u = normalize(u)
    v = normalize(v)
    w = cross(u,v)
    return Plane(SVector{3}(p), SVector{3}(u), SVector{3}(v), SVector{3}(w))
end

function euclidean_dist(q, P::Plane)
    abs(dot((q - P.p), P.w))
end
