module AnalysisMethods

using DataFrames

export  timeserieses,
        collect_to_dataframe,
        popsize_timeseries_per_genotype,
        ps,
        padright


## Methods for extraction of observables
## like converting to timeseries and reshaping.

function timeserieses(X::AbstractArray)
    map(X) do x
        D = Dict{Symbol,Any}()
        for m in x
            if !haskey(D, m.name)
                push!(D, m.name => Union{typeof(m.val), Missing}[m.val])
            else
                push!(D[m.name], m.val)
            end
        end
        # Flatten all arrays of length 1
        for d in D
            if length(d[2])==1
                D[d[1]] = d[2][1]
            end
        end
        D
    end
end

function _genotype(x,g)
    genotypes = map(x->x[1], x)
    idx = findfirst(s->s==g, genotypes)
    if idx !== nothing
        return x[idx][2]
    else
        return 0.0
    end
end

"""
    ps(pop_size_entry,g)

Timeseries for genotype `g`.
Takes a single entry _not_ the whole DataFrame as first argument.
Broadcast like `ps.(X.pop_size,g)`.
"""
function ps(X,g)
    # test if X is flattened or not
    if typeof(X) <: Array{Array{T,1},1} where T # not flat
        return map(X) do x
            _genotype(x,g)
        end
    elseif X === missing
        return missing
    else
        return [_genotype(X,g)] # wrap in array for consistency
    end
end

function padright(X::AbstractVector, val)
    Y = skipmissing(X)
    maxlen = maximum(length.(Y))
    foreach(Y) do x
        append!(x, fill(val, maxlen-length(x)))
    end
    X
end


function popsize_timeseries_per_genotype(X)
    tmp = Vector{Tuple{Int,Int,Float64}}(0)
    genotypes() = map(x->x[1], tmp)
    for (t,Xt) in enumerate(X)
        for gt in Xt
            push!(tmp, (gt[1],t,gt[2]))
        end
    end
    tmp
end



"Collect observables from simulation into a DataFrame. Expects a timeseries."
function collect_to_dataframe(X::Dict)
    Y = X[:observables]
    col_names = [:N, :s, :mu, :d, :f0]
    eltypes = [Float64, Float64, Float64, Float64, Float64]

    append!(col_names, keys(Y[1]))
    append!(eltypes, typeof.(values(Y[1])))

    df = DataFrame(eltypes,col_names,0)

    all_rows = Dict{Symbol,Any}(:N => X[:N], :s => X[:s], :mu => X[:mu], :d => X[:d], :f0 => X[:f0])

    for y in enumerate(Y)
        newrow = Dict()
        try
            newrow = merge(all_rows,y[2])
            push!(df,newrow)
        catch err
            @error "Encountered an empty row."
        end
    end

    return df
end


end
