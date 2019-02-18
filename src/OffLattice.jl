module OffLattice

export  Cell,
        FreeSpace,
        random_config,
        get_positions,
        get_genotypes,
        get_rates,
        totalrate,
        set_birthrates!,
        birthrates,
        new_position,
        delete_cell!,
        new_cell!


import GeometryTypes: Point,Point2f0,Point3f0
import StaticArrays: SVector
import LightGraphs, MetaGraphs

P2or3f0 = Union{Point2f0,Point3f0}

using OpenCL

try
    global cl_device = Main.cl_device
    global cl_ctx = Main.cl_ctx
    global cl_queue = Main.cl_queue
catch err
    @warn("Provide valid `cl_device`, `cl_ctx`, `cl_queue` in `Main` when
            importing module `OffLattice`!")
    throw(err)
end

const max_work_group_size = cl.info(cl_device, :max_work_group_size)|>Int
offlattice_source = read(joinpath("..","kernel","distMat.cl"))
offlattice_prog = cl.Program(cl_ctx, source=offlattice_source) |> cl.build!
distMat2_kernel = cl.Kernel(offlattice_prog, "distMat2");
update_distMat2_kernel = cl.Kernel(offlattice_prog, "update_distMat2");
birthRates_kernel = cl.Kernel(offlattice_prog, "birthRates");
gaussianWeights_kernel = cl.Kernel(offlattice_prog, "gaussianWeights");
addCell2_kernel = cl.Kernel(offlattice_prog, "addCell2");
deleteCell_kernel = cl.Kernel(offlattice_prog, "deleteCell");
surface_kernel = cl.Kernel(offlattice_prog, "surface");

mutable struct FreeSpace{G}
    MaxPopulation::Int32
    Phylo
    positions::Array{Float32,2}
    _positions_buf::OpenCL.cl.Buffer
    genotypes::Vector{G}
    _genotypes_buf::OpenCL.cl.Buffer
    birthrates::Vector{Float32}
    _birthrates_buf::OpenCL.cl.Buffer
    deathrates::Vector{Float32}
    _deathrates_buf::OpenCL.cl.Buffer

    dMat::Matrix{Float32}
    _dMat_buf::OpenCL.cl.Buffer
    _mask::Vector{UInt32}
    _mask_buf::OpenCL.cl.Buffer
    _local_buf
    _int_buf
    _locked::Bool
    function FreeSpace{G}(
        N,
        PH,
        A::Array{Float32,2},
        B::Vector{G},
        C,
        D,
        dMat,
        _mask
        ) where {G}
        _p_buf = OpenCL.cl.Buffer(cl.CL_float,cl_ctx, (:r,:use), hostbuf=A)
        _g_buf = OpenCL.cl.Buffer(G,cl_ctx, (:r,:use), hostbuf=B)
        _b_buf = OpenCL.cl.Buffer(cl.CL_float,cl_ctx, (:w,:use), hostbuf=C)
        _d_buf = OpenCL.cl.Buffer(cl.CL_float,cl_ctx, (:w,:use), hostbuf=D)
        _dMat_buf = OpenCL.cl.Buffer(cl.CL_float,cl_ctx, (:rw,:copy), hostbuf=dMat)
        _mask_buf = OpenCL.cl.Buffer(cl.CL_bool,cl_ctx, (:r,:use), hostbuf=_mask)
        _local_buf = OpenCL.cl.LocalMem(cl.CL_float,N)
        _int_buf = OpenCL.cl.Buffer(cl.CL_int,cl_ctx,(:w,:copy), hostbuf=zeros(Int32,N))
        new(N,A,_p_buf,B,_g_buf,C,_b_buf,D,_d_buf,dMat,_dMat_buf,_mask,_mask_buf,_local_buf,_int_buf
        )
    end
end
FreeSpace(dim, N) = FreeSpace{Int32}(
 N,                             # MaxPopulation
 zeros(Float32,dim,N),          # positions
 ones(Int32, N),                # genotypes
 zeros(Float32,N),              # birthrates
 zeros(Float32,N),              # deathrates
 zeros(Float32,N,N),            # distance matrix
 zeros(UInt32,N)                # mask
)

function sync_to_host!(FS::FreeSpace{T}) where {T}
    # FS.positions  = reshape(OpenCL.cl.read(cl_queue,FS._positions_buf),size(FS.positions))
    FS.birthrates = OpenCL.cl.read(cl_queue,FS._birthrates_buf)
    # FS.deathrates = OpenCL.cl.read(cl_queue,FS._deathrates_buf)
    # FS._mask      = OpenCL.cl.read(cl_queue,FS._mask_buf)
    #OpenCL.cl.finish(cl_queue)
    FS._locked = true
    nothing
end
function sync_to_device(FS::FreeSpace{T}) where {T}
    cl.write!(cl_queue, FS._positions_buf, FS.positions)
    cl.write!(cl_queue, FS._mask_buf, FS._mask)
    FS._locked = false
    nothing
end

function new_cell!(FS, pos::P2or3f0, g; d=0f0)
    newindex = findfirst(x->!Bool(x), FS._mask)
    if newindex == 0
        return
    end
    FS.positions[:,newindex] = pos
    FS.birthrates[newindex] = 0f0
    FS.deathrates[newindex] = d
    FS._mask[newindex] = UInt32(1)
    FS.genotypes[newindex] = g

    cl.set_args!(addCell2_kernel,
        FS._positions_buf,
        FS._genotypes_buf,
        FS._deathrates_buf,
        FS._birthrates_buf,
        FS._mask_buf,
        FS.MaxPopulation, Int32(newindex-1), pos, Int32(g), 0f0, d
    )
    evt = cl.enqueue_kernel(cl_queue, addCell2_kernel, 1,1)
    cl.wait(evt)

    cl.set_args!(update_distMat2_kernel, FS._dMat_buf, FS._positions_buf,FS._local_buf,UInt32(newindex-1))
    evt = cl.enqueue_kernel(cl_queue,update_distMat2_kernel,(FS.MaxPopulation,1),(min(FS.MaxPopulation,max_work_group_size),1))
    cl.wait(evt)
    nothing
end

function delete_cell!(FS,index)
    if length(FS.positions) >= index
        FS._mask[index] = UInt32(0)
    end
    cl.set_args!(deleteCell_kernel, FS._mask_buf, Int32(index-1))
    cl.enqueue_kernel(cl_queue, deleteCell_kernel, 1,1) |> cl.wait
    nothing
end

function get_positions(FS)
    # FS.positions[:,Bool.(FS._mask)]
    (FS.positions[:,j] for j in 1:size(FS.positions,2) if FS._mask[j]==1)
end
function get_genotypes(FS)
    ( FS.genotypes[j] for j in 1:size(FS.genotypes,1) if FS._mask[j]==1 )
end
function get_rates(FS)
    (FS.birthrates[Bool.(FS._mask)], FS.deathrates[Bool.(FS._mask)])
end
function distMatrix(positions::Array{Float32,2})
    dMat = Array{Float32,2}(size(positions,2),size(positions,2))
    @inbounds for i in 1:size(positions,2)
    for j in 1:size(positions,2)
        p_i = positions[:,i]
        p_j = positions[:,j]
        delta = p_i-p_j
        delta .= min.( abs.(delta), 1f0.-abs.(delta) )
        dMat[j,i] = norm(delta)
    end
    end
    dMat
end
distMatrix(FS::T) where T<:FreeSpace = distMatrix(FS.positions)

function setDistMatrix!(FS)
    cl.set_args!(distMat2_kernel, FS._dMat_buf, FS._positions_buf)
    cl.enqueue_kernel(cl_queue, distMat2_kernel, (FS.MaxPopulation,FS.MaxPopulation), (min(FS.MaxPopulation,max_work_group_size),1))
    cl.finish(cl_queue)
end

function totalrate(FS)
    r = 0.
    @inbounds for j in eachindex(FS._mask)
        if Bool(FS._mask[j])
            r += FS.birthrates[j]
            r += FS.deathrates[j]
        end
    end
    return r
end

@inline birthrate(density, max_density) = clamp(1f0 - density/max_density, 0f0,1f0)

function set_birthrates!(::Type{Val{false}},FS; rCell=1f0, sigma=1f0, max_density=1f0, basebr=1f0)
    cl.set_args!(birthRates_kernel,
     FS._birthrates_buf, FS._dMat_buf, FS._mask_buf,
     UInt32(FS.MaxPopulation), Float32(sigma^2), Float32(max_density), Float32(rCell)
    )
    cl.enqueue_kernel(cl_queue, birthRates_kernel,
        (1,FS.MaxPopulation),(1,min(FS.MaxPopulation,max_work_group_size))
    )
    cl.finish(cl_queue)
    nothing
end

function set_birthrates!(::Type{Val{true}},FS; rCell=1f0, sigma=1f0, max_density=1f0, basebr=1f0)
    @inbounds for p in 1:length(FS.positions)
        if Bool(FS._mask[p])
            FS.birthrates[p] = basebr*birthrate(gaussian_density(FS, p, dCell=rCell,a=100f0,sigma=sigma),max_density)
        end
    end
end

function birthrates(FS; rCell=1f0, sigma=1f0, max_density=1f0, basebr=1f0,a=100f0)
    map( (1:size(FS.positions,2))[Bool.(FS._mask)]) do x
        basebr*birthrate(gaussian_density(FS,x;dCell=rCell,a=a,sigma=sigma),max_density)
    end
end

"Weighs all neighbors at max. distance `a` with a Gaussian of variance `σ^2`. Utilizes CPU."
function gaussian_density(::Type{Val{true}},FS, index::Int; dCell=0f0, a=0f0,sigma=2*a)
    x = 0f0
    @inbounds for j in 1:size(FS.dMat,1)
        if Bool(FS._mask[j]) && FS.dMat[j,index] < a^2
            x += FS.dMat[j,index] > dCell ? exp(-1/2f0*(FS.dMat[j,index]^2-dCell^2)/sigma^2) : 1f0
        end
    end
    return x-1f0
end

function gaussian_density(::Type{Val{false}},FS; dCell=0f0, a=0f0,sigma=2*a, kwargs...)
    out_buf = cl.Buffer(cl.CL_float, cl_ctx, (:w,:alloc), FS.MaxPopulation)
    cl_queue(gaussianWeights_kernel,FS.MaxPopulation,min(FS.MaxPopulation,max_work_group_size),
        out_buf, FS._dMat_buf, FS._mask_buf, FS.MaxPopulation, Float32(sigma^2)
    )
    cl.read(cl_queue,out_buf)
end

function box(x, upper=1.0, lower=0.0)
    return min(max(lower,x),upper)
end

function new_position(pos::Point2f0, a, sigma)
    phi = rand()*2pi
    r = abs(sigma*randn()) + a
    nx = pos[1] + Float32(r*cos(phi))
    ny = pos[2] + Float32(r*sin(phi))

    if nx<0f0; nx = 1.0f0
    elseif nx>1f0; nx = nx-1f0 end
    if ny<0f0; ny = 1.0f0
    elseif ny>1f0; ny = ny-1f0 end

    return Point2f0(nx,ny)
end
function new_position(pos::Point3f0, a, sigma)
    phi = rand()*2pi
    theta = rand()*pi
    r = abs(sigma*randn()) + a
    return box.(pos + Point3f0(r*cos(phi)*sin(theta),r*sin(phi)*sin(theta),r*cos(theta)))
end

function random_config(N,d=3)
    FS = FreeSpace(d,N)
    for _ in 1:N
        new_cell!(FS, rand(Point{d,Float32}), 1)
    end
    return FS
end


end
