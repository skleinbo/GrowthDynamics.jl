using Base.Test

using OffLattice
using OpenCL

N = 256

@time FS = OffLattice.random_config(N,2)
cl.finish(OffLattice.cl_queue)
@time OffLattice.setDistMatrix!(FS)

rCell = 1/sqrt(FS.MaxPopulation)/2|>Float32

@test begin
    FS.dMat = reshape(OpenCL.cl.read(OffLattice.cl_queue, FS._dMat_buf), N,N)
    FS.dMat ≈ OffLattice.distMatrix(FS.positions)
end
@test begin
    OffLattice.sync_to_device(FS)
    OffLattice.set_birthrates!(FS, rCell=rCell,sigma=2f0*rCell, max_density=12f0)
    OffLattice.sync_to_host!(FS)
    # FS.dMat = reshape(cl.read(OffLattice.cl_queue, FS._dMat_buf), N,N)
    FS.birthrates ≈ OffLattice.birthrates(FS, rCell, 2f0*rCell, 12f0, 100f0)
end
