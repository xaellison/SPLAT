"""
Borrows logic for cuda quicksort to perform a partition using O(N) global swap
"""

using CUDA
using ..CUDA: i32
import CUDA.QuickSortImpl.cumsum!
include("geo.jl")

function batch_partition(
    values,
    dest,
    atomic_floor,
    atomic_ceil,
    write_floor,
    write_ceil,
    swap,
    sums,
    lo,
    hi,
    parity,
    by::F,
) where {F}
    idx0 = lo + (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    @inbounds if idx0 <= hi
        val = values[idx0]
        comparison = by(val)
    end

    @inbounds if idx0 <= hi
        sums[threadIdx().x] = 1 & comparison
    else
        sums[threadIdx().x] = 1
    end
    sync_threads()

    cumsum!(sums)

    @inbounds if idx0 <= hi
        dest_idx = @inbounds comparison ? blockDim().x - sums[end] + sums[threadIdx().x] :
                  threadIdx().x - sums[threadIdx().x]
        if dest_idx <= length(swap)
            swap[dest_idx] = val
        end
    end
    sync_threads()

    if threadIdx().x == 1 && idx0 <= hi
        write_ceil[1] = CUDA.atomic_add!(pointer(atomic_ceil, 1), -sums[end])
        write_floor[1] = CUDA.atomic_add!(pointer(atomic_floor, 1), blockDim().x - sums[end])
    end
    sync_threads()
    overhang = gridDim().x * blockDim().x - hi
    @inbounds if idx0 <= hi
        N_hi = sums[end]
        N_lo = blockDim().x - N_hi
        if threadIdx().x <= N_lo
            write_view = @view dest[write_floor[1] + 1: write_floor[1] + 1 + N_lo]
            write_view[threadIdx().x] = swap[threadIdx().x]
        else
            c = write_ceil[1]
            write_view = @view dest[c - sums[end] + overhang + 1:c + 1]
            write_view[threadIdx().x - N_lo] = swap[threadIdx().x]
        end
    end
end


function partition_batches_kernel(
    values::AbstractArray{T},
    dest,
    atomic_floor,
    atomic_ceil,
    lo,
    hi,
    parity,
    by::F,
) where {T,F}
    sums = CuDynamicSharedArray(Int, blockDim().x)
    swap = CuDynamicSharedArray(T, blockDim().x, sizeof(sums))
    write_floor = CuDynamicSharedArray(Int, 1, sizeof(sums) + sizeof(swap))
    write_ceil = CuDynamicSharedArray(Int, 1, sizeof(sums) + sizeof(swap) + sizeof(write_floor))
    batch_partition(values, dest, atomic_floor, atomic_ceil, write_floor, write_ceil, swap, sums, lo, hi, parity, by)
    return
end


"""
Copies all values, partitioned (sort of boolean values) from `vals` into `dest`

Returns a length-1 CuArray containing the number of false values in the first
half of `dest`
"""
function partition_copy!(
    dest::AbstractArray{T},
    vals::AbstractArray{T},
    by::F,
) where {T,F}
    L = length(vals)
    block_dim = 256
    atomic_ceil = CUDA.zeros(Int, 1)
    atomic_ceil .= L
    atomic_floor = CUDA.zeros(Int, 1)
    @cuda(
        blocks = cld(L, block_dim),
        threads = block_dim,
        shmem = block_dim * (sizeof(Int) + sizeof(T)) + 2 * sizeof(Int),
        partition_batches_kernel(vals, dest, atomic_floor, atomic_ceil, 0, L, false, by)
    )

    return atomic_floor
end

"""
Partitions `vals` according to `by`, `lt`. Overwrites `swap` as scratch space.
`swap` may be larger than `vals`.
Returns an Integer, the number of false values in the first half of `vals`
"""
function partition!(vals, swap; by)
    partition_holder = partition_copy!(swap, vals, by)
    # faster than .=
    v =  @view swap[1:length(vals)]
#    vals .= v
    copy!(vals, v)
    #vals .=
    partition = Array(partition_holder)[1]
    # TODO - use unified memory
    return partition
end
