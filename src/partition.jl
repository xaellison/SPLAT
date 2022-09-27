using CUDA
using ..CUDA: i32
import CUDA.QuickSortImpl.cumsum!
import CUDA.QuickSortImpl.flex_lt
include("geo.jl")

function batch_partition(
    values,
    dest,
    atomic_floor,
    atomic_ceil,
    write_floor,
    write_ceil,
    pivot,
    swap,
    sums,
    lo,
    hi,
    parity,
    lt::F1,
    by::F2,
) where {F1,F2}
    sync_threads()
    idx0 = lo + (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    @inbounds if idx0 <= hi
        val = values[idx0]
        comparison = flex_lt(pivot, val, parity, lt, by)
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
    @inbounds if idx0 <= hi
        N_hi = sums[end]
        N_lo = blockDim().x - N_hi
        if threadIdx().x <= N_lo
            write_view = @view dest[write_floor[1]: write_floor[1] + N_lo]
            write_view[threadIdx().x] = swap[threadIdx().x]
        else
            c = write_ceil[1]
            write_view = @view dest[c- sums[end]:c]
            write_view[threadIdx().x - N_lo] = swap[threadIdx().x]
        end
    end
    sync_threads()
end


function partition_batches_kernel(
    values::AbstractArray{T},
    dest,
    atomic_floor,
    atomic_ceil,
    pivot,
    lo,
    hi,
    parity,
    lt::F1,
    by::F2,
) where {T,F1,F2}
    sums = CuDynamicSharedArray(Int, blockDim().x)
    swap = CuDynamicSharedArray(T, blockDim().x, sizeof(sums))
    write_floor = CuDynamicSharedArray(Int, 1, sizeof(sums) + sizeof(swap))
    write_ceil = CuDynamicSharedArray(Int, 1, sizeof(sums) + sizeof(swap) + sizeof(write_floor))
    batch_partition(values, dest, atomic_floor, atomic_ceil, write_floor, write_ceil, pivot, swap, sums, lo, hi, parity, lt, by)
    return
end

function partition(
    vals::AbstractArray{T},
    pivot,
    lo,
    hi,
    parity,
    lt::F1,
    by::F2,
) where {T,F1,F2}
    L = hi - lo
    block_dim = 256

    atomic_ceil = CUDA.zeros(Int, 1)
    atomic_ceil .= L
    atomic_floor = CUDA.zeros(Int, 1)
    dest = CUDA.zeros(eltype(vals), length(vals))
    @cuda(
        blocks = cld(L, block_dim),
        threads = block_dim,
        shmem = block_dim * (sizeof(Int) + sizeof(T)) + 2 * sizeof(Int),
        partition_batches_kernel(vals, dest, atomic_floor, atomic_ceil, pivot, lo, hi, parity, lt, by)
    )
    synchronize()
    @info atomic_floor
    @info atomic_ceil

    @info atomic_floor[1] + atomic_ceil[1]
    return dest
end

function partition(vals, pivot, lt, by)
    return partition(vals, pivot, 0, length(vals), false, lt, by)
end

function main()
    CUDA.NVTX.@range "warmup" CUDA.@sync begin
        c = CUDA.zeros(ADRay, 1_000_000)
        c = retire.(c, CUDA.rand(UInt8, length(c)) .% 3)
        partition(c, zero(ADRay), 0, length(c), false, isless, r -> r.status)
    end
    for N in [256^2, 512^2, 1024^2]
        CUDA.NVTX.@range "test $(Int(sqrt(N)))^2" CUDA.@sync begin
            c = CUDA.zeros(ADRay, N)
            c = retire.(c, CUDA.rand(UInt8, length(c)) .% 3)
            partition(c, zero(ADRay), 0, length(c), false, isless, r -> r.status)
            c = CUDA.zeros(ADRay, N)
            c = retire.(c, CUDA.rand(UInt8, length(c)) .% 3)
            partition(c, zero(ADRay), 0, length(c), false, isless, r -> r.status)

        end
    end
end


main()
