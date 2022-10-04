using Tullio
using ForwardDiff

## Geometric functions

## Tullio-based StableHitter

function typemax(::Type{Tuple{Tuple{Float32,Float32},Int32}})
    return ((Inf32, Inf32), one(Int32))
end

function typemax(::Type{Tuple{Float32,Int32}})
    return (Inf32, one(Int32))
end

function hit_argmin(i_T, r::ADRay)::Tuple{Float32,Int32}
    return hit_argmin(i_T, FastRay(r))
end

function hit_argmin(i_T, r::FastRay)::Tuple{Float32,Int32}
    return get_hit(i_T, r)[1:2]
end

function next_hit!(tracer, hitter::StableHitter, rays, n_tris::AbstractArray{X}) where {X}
    tmp_view = @view hitter.tmp[1:length(rays)]
    @tullio (min) tmp_view[i] = hit_argmin(n_tris[j], rays[i])
    d_view = @view tracer.hit_idx[:]
    d_view = reshape(d_view, length(d_view))
    map!(x -> x[2], d_view, tmp_view)
end

## ExperimentalHitter

function next_hit_kernel(rays, n_tris :: AbstractArray{X}, dest :: AbstractArray{UInt64}, default ) where {X}
    # TODO: rename everything
    shmem = CuDynamicSharedArray(Tuple{Int32, Tri}, blockDim().x)

    dest_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if dest_idx > length(rays)
        return
    end
    r = rays[dest_idx]

    arg_min = default
    min_val = Inf32

    if threadIdx().x + (blockIdx().y - 1) * blockDim().x <= length(n_tris)
        shmem[threadIdx().x] = n_tris[threadIdx().x + (blockIdx().y - 1) * blockDim().x]
    else
        shmem[threadIdx().x] = 1, zero(Tri)
    end
    sync_threads()
    for scan = 1:blockDim().x
        i, T = shmem[scan]
        t = distance_to_plane(r, T)
        p = r.pos + r.dir * t
        if in_triangle(p, T) && min_val > t > 0 && r.ignore_tri != i && i != 1
            arg_min = i
            min_val = t
        end
    end

    operand = unsafe_encode(min_val, UInt32(arg_min))
    CUDA.@atomic dest[dest_idx] = min(dest[dest_idx], operand)
    return nothing
end


function next_hit!(tracer, hitter::ExperimentalHitter, rays, n_tris)
    tmp_view = @view hitter.tmp[1:length(rays)]
    my_args = rays, n_tris, tmp_view, Int32(1)

    kernel = @cuda launch = false next_hit_kernel(my_args...)
    src_view = @view hitter.tmp[1:length(rays)]
    src_view .= typemax(UInt64)
    get_shmem(threads) = threads * sizeof(Tuple{Int32,Tri})
    config = launch_configuration(kernel.fun, shmem = threads -> get_shmem(threads))
    threads = 1 << exponent(config.threads)
    blocks = (cld(length(rays), threads), cld(length(n_tris), threads))
    kernel(my_args...; blocks = blocks, threads = threads, shmem = get_shmem(threads))
    dest_view = @view tracer.hit_idx[1:length(rays)]
    dest_view .= unsafe_decode.(src_view)
    return
end

## ExperimentalHitter2

function next_hit_kernel2(rays, n_tris :: AbstractArray{X}, dest :: AbstractArray{UInt64}, default ) where {X}
    ray_idx = threadIdx().y + (blockIdx().x - 1) * blockDim().y
    tri_idx = threadIdx().x + (blockIdx().y - 1) * blockDim().x

    r = rays[ray_idx]

    arg_min = default
    min_val = Inf32

    i = zero(Int32)
    T = Tri(zero(ℜ³), zero(ℜ³), zero(ℜ³), zero(ℜ³))
    if tri_idx <= length(n_tris)
        i, T = n_tris[tri_idx]#[1]
    #    T = Tri(n_tris[tri_idx][2][1], n_tris[tri_idx][2][2], n_tris[tri_idx][2][3], n_tris[tri_idx][2][4])
        #T = Tri(FT[1], FT[2], FT[3], FT[4])
    end

    t = distance_to_plane(r, T)
    p = r.pos + r.dir * t
    if in_triangle(p, T) && min_val > t > 0 && r.ignore_tri != i
        arg_min = i
        min_val = t
    end


    for shuffle_step in 0:4
        sync_warp()
        neighbor_arg_min = shfl_down_sync(0xFFFFFFFF, arg_min, 1<<shuffle_step)
        neighbor_min_val = shfl_down_sync(0xFFFFFFFF, min_val, 1<<shuffle_step)
        if neighbor_min_val < min_val
            arg_min = neighbor_arg_min
            min_val = neighbor_min_val
        end
    end

    if threadIdx().x == 1
        operand = unsafe_encode(min_val, UInt32(arg_min))
        CUDA.@atomic dest[ray_idx] = min(dest[ray_idx], operand)
    end
    return nothing
end


function next_hit!(tracer, hitter::ExperimentalHitter2, rays, n_tris)
    # fuzzy req: length(rays) should = 0 mod 128/256/512
    my_args = rays, n_tris, hitter.tmp, Int32(1)

    kernel = @cuda launch = false next_hit_kernel2(my_args...)
    hitter.tmp .= typemax(UInt64)
    # TODO: this is running <= 50% occupancy. Need to put a cap on shmem smaller than block
    config = launch_configuration(kernel.fun)
    thread_count = 1 << exponent(config.threads)
    threads = (32, thread_count ÷ 32)
    #@assert length(rays) % threads == 0
    # the totally confusing flip of xy for ray/tri at the block/grid level
    # is to keep grid size within maximum but also tris along thread_x (warp)
    blocks = (cld(length(rays), threads[2]), cld(length(n_tris), threads[1]))
    kernel(my_args...; blocks = blocks, threads = threads)
    tracer.hit_idx .= unsafe_decode.(hitter.tmp)
    return
end
