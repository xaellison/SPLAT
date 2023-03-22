using Tullio
using ForwardDiff
using Serialization
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

    r = rays[dest_idx]

    arg_min = default
    min_val = Inf32

    if threadIdx().x + (blockIdx().y - 1) * blockDim().x <= length(n_tris)
        shmem[threadIdx().x] = n_tris[threadIdx().x + (blockIdx().y - 1) * blockDim().x]
    else
        shmem[threadIdx().x] = 1, Tri(zero(ℜ³), zero(ℜ³), zero(ℜ³), zero(ℜ³))
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

    operand = monotonic_reinterpret(UInt64, (min_val, UInt32(arg_min)))
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
    dest_view .= retreive_arg.(src_view)
    return
end

## ExperimentalHitter2
# theoretical max occupancy 50% on rtx 2070

function next_hit_kernel2(rays :: AbstractArray{R}, n_tris :: AbstractArray{X}, dest :: AbstractArray{UInt64}, default) where {R, X}
    ray_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    shmem = CuDynamicSharedArray(FastRay, blockDim().x)
    shmem[threadIdx().x] = ray_idx <= length(rays) ? FastRay(rays[ray_idx]) : zero(FastRay)
    sync_threads()
    tri_idx = threadIdx().x + (blockIdx().y - 1) * blockDim().x

    
    i = zero(Int32)
    T = Tri(zero(ℜ³), zero(ℜ³), zero(ℜ³), zero(ℜ³))
    if tri_idx <= length(n_tris)
        i, T = n_tris[tri_idx]
    end

    @inbounds for warp_iter in 1:(blockDim().x ÷ 32)
        # download data for warp
        
        r = shmem[(warp_iter - 1) * 32 + laneid()]#rays[ray_idx]
        sync_warp()
        arg_min = default
        min_val = Inf32

        # time to shuffle rays

        for shuffle in 1:32
            t = distance_to_plane(r, T)
            p = r.pos + r.dir * t
            if i != 1 && in_triangle(p, T) && min_val > t > 0 && r.ignore_tri != i
                arg_min = i
                min_val = t
            end
            sync_warp()
            
            arg_min = CUDA.shfl_sync(0xFFFFFFFF, arg_min, laneid() % 32 + 1)
            min_val = CUDA.shfl_sync(0xFFFFFFFF, min_val, laneid() % 32 + 1)
            p_x = CUDA.shfl_sync(0xFFFFFFFF, r.pos[1], laneid() % 32 + 1)
            p_y = CUDA.shfl_sync(0xFFFFFFFF, r.pos[2], laneid() % 32 + 1)
            p_z = CUDA.shfl_sync(0xFFFFFFFF, r.pos[3], laneid() % 32 + 1)
            d_x = CUDA.shfl_sync(0xFFFFFFFF, r.dir[1], laneid() % 32 + 1)
            d_y = CUDA.shfl_sync(0xFFFFFFFF, r.dir[2], laneid() % 32 + 1)
            d_z = CUDA.shfl_sync(0xFFFFFFFF, r.dir[3], laneid() % 32 + 1)
            ig = CUDA.shfl_sync(0xFFFFFFFF, r.ignore_tri, laneid() % 32 + 1)
            r = FastRay(ℜ³(p_x, p_y, p_z), ℜ³(d_x, d_y, d_z), ig)
        end 

        # r should be its original value - upload
        operand = monotonic_reinterpret(UInt64, (min_val, UInt32(arg_min)))
        CUDA.@atomic dest[ray_idx] = min(dest[ray_idx], operand)
    end
    return nothing
end


function next_hit!(tracer, hitter::ExperimentalHitter2, rays, n_tris)
    begin
    # fuzzy req: length(rays) should = 0 mod 128/256/512
    my_args = rays, n_tris, hitter.tmp, Int32(1)
    kernel = @cuda launch = false next_hit_kernel2(my_args...)
    src_view = @view hitter.tmp[1:length(rays)]
    src_view .= typemax(UInt64)
    # TODO: this is running <= 50% occupancy. Need to put a cap on shmem smaller than block
    get_shmem(threads) = threads * sizeof(FastRay)
    config = launch_configuration(kernel.fun, shmem = threads -> get_shmem(threads))
    
    threads = config.threads#256#1 << exponent(config.threads)
    #threads = 32
    #@assert length(rays) % threads == 0
    # the totally confusing flip of xy for ray/tri at the block/grid level
    # is to keep grid size within maximum but also tris along thread_x (warp)
    blocks = (cld(length(rays), threads), cld(length(n_tris), threads))
    kernel(my_args...; blocks = blocks, threads = threads, shmem=get_shmem(threads))
    dest_view = @view tracer.hit_idx[1:length(rays)]
    dest_view .= retreive_arg.(src_view)
    end
    return
end


## ExperimentalHitter3
# Higher theoretical occupancy than EH2, performs slightly worse


function next_hit_kernel3(rays :: AbstractArray{R}, n_tris :: AbstractArray{X}, dest :: AbstractArray{UInt64}, default) where {R, X}
    #

    # virtual blocks!
    v_x = (threadIdx().x - 1) % 32 + 1
    v_y = (threadIdx().x - 1) ÷ 32 + 1
    v_bx = 32
    v_by = blockDim().x ÷ 32
    ray_idx = v_x + (v_y - 1) * v_bx + (blockIdx().x - 1) * v_bx * v_by
    tri_idx = v_x + (blockIdx().y - 1) * v_bx

    # download data for warp
    r = rays[ray_idx]
    arg_min = default
    min_val = Inf32

    i = zero(Int32)
    T = Tri(zero(ℜ³), zero(ℜ³), zero(ℜ³), zero(ℜ³))
    if tri_idx <= length(n_tris)
        i, T = n_tris[tri_idx]
    end

    # time to shuffle rays

    @inbounds for shuffle = 1:32
        t = distance_to_plane(r, T)
        p = r.pos + r.dir * t
        if i != 1 && min_val > t > 0 && r.ignore_tri != i && in_triangle(p, T)
            arg_min = i
            min_val = t
        end
        #t = distance_to_plane(r, T)

        #sync_warp()

        arg_min = CUDA.shfl_sync(0xFFFFFFFF, arg_min, laneid() % 32 + 1)
        min_val = CUDA.shfl_sync(0xFFFFFFFF, min_val, laneid() % 32 + 1)
        p_x = CUDA.shfl_sync(0xFFFFFFFF, r.pos[1], laneid() % 32 + 1)
        p_y = CUDA.shfl_sync(0xFFFFFFFF, r.pos[2], laneid() % 32 + 1)
        p_z = CUDA.shfl_sync(0xFFFFFFFF, r.pos[3], laneid() % 32 + 1)
        d_x = CUDA.shfl_sync(0xFFFFFFFF, r.dir[1], laneid() % 32 + 1)
        d_y = CUDA.shfl_sync(0xFFFFFFFF, r.dir[2], laneid() % 32 + 1)
        d_z = CUDA.shfl_sync(0xFFFFFFFF, r.dir[3], laneid() % 32 + 1)
        ig = CUDA.shfl_sync(0xFFFFFFFF, r.ignore_tri, laneid() % 32 + 1)
        r = FastRay(ℜ³(p_x, p_y, p_z), ℜ³(d_x, d_y, d_z), ig)
    end

    # r should be its original value - upload
    operand = monotonic_reinterpret(UInt64, (min_val, UInt32(arg_min)))
    CUDA.@atomic dest[ray_idx] = min(dest[ray_idx], operand)
    return nothing
end


function next_hit!(tracer, hitter::ExperimentalHitter3, rays, n_tris)
    # fuzzy req: length(rays) should = 0 mod 128/256/512
    my_args = rays, n_tris, hitter.tmp, Int32(1)
    kernel = @cuda launch = false next_hit_kernel3(my_args...)
    src_view = @view hitter.tmp[1:length(rays)]
    src_view .= typemax(UInt64)
    # TODO: this is running <= 50% occupancy. Need to put a cap on shmem smaller than block
    config = launch_configuration(kernel.fun)
    threads = 1 << exponent(config.threads) 
    threads = 32 #threads ÷ 2
   # @info "threads exp3 = $threads"
    #threads = config.threads
    #@assert length(rays) % threads == 0
    # the totally confusing flip of xy for ray/tri at the block/grid level
    # is to keep grid size within maximum but also tris along thread_x (warp)
    blocks = (cld(length(rays), threads), cld(length(n_tris), 32))
    kernel(my_args...; blocks = blocks, threads = threads)
    dest_view = @view tracer.hit_idx[1:length(rays)]
    dest_view .= retreive_arg.(src_view)
    return
end


## BV Hitter
# Make use of 1-level bounding volume 'tree' (shrub?)


# copy of next_hit_kernel3 with index_view addition
function next_hit_kernel4(rays :: AbstractArray{R}, index_view, n_tris :: AbstractArray{X}, dest :: AbstractArray{UInt64}, default) where {R, X}
    meta_index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    ray_idx = meta_index <= length(index_view) ? index_view[meta_index] : 1
    tri_idx = threadIdx().x + (blockIdx().y - 1) * blockDim().x
    shmem = CuDynamicSharedArray(Tuple{Int, FastRay}, blockDim().x)
    # download data for warp
    
    if meta_index <= length(index_view) && ray_idx <= length(rays)
        shmem[threadIdx().x] = ray_idx, FastRay(rays[ray_idx])
    else
        shmem[threadIdx().x] = typemax(Int), zero(FastRay)
    end
    sync_threads()
    i = zero(Int32)
    T = Tri(zero(ℜ³), zero(ℜ³), zero(ℜ³), zero(ℜ³))
    if tri_idx <= length(n_tris)
        i, T = n_tris[tri_idx]
    end

    # each warp should pull a different chunk of rays so atomic ops don't clash
    original_warp = (threadIdx().x - 1) ÷ 32
    warps_in_block = blockDim().x ÷ 32
    for warp_shift in 1:warps_in_block
        ray_idx, r = shmem[((original_warp + warp_shift) % warps_in_block) * 32 + laneid()]
        sync_warp()
        arg_min = default
        min_val = Inf32

        # time to shuffle rays

        for _ in 1:32
            t = distance_to_plane(r, T)
            p = r.pos + r.dir * t
            if i != 1 && in_triangle(p, T) && min_val > t > 0 && r.ignore_tri != i
                arg_min = i
                min_val = t
            end
            sync_warp()

            arg_min = CUDA.shfl_sync(0xFFFFFFFF, arg_min, laneid() % 32 + 1)
            min_val = CUDA.shfl_sync(0xFFFFFFFF, min_val, laneid() % 32 + 1)
            p_x = CUDA.shfl_sync(0xFFFFFFFF, r.pos[1], laneid() % 32 + 1)
            p_y = CUDA.shfl_sync(0xFFFFFFFF, r.pos[2], laneid() % 32 + 1)
            p_z = CUDA.shfl_sync(0xFFFFFFFF, r.pos[3], laneid() % 32 + 1)
            d_x = CUDA.shfl_sync(0xFFFFFFFF, r.dir[1], laneid() % 32 + 1)
            d_y = CUDA.shfl_sync(0xFFFFFFFF, r.dir[2], laneid() % 32 + 1)
            d_z = CUDA.shfl_sync(0xFFFFFFFF, r.dir[3], laneid() % 32 + 1)
            ig = CUDA.shfl_sync(0xFFFFFFFF, r.ignore_tri, laneid() % 32 + 1)
            r = FastRay(ℜ³(p_x, p_y, p_z), ℜ³(d_x, d_y, d_z), ig)
        end

        # r should be its original value - upload
        operand = monotonic_reinterpret(UInt64, (min_val, UInt32(arg_min)))
        if ray_idx <= length(rays)
            CUDA.@atomic dest[ray_idx] = min(dest[ray_idx], operand)
        end
    end
    return nothing
end


function next_hit!(tracer, hitter::ExperimentalHitter4, rays, index_view, n_tris)
    # fuzzy req: length(index_view) should = 0 mod 128/256/512
    my_args = rays, index_view, n_tris, hitter.tmp, Int32(1)
    kernel = @cuda launch = false next_hit_kernel4(my_args...)
    
    # TODO: this is running <= 50% occupancy. Need to put a cap on shmem smaller than block
    
    #threads = 32 #threads ÷ 2
   # @info "threads exp3 = $threads"
    threads = 32
    get_shmem(threads) = threads * (sizeof(Int) + sizeof(FastRay))
    config = launch_configuration(kernel.fun, shmem = threads -> get_shmem(threads))
    
    threads = config.threads#256#1 << exponent(config.threads)
    # non-maximal blocksize seems to perform slightly better
    threads = min(256, threads)
   #@assert length(rays) % threads == 0
   # the totally confusing flip of xy for ray/tri at the block/grid level
   # is to keep grid size within maximum but also tris along thread_x (warp)
    blocks = (cld(length(index_view), threads), cld(length(n_tris), threads))
    kernel(my_args...; blocks = blocks, threads = threads, shmem=get_shmem(threads))
    
  #  @assert false
    return
end



function queue_rays_kernel(rays, bvs :: AbstractArray{BV}, queue_index, queues, queue_counter) where BV
    
    ray_idx = threadIdx().x + (blockIdx().y - 1) * blockDim().x
    r = zero(FastRay)
    if 1 <= ray_idx <= length(rays)
        r = FastRay(rays[ray_idx])
    end
    
    
    bv = bvs[1]

    distance = get_hit((Int32(queue_index), bv), r)[1]
    condition = ! isinf(distance) && distance > 0
    thread_cumsum = 1 & condition

    for iter in 0:4
        Δ = 1 << iter
        δ = CUDA.shfl_up_sync(0xFFFFFFFF, thread_cumsum, Δ)
        thread_cumsum += laneid() - Δ > 0 ? δ : 0
    end
    block_write_floor = 0
    if laneid() == 32
        block_write_floor = CUDA.atomic_add!(pointer(queue_counter), thread_cumsum)
    end
    sync_warp()
    block_write_floor = CUDA.shfl_sync(0xFFFFFFFF, block_write_floor, 32)
    write_index = block_write_floor + thread_cumsum
    if condition && write_index <= size(queues)[2]
        queues[queue_index, write_index] = ray_idx
    end
    return
end


function next_hit!(tracer, hitter::BoundingVolumeHitter{BV}, rays, n_tris) where BV
    Q_block_size = 256

    CUDA.@sync begin
    tests = 0
    bv_count = length(hitter.bvs)
    concurrency = length(hitter.ray_queue_atomic_counters)
    device_bvs = CuArray(hitter.bvs)
    hitter.hitter.tmp .= monotonic_reinterpret(UInt64, (Inf32, UInt32(1)))

    @sync begin
        for task_index in 1:concurrency
            @async begin
                for bv_index in task_index:concurrency:bv_count
                    # within this block, we are focussed on a single bv
                    counter_view = @view hitter.ray_queue_atomic_counters[task_index]
                    counter_view .= 0
                    bv_view = @view device_bvs[bv_index]
                    @cuda blocks=(1, cld(length(rays), Q_block_size), ) threads=Q_block_size  queue_rays_kernel(rays, bv_view, task_index, hitter.ray_queues, counter_view)
                    rays_in_queue = Array(counter_view)[1]
                    padded_rays_in_queue = min(length(rays), (rays_in_queue ÷ 32 + 0) * 32) # WARNING padding disabled
                    ray_index_view = @view hitter.ray_queues[task_index, 1:padded_rays_in_queue]
                    if length(ray_index_view) >= 32
                        tri_view = @view n_tris[hitter.bv_tris[bv_index]]
                        next_hit!(tracer, hitter.hitter, rays, ray_index_view, tri_view)   
                        tests += length(ray_index_view) * length(tri_view)
                    end
                    
                end
                synchronize()
            end
        end
    end

   # @info "tests reduced -> $(tests / (length(rays) * length(n_tris)))"
    tracer.hit_idx .= retreive_arg.(hitter.hitter.tmp)
    end
    return
end


# DPBVHitter



function queue_rays_kernel2(rays, spheres, queues, queue_counter)
    
    ray_idx = threadIdx().x + (blockIdx().y - 1) * blockDim().x
    r = zero(FastRay)
    if 1 <= ray_idx <= length(rays)
        r = FastRay(rays[ray_idx])
    end
    
    
    bv = spheres[blockIdx().x]

    distance = get_hit((Int32(1), bv), r)[1]
    condition = ! isinf(distance) && distance > 0
    thread_cumsum = 1 & condition

    for iter in 0:4
        Δ = 1 << iter
        δ = CUDA.shfl_up_sync(0xFFFFFFFF, thread_cumsum, Δ)
        thread_cumsum += laneid() - Δ > 0 ? δ : 0
    end
    block_write_floor = 0
    if laneid() == 32
        block_write_floor = CUDA.atomic_add!(pointer(queue_counter, blockIdx().x), thread_cumsum)
    end
    sync_warp()
    block_write_floor = CUDA.shfl_sync(0xFFFFFFFF, block_write_floor, 32)
    write_index = block_write_floor + thread_cumsum
    if condition && write_index <= size(queues)[2]
        queues[blockIdx().x, write_index] = ray_idx
    end
    return
end



function next_hit_kernel5(rays :: AbstractArray{R}, ray_index_view, n_tris :: AbstractArray{X}, tri_index_view, dest :: AbstractArray{UInt64}, default) where {R, X}
    ray_meta_index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    ray_idx = ray_meta_index <= length(ray_index_view) ? ray_index_view[ray_meta_index] : 1
    tri_meta_index = threadIdx().x + (blockIdx().y - 1) * blockDim().x
    tri_idx = tri_meta_index <= length(tri_index_view) ? tri_index_view[tri_meta_index] : 1

    shmem = CuDynamicSharedArray(Tuple{Int, FastRay}, blockDim().x)
    # download data for warp
    
    if ray_meta_index <= length(ray_index_view) && ray_idx <= length(rays)
        shmem[threadIdx().x] = ray_idx, FastRay(rays[ray_idx])
    else
        shmem[threadIdx().x] = typemax(Int), zero(FastRay)
    end
    sync_threads()
    i = zero(Int32)
    T = Tri(zero(ℜ³), zero(ℜ³), zero(ℜ³), zero(ℜ³))
    if tri_meta_index <= length(tri_index_view) && tri_idx <= length(n_tris)
        i, T = n_tris[tri_idx]
        # could assert i == tri_idx
    end

    # each warp should pull a different chunk of rays so atomic ops don't clash
    original_warp = (threadIdx().x - 1) ÷ 32
    warps_in_block = blockDim().x ÷ 32
    for warp_shift in 1:warps_in_block
        ray_idx, r = shmem[((original_warp + warp_shift) % warps_in_block) * 32 + laneid()]
        sync_warp()
        arg_min = default
        min_val = Inf32

        # time to shuffle rays

        for _ in 1:32
            t = distance_to_plane(r, T)
            p = r.pos + r.dir * t
            if i != 1 && in_triangle(p, T) && min_val > t > 0 && r.ignore_tri != i
                arg_min = i
                min_val = t
            end
            sync_warp()

            arg_min = CUDA.shfl_sync(0xFFFFFFFF, arg_min, laneid() % 32 + 1)
            min_val = CUDA.shfl_sync(0xFFFFFFFF, min_val, laneid() % 32 + 1)
            p_x = CUDA.shfl_sync(0xFFFFFFFF, r.pos[1], laneid() % 32 + 1)
            p_y = CUDA.shfl_sync(0xFFFFFFFF, r.pos[2], laneid() % 32 + 1)
            p_z = CUDA.shfl_sync(0xFFFFFFFF, r.pos[3], laneid() % 32 + 1)
            d_x = CUDA.shfl_sync(0xFFFFFFFF, r.dir[1], laneid() % 32 + 1)
            d_y = CUDA.shfl_sync(0xFFFFFFFF, r.dir[2], laneid() % 32 + 1)
            d_z = CUDA.shfl_sync(0xFFFFFFFF, r.dir[3], laneid() % 32 + 1)
            ig = CUDA.shfl_sync(0xFFFFFFFF, r.ignore_tri, laneid() % 32 + 1)
            r = FastRay(ℜ³(p_x, p_y, p_z), ℜ³(d_x, d_y, d_z), ig)
        end

        # r should be its original value - upload
        operand = monotonic_reinterpret(UInt64, (min_val, UInt32(arg_min)))
        if ray_idx <= length(rays)
            CUDA.@atomic dest[ray_idx] = min(dest[ray_idx], operand)
        end
    end
    return nothing
end



function dp_launcher_kernel(bv_floor, child_threads, child_shmem, bv_tri_count, bv_tris, ray_queue_atomic_counters, rays, queues, n_tris, tmp, default)
    bv_index = bv_floor + threadIdx().x - 1
    queue_index = threadIdx().x
    rays_in_queue = ray_queue_atomic_counters[queue_index]
    # this min should never be necessary?
    padded_rays_in_queue = min(length(rays), rays_in_queue)
    threads=child_threads
    
    if padded_rays_in_queue <= 0
        return
    end
    queue_view = @view queues[queue_index, 1:padded_rays_in_queue]
    tri_counts = bv_tri_count[bv_index]
    tri_index_view = @view bv_tris[bv_index, 1:tri_counts]
    blocks = (cld(padded_rays_in_queue, threads), cld(tri_counts, threads))
    s = CuDeviceStream()
    @cuda dynamic=true stream=s  threads=threads shmem=child_shmem blocks=blocks next_hit_kernel5(rays, queue_view, n_tris, tri_index_view, tmp, default)
   
    CUDA.device_synchronize()
    return
end


function next_hit!(tracer, hitter::DPBVHitter{BV}, rays, n_tris) where BV
    Q_block_size = 256

    CUDA.@sync begin
    tests = 0
    bv_count = length(hitter.bvs)
    concurrency = length(hitter.ray_queue_atomic_counters)
    device_bvs = CuArray(hitter.bvs)
    hitter.tmp .= monotonic_reinterpret(UInt64, (Inf32, UInt32(1)))

    for bv_floor in 1:concurrency:bv_count
        hitter.ray_queue_atomic_counters .= 0
        bv_view = @view device_bvs[bv_floor: min(length(device_bvs), bv_floor + concurrency - 1)]
        # todo: replace loop with blocks
        let#for task_index in 1:concurrency
            hitter.ray_queue_atomic_counters .= 0
            @cuda blocks=(concurrency, cld(length(rays), Q_block_size), ) threads=Q_block_size queue_rays_kernel2(rays, bv_view, hitter.ray_queues, hitter.ray_queue_atomic_counters)
        end
        let 
            get_shmem(threads) = threads * (sizeof(Int) + sizeof(FastRay))
            child_threads = 256
            child_shmem = get_shmem(child_threads)
            @cuda threads=concurrency dp_launcher_kernel(bv_floor, child_threads, child_shmem, hitter.bv_tri_count, hitter.bv_tris, hitter.ray_queue_atomic_counters, rays, hitter.ray_queues, n_tris, hitter.tmp, Int32(1))
            
        end

       # tests += sum(Array(hitter.ray_queue_atomic_counters))
    end

   # @info "tests reduced -> $(tests / (length(rays) * length(n_tris)))"
    tracer.hit_idx .= retreive_arg.(hitter.tmp)
    end
    return
end