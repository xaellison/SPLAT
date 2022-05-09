using CUDA
## new gpu AD only

function next_hit_kernel(rays, tris :: AbstractArray{T}, dest :: AbstractArray{D}, default :: D) where {T, D}
    shmem = @cuDynamicSharedMem(T, blockDim().x)
    i = threadIdx().x
    dest_idx = i + (blockIdx().x - 1) * blockDim().x
    r = rays[dest_idx]
    cap = 0
    iter = 0
    arg_min = default
    min_val = Inf32
    while iter <= length(tris) ÷ blockDim().x
        if i + iter * blockDim().x <= length(tris)
            shmem[i] = tris[i+iter*blockDim().x]
        end

        sync_threads()
        for scan = 1:min(blockDim().x, length(tris) - iter * blockDim().x)
            sync_threads()
            n, t = shmem[scan]
            ####
            d0 = distance_to_plane(r.pos, r.dir, t[2], t[1])
            d(λ) = distance_to_plane(r.pos + r.pos′ * (λ - r.λ), r.dir + r.dir′ * (λ - r.λ), t[2], t[1])
            p = r.pos + r.dir * d0
            if in_triangle(p, t[2], t[3], t[4]) && min_val > d0 > 0 && r.ignore_tri != n
                arg_min = n
                min_val = d0
            end
            ####
        end
        iter += 1
    end
    if dest_idx <= length(rays)
        dest[dest_idx] = arg_min
    end
    return nothing
end


function next_hit!(dest :: CuArray{I}, rays, n_tris:: CuArray{Tuple{I, T}}, override) where {I, T}
    block_size = 256
    @assert length(rays) % block_size == 0
    blocks = cld(length(rays), block_size) |> Int
    @info blocks, block_size
    @cuda threads = block_size blocks = blocks shmem = (sizeof(I)+sizeof(T)) * block_size next_hit_kernel(
        rays,
        n_tris,
        dest,
        Int32(1), # this default is the degenerate triangle which no ray can hit
    )
    return dest
    return nothing
end
