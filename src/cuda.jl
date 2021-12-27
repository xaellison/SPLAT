
function next_hit_kernel(rays, tris, dest, default)
    shmem = @cuDynamicSharedMem(STri, blockDim().x)
    i = threadIdx().x
    dest_idx = i + (blockIdx().x - 1) * blockDim().x
    r = rays[dest_idx]
    cap = 0
    iter = 0
    arg_min = default
    min_val = Inf32
    while iter <= length(tris) ÷ blockDim().x
        if i + iter * blockDim().x <= length(tris)
            shmem[i] = tris[i+iter*blockDim().x][2]
        end

        sync_threads()
        for scan = 1:min(blockDim().x, length(tris) - iter * blockDim().x)
            sync_threads()
            t = shmem[scan]
            d = distance_to_plane(r.pos, r.dir, t[2], t[1])
            p = r.pos + r.dir * d
            n = Int32(iter * blockDim().x + scan)
            if min_val > d > 0 && r.ignore_tri != n && in_triangle(p, t[2], t[3], t[4])
                min_val = d
                arg_min = (d, n, t)
            end
        end        iter += 1
    end
    if dest_idx <= length(rays)
        dest[dest_idx] = arg_min
    end
    return nothing
end

function next_hit(rays, n_tris)
    dest = CuArray{Tuple{Float32,Int,STri}}(undef, size(rays))
    blocks = length(rays) ÷ 256
    @cuda threads = 256 blocks = blocks shmem = sizeof(STri) * 256 next_hit_kernel(
        rays,
        n_tris,
        dest,
        (Inf32, typemax(Int32), zeros(STri)),
    )
    synchronize()
    return dest
end

function next_hit!(dest, rays, n_tris)
    blocks = length(rays) ÷ 256
    @cuda threads = 256 blocks = blocks shmem = sizeof(STri) * 256 next_hit_kernel(
        rays,
        n_tris,
        dest,
        (Inf32, typemax(Int32), zeros(STri)),
    )
    synchronize()
    return dest
end
