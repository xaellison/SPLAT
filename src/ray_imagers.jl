# Functions that take a set of evolved rays and convert them to image output
# (including light-map textures)

# The first group is shaders - given a ray, wavelength, hit: return intensity

function shade(r::FastRay, n, t, first_diffuse_index)::Float32
    # NB disregards in_medium
    # evolve to hit a diffuse surface
    d, n, t = get_hit((n, t), r)
    r = FastRay(r.pos + r.dir * d, zero(V3), r.ignore_tri)

    if n >= first_diffuse_index
        # compute the position in the new triangle, set dir to zero
        u, v = reverse_uv(r.pos, t)
        s = 0.5f0
        if xor(u % s > s / 2, v % s > s / 2)
            return 1.0f0
        else
            return 0.0f0
        end
    end
    if isinf(d)
        return 0.0f0# retire(r, RAY_STATUS_INFINITY)
    end
    return 0.0f0
end

function shade(adr::ADRay, n, t, first_diffuse_index, λ)::Float32
    # NB disregards in_medium
    # evolve to hit a diffuse surface
    r = expand(adr, λ)
    d, n, t = get_hit((n, t), r)
    r = FastRay(r.pos + r.dir * d, zero(V3), r.ignore_tri)

    if n >= first_diffuse_index
        # compute the position in the new triangle, set dir to zero
        u, v = reverse_uv(r.pos, t)
        s = 0.5f0
        if xor(u % s > s / 2, v % s > s / 2)
            return 1.0f0
        else
            return 0.0f0
        end
    end
    if isinf(d)
        return 0.0f0# retire(r, RAY_STATUS_INFINITY)
    end
    return 0.0f0
end


function shade_tex(r::FastRay, n, t, first_diffuse_index, tex :: AbstractArray{Float32})::Float32
    # NB disregards in_medium
    # evolve to hit a diffuse surface
    d, n, t = get_hit((n, t), r)
    r = FastRay(r.pos + r.dir * d, zero(V3), r.ignore_tri)

    if n >= first_diffuse_index
        # compute the position in the new triangle, set dir to zero
        u, v = tex_uv(r.pos, t)
        CUDA.@assert !isnan(u) && !isnan(v)
        # it's theoretically possible u, v could come back as zero
        i = clamp(Int(ceil(u * (size(tex)[1]))), 1, size(tex)[1])
        j = clamp(Int(ceil(v * (size(tex)[2]))), 1, size(tex)[2])
        return tex[i, j]
    end
    if isinf(d)
        return 0.0f0# retire(r, RAY_STATUS_INFINITY)
    end
    return 0.0f0
end

function shade_tex(adr::ADRay, n, t, first_diffuse_index, λ, tex :: AbstractArray{Float32})::Float32
    # NB disregards in_medium
    # evolve to hit a diffuse surface
    r = expand(adr, λ)
    d, n, t = get_hit((n, t), r, unsafe=true)
    r = FastRay(r.pos + r.dir * d, zero(V3), r.ignore_tri)

    if n >= first_diffuse_index
        # compute the position in the new triangle, set dir to zero
        u, v = tex_uv(r.pos, t)

        CUDA.@assert !isnan(u) && !isnan(v)
        # it's theoretically possible u, v could come back as zero
        i = clamp(Int(ceil(u * (size(tex)[1]))), 1, size(tex)[1])
        j = clamp(Int(ceil(v * (size(tex)[2]))), 1, size(tex)[2])
        return tex[i, j]
    end
    if isinf(d)
        return 0.0f0# retire(r, RAY_STATUS_INFINITY)
    end
    return 0.0f0
end

# the second group is functions that invoke a kernel of a group one function

function continuum_shade(RGB3, RGB, tris, hits, tmp, rays, n_tris, spectrum, expansion, first_diffuse, retina_factor, intensity, dλ, tex)
    RGB3 .= 0.0f0

    hits .= Int32(1)
    next_hit!(hits, tmp, rays, n_tris)
    tri_view = @view tris[hits]

    s(args...) = shade_tex(args..., tex)
    #s(args...) = shade(args...)
    broadcast = @~ s.(rays, hits, tri_view, first_diffuse, spectrum) .* retina_factor .* intensity .* dλ
    # broadcast rule not implemeted for sum!
    RGB3 .= sum(broadcast, dims=3) |> a -> reshape(a, length(rays), 3)

    map!(brightness -> clamp(brightness, 0, 1), RGB3, RGB3)
    RGB .= RGBf.(RGB3[:, 1], RGB3[:, 2], RGB3[:, 3])
end

function expansion_loop_shade(RGB3, RGB, tris, hits, tmp, rays, n_tris, spectrum, expansion, first_diffuse, retina_factor, intensity, dλ, tex)
    RGB3 .= 0.0f0
    for (n, λ) in enumerate(spectrum)
        begin
            expansion .= expand.(rays, λ)
            hits .= Int32(1)
            next_hit!(hits, tmp, expansion, n_tris)
            tri_view = @view tris[hits]
        end
        #@info "Stage 3: Images"
        begin
            s(args...) = shade_tex(args..., tex)
            #s(args...) = shade(args...)
            # I tried pre-allocating α and it made it slower
            α = s.(expansion, hits, tri_view, first_diffuse)
            broadcast = (α .* retina_factor[:, :, n] .* intensity .* dλ)
            RGB3 .+= broadcast |> a -> reshape(a, length(rays), 3)
        end
    end
    map!(brightness -> clamp(brightness, 0, 1), RGB3, RGB3)
    RGB .= RGBf.(RGB3[:, 1], RGB3[:, 2], RGB3[:, 3])
end


function atomic_light_kernel(rays ::AbstractArray{FastRay}, hits, tris, first_diffuse_index, spectrum, R, G, B, tex )
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    r = rays[idx]
    n = hits[idx]
    t = tris[idx]
    if n >= first_diffuse_index
        d, n, t = get_hit((n, t), r, unsafe=true)
        r = FastRay(r.pos + r.dir * d, zero(V3), r.ignore_tri)

        u, v = tex_uv(r.pos, t)
        if !isnan(u) && !isnan(v) && !isinf(u) && !isinf(v)
            # it's theoretically possible u, v could come back as zero
            A = 1024 # equal to image height
            i = clamp(Int(ceil(u * A)), 1, A)
            j = clamp(Int(ceil(v * A)), 1, A)

            CUDA.@atomic tex[(i - 1) * A + j, 1] += R
            CUDA.@atomic tex[(i - 1) * A + j, 2] += G
            CUDA.@atomic tex[(i - 1) * A + j, 3] += B

        end
    end
    nothing
end

function atomic_light_kernel(rays ::AbstractArray{ADRay}, hits, tris, first_diffuse_index, spectrum, retina, tex )
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    adr = rays[idx]
    n = hits[idx]
    t = tris[idx]
    if n >= first_diffuse_index
        for (n_λ, λ) in enumerate(spectrum)
            r = expand(adr, λ)
            d, n, t = get_hit((n, t), r, unsafe=true)
            r = FastRay(r.pos + r.dir * d, zero(V3), r.ignore_tri)

            u, v = tex_uv(r.pos, t)
            if !isnan(u) && !isnan(v) && !isinf(u) && !isinf(v)
                # it's theoretically possible u, v could come back as zero
                A = 1024
                i = clamp(Int(ceil(u * A)), 1, A)
                j = clamp(Int(ceil(v * A)), 1, A)

                CUDA.@atomic tex[(i - 1) * A + j, 1] += retina[1, 1, n_λ]
                CUDA.@atomic tex[(i - 1) * A + j, 2] += retina[1, 2, n_λ]
                CUDA.@atomic tex[(i - 1) * A + j, 3] += retina[1, 3, n_λ]

            end
        end
    end
    nothing
end

function light_map!(RGB3, RGB, tris, hits, tmp, rays, n_tris, spectrum, expansion, first_diffuse, retina_factor, intensity, dλ, tex)

    for (n_λ, λ) in enumerate(spectrum)
        expansion .= expand.(rays, λ)
        hits .= Int32(1)
        next_hit!(hits, tmp, expansion, n_tris)
        tri_view = @view tris[hits]

        @assert length(rays) % 256 == 0
        @assert length(rays) == length(hits) == length(tri_view)
        @assert eltype(retina_factor) == eltype(RGB3)
        @info size(RGB3)
        @cuda blocks=length(rays) ÷ 256 threads=256 atomic_light_kernel(expansion, hits, tri_view, first_diffuse, spectrum, retina_factor[1, :, n_λ]..., RGB3)
        synchronize()

    end

    RGB .= RGBf.(RGB3[:, 1], RGB3[:, 2], RGB3[:, 3])
end

function light_map2!(RGB3, RGB, tris, hits, tmp, rays, n_tris, spectrum, expansion, first_diffuse, retina_factor, intensity, dλ, tex)

    hits .= Int32(1)
    next_hit!(hits, tmp, rays, n_tris)
    tri_view = @view tris[hits]

    @assert length(rays) % 256 == 0
    @assert length(rays) == length(hits) == length(tri_view)
    @assert eltype(retina_factor) == eltype(RGB3)
    @info size(RGB3)
    @cuda blocks=length(rays) ÷ 256 threads=256 atomic_light_kernel(rays, hits, tri_view, first_diffuse, spectrum, retina_factor, RGB3)
    synchronize()

    RGB .= RGBf.(RGB3[:, 1], RGB3[:, 2], RGB3[:, 3])
end
