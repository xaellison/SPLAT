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
    r = FastRay(r.pos + r.dir * d, r.dir, r.ignore_tri)

    if n >= first_diffuse_index
        # compute the position in the new triangle, set dir to zero
        u, v = tex_uv(r.pos, t)
        CUDA.@assert !isnan(u) && !isnan(v)
        # it's theoretically possible u, v could come back as zero
        i = clamp(Int(ceil(u * (size(tex)[1]))), 1, size(tex)[1])
        j = clamp(Int(ceil(v * (size(tex)[2]))), 1, size(tex)[2])
        return tex[i, j] * cosine_shading(r, t)
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
    # WARNING inconsistent ray defined
    r = FastRay(r.pos + r.dir * d, r.dir, r.ignore_tri)

    if n >= first_diffuse_index
        # compute the position in the new triangle, set dir to zero
        u, v = tex_uv(r.pos, t)

        CUDA.@assert !isnan(u) && !isnan(v)
        # it's theoretically possible u, v could come back as zero
        i = clamp(Int(ceil(u * (size(tex)[1]))), 1, size(tex)[1])
        j = clamp(Int(ceil(v * (size(tex)[2]))), 1, size(tex)[2])
        return tex[i, j] * cosine_shading(r, t)
    end
    if isinf(d)
        return 0.0f0# retire(r, RAY_STATUS_INFINITY)
    end
    return 0.0f0
end

# the second group is functions that invoke a kernel of a group one function

function continuum_shade(;RGB3, RGB, tris, hit_idx, tmp, rays, n_tris, spectrum, expansion, first_diffuse, retina_factor, intensity=100, dλ, tex, kwargs...)
    RGB3 .= 0.0f0

    hit_idx .= Int32(1)
    next_hit!(hit_idx, tmp, rays, n_tris)
    tri_view = @view tris[hit_idx]

    s(args...) = shade_tex(args..., tex)
    #s(args...) = shade(args...)
    broadcast = @~ s.(rays, hit_idx, tri_view, first_diffuse, spectrum) .* retina_factor .* intensity .* dλ
    # broadcast rule not implemeted for sum!
    # WARNING this next line is ~90% of pure_sphere runtime at res=1024^2
    RGB3 .= sum(broadcast, dims=3) |> a -> reshape(a, length(rays), 3)

    map!(brightness -> clamp(brightness, 0, 1), RGB3, RGB3)
    @info maximum(RGB3)
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


function atomic_spectrum_kernel(rays ::AbstractArray{ADRay}, hits, tris, first_diffuse_index, spectrum, tex)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    adr = rays[idx]
    n = hits[idx]
    t = tris[idx]
    if n >= first_diffuse_index
        for (n_λ, λ) in enumerate(spectrum)
            r = expand(adr, λ)
            d, n, t = get_hit((n, t), r, unsafe=true)
            # NOTE this r breaks convention of dir being zero for retired rays
            r = FastRay(r.pos + r.dir * d, r.dir, r.ignore_tri)
            u, v = tex_uv(r.pos, t)

            if !isnan(u) && !isnan(v) && !isinf(u) && !isinf(v)
                # it's theoretically possible u, v could come back as zero
                w, h = size(tex)[1:2]
                i = clamp(Int(ceil(u * w)), 1, w)
                j = clamp(Int(ceil(v * h)), 1, h)

                intensity = cosine_shading(r, t)
                CUDA.@atomic tex[i, j, n_λ] += intensity
            end
        end
    end
    nothing
end


function spectral_light_map!(; tris, hit_idx, tmp, rays, n_tris, spectrum, first_diffuse, dλ, tex, kwargs...)
    hit_idx .= Int32(1)
    next_hit!(hit_idx, tmp, rays, n_tris)
    tri_view = @view tris[hit_idx]
    @info maximum(map(r->norm(r.dir′), rays))
    @assert length(rays) % 256 == 0
    @assert length(rays) == length(hit_idx) == length(tri_view)


    @cuda blocks=length(rays) ÷ 256 threads=256 atomic_spectrum_kernel(rays, hit_idx, tri_view, first_diffuse, spectrum, tex)
    synchronize()
end


function continuum_shade2(;RGB3, RGB, tris, hit_idx, tmp, rays, n_tris, spectrum, expansion, first_diffuse, retina_factor, intensity=1f-2, dλ, tex, kwargs...)
    RGB3 .= 0.0f0
    @info maximum(map(r->norm(r.dir′), rays))
    hit_idx .= Int32(1)
    next_hit!(hit_idx, tmp, rays, n_tris)
    tri_view = @view tris[hit_idx]
    for (n_λ, λ) in enumerate(Array(spectrum))
        tex_view = @view tex[:, :, n_λ]
        s(args...) = shade_tex(args..., tex_view)
        #s(args...) = shade(args...)
        @info size(retina_factor)
        @info size(rays)
        broadcast = @~ s.(rays, hit_idx, tri_view, first_diffuse, spectrum[:, :, n_λ]) .* retina_factor[:, :, n_λ] .* intensity .* dλ
        # broadcast rule not implemeted for sum!
        # WARNING this next line is ~90% of pure_sphere runtime at res=1024^2
        RGB3 .+= sum(broadcast, dims=3) |> a -> reshape(a, length(rays), 3)
    end

    map!(brightness -> clamp(brightness, 0, 1), RGB3, RGB3)
    @info maximum(RGB3)
    RGB .= RGBf.(RGB3[:, 1], RGB3[:, 2], RGB3[:, 3])
end


function expansion_shade2(;RGB3, RGB, tris, hit_idx, tmp, rays, n_tris, spectrum, expansion, first_diffuse, retina_factor, intensity=1f-2, dλ, tex, kwargs...)
    RGB3 .= 0.0f0

    @info maximum(map(r->norm(r.dir′), rays))

    for (n_λ, λ) in enumerate(Array(spectrum))
        expansion .= expand.(rays, λ)
        hit_idx .= Int32(1)
        next_hit!(hit_idx, tmp, expansion, n_tris)
        tri_view = @view tris[hit_idx]

        tex_view = @view tex[:, :, n_λ]
        s(args...) = shade_tex(args..., tex_view)
        #s(args...) = shade(args...)
        @info size(retina_factor)
        @info size(rays)
        broadcast = @~ s.(rays, hit_idx, tri_view, first_diffuse, spectrum[:, :, n_λ]) .* retina_factor[:, :, n_λ] .* intensity .* dλ
        # broadcast rule not implemeted for sum!
        # WARNING this next line is ~90% of pure_sphere runtime at res=1024^2
        RGB3 .+= sum(broadcast, dims=3) |> a -> reshape(a, length(rays), 3)
    end

    map!(brightness -> clamp(brightness, 0, 1), RGB3, RGB3)
    @info maximum(RGB3)
    RGB .= RGBf.(RGB3[:, 1], RGB3[:, 2], RGB3[:, 3])
end
