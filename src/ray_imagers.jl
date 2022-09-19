# Functions that take a set of evolved rays and convert them to image output
# (including light-map textures)

# The first group is shaders - given a ray, wavelength, hit: return intensity

function shade(r::FastRay, n, t, first_diffuse_index)::Float32
    # NB disregards in_medium
    # evolve to hit a diffuse surface
    d, n, t = get_hit((n, t), r)
    r = FastRay(r.pos + r.dir * d, zero(ℜ³), r.ignore_tri)

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
        return 0.0f0
    end
    return 0.0f0
end

function shade(adr::ADRay, n, t, first_diffuse_index, λ)::Float32
    # NB disregards in_medium
    # evolve to hit a diffuse surface
    r = expand(adr, λ)
    d, n, t = get_hit((n, t), r)
    r = FastRay(r.pos + r.dir * d, zero(ℜ³), r.ignore_tri)

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
        return 0.0f0
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
        return 0.0f0
    end
    return 0.0f0
end

function shade_tex(adr::ADRay, n, t, first_diffuse_index, λ, i_λ, tex :: AbstractArray{Float32})::Float32
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
        return tex[i, j, i_λ] * cosine_shading(r, t)
    end
    if isinf(d)
        return 0.0f0
    end
    return 0.0f0
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


function atomic_spectrum_kernel(rays ::AbstractArray{FastRay}, hits, tris, first_diffuse_index, spectrum, tex, n_λ)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    r = rays[idx]
    n = hits[idx]
    t = tris[idx]
    if n >= first_diffuse_index
        d, n, t = get_hit((n, t), r)
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
    nothing
end

# the second group is functions that invoke a kernel of a group one function

function continuum_light_map!(;tracer, tris, rays, n_tris, spectrum, first_diffuse, dλ, tex, kwargs...)
    tri_view = @view tris[tracer.hit_idx]
    @assert length(rays) % 256 == 0

    @cuda blocks=length(rays) ÷ 256 threads=256 atomic_spectrum_kernel(rays, tracer.hit_idx, tri_view, first_diffuse, spectrum, tex)
    synchronize()
end

function expansion_light_map!(; tris, hit_idx, tmp, rays, expansion, n_tris, spectrum, first_diffuse, dλ, tex, kwargs...)
    for (n_λ, λ) in enumerate(Array(spectrum))
        expansion .= expand.(rays, λ)
        hit_idx .= Int32(1)
        next_hit!(hit_idx, tmp, expansion, n_tris)
        tri_view = @view tris[hit_idx]


        @assert length(rays) % 256 == 0
        @assert length(rays) == length(hit_idx) == length(tri_view)

        @cuda blocks=length(rays) ÷ 256 threads=256 atomic_spectrum_kernel(expansion, hit_idx, tri_view, first_diffuse, spectrum, tex, n_λ)
    end
    synchronize()
end


function continuum_shade!(;tracer, RGB3, RGB, tris, rays, n_tris, spectrum, first_diffuse, retina_factor, intensity=7f-2, dλ, tex, kwargs...)
    RGB3 .= 0.0f0
    tri_view = @view tris[tracer.hit_idx]
    # putting tex in a Ref prevents it from being broadcast over
    # TODO don't alloc on fly - change made as micro-opt to avoid allocs
    i_Λ = CuArray(1:length(spectrum)) |> a -> reshape(a, size(spectrum))
    s(args...) = shade_tex(args...,  tex)

    broadcast = @~ s.(rays, tracer.hit_idx, tri_view, first_diffuse, spectrum, i_Λ) .* retina_factor .* intensity .* dλ
    # broadcast rule not implemeted for sum!
    # WARNING this next line is ~90% of pure_sphere runtime at res=1024^2
    RGB3 .+= sum(broadcast, dims=3) |> a -> reshape(a, length(rays), 3)

    map!(brightness -> clamp(brightness, 0, 1), RGB3, RGB3)

    RGB .= RGBf.(RGB3[:, 1], RGB3[:, 2], RGB3[:, 3])
end


function expansion_shade!(;RGB3, RGB, tris, hit_idx, tmp, rays, n_tris, spectrum, expansion, first_diffuse, retina_factor, intensity=7f-2, dλ, tex, kwargs...)
    RGB3 .= 0.0f0
    for (n_λ, λ) in enumerate(Array(spectrum))
        expansion .= expand.(rays, λ)
        hit_idx .= Int32(1)
        next_hit!(hit_idx, tmp, expansion, n_tris)
        tri_view = @view tris[hit_idx]

        tex_view = @view tex[:, :, n_λ]
        s(args...) = shade_tex(args..., tex_view)
        broadcast = @~ s.(rays, hit_idx, tri_view, first_diffuse, spectrum[:, :, n_λ]) .* retina_factor[:, :, n_λ] .* intensity .* dλ
        # broadcast rule not implemeted for sum!
        RGB3 .+= sum(broadcast, dims=3) |> a -> reshape(a, length(rays), 3)
    end

    map!(brightness -> clamp(brightness, 0, 1), RGB3, RGB3)
    RGB .= RGBf.(RGB3[:, 1], RGB3[:, 2], RGB3[:, 3])
end
