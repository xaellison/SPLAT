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
