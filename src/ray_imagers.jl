
## Light mapping

function atomic_spectrum_kernel(
    rays::AbstractArray{ADRay},
    hits,
    tris,
    first_diffuse_index,
    spectrum,
    δx,
    δy,
    tex,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    adr = rays[idx]
    n = hits[idx]
    t = tris[idx]
    if n >= first_diffuse_index
        for (n_λ, λ) in enumerate(spectrum)
            # TODO generalize wrt Tracer
            for x in δx
                for y in δy
                    r = expand(adr, λ, adr.x + x, adr.y + y)
                    d, n, t = get_hit((n, t), r, unsafe = true)
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
        end
    end
    nothing
end


function continuum_light_map!(;
    tracer,
    tris,
    rays,
    n_tris,
    spectrum,
    first_diffuse,
    dλ,
    tex,
    kwargs...,
)
    tri_view = @view tris[tracer.hit_idx]
    @assert length(rays) % 256 == 0

    @cuda blocks = length(rays) ÷ 256 threads = 256 atomic_spectrum_kernel(
        rays,
        tracer.hit_idx,
        tri_view,
        first_diffuse,
        spectrum,
        tracer.δ,
        tracer.δ,
        tex,
    )
end

## Final image rendering

function shade_tex(
    adr::ADRay,
    n,
    t,
    first_diffuse_index,
    λ, δx, δy,
    i_λ,
    tex::AbstractArray{Float32},
)::Float32
    # NB disregards in_medium
    # evolve to hit a diffuse surface
    r = expand(adr, λ, δx, δy)
    d, n, t = get_hit((n, t), r, unsafe = true)
    # WARNING inconsistent ray defined
    r = FastRay(r.pos + r.dir * d, r.dir, r.ignore_tri)

    @inbounds if n >= first_diffuse_index
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


function continuum_shade!(imager::StableImager;
    tracer,
    RGB3,
    RGB,
    tris,
    rays,
    n_tris,
    spectrum,
    first_diffuse,
    retina_factor,
    intensity = 7.0f-2,
    dλ,
    tex,
    width,
    height,
    kwargs...,
)
    RGB3 .= 0.0f0
    tri_view = @view tris[tracer.hit_idx]
    # putting tex in a Ref prevents it from being broadcast over
    # TODO don't alloc on fly - change made as micro-opt to avoid allocs
    i_Λ = CuArray(1:length(spectrum)) |> a -> reshape(a, size(spectrum))
    s(args...) = shade_tex(args..., tex)
    δx = tracer.δ |> a -> reshape(a, (1, 1, 1, length(a)))
    δy = tracer.δ |> a -> reshape(a, (1, 1, 1, 1, length(a)))
    broadcast = @~ s.(rays, tracer.hit_idx, tri_view, first_diffuse, spectrum, δx, δy, i_Λ) .*
       retina_factor .* intensity .* dλ
    # broadcast rule not implemeted for sum!
    # WARNING this next line is ~90% of pure_sphere runtime at res=1024^2
    summation = sum(broadcast, dims = 3)
    summation = summation |> a -> reshape(a, (width ÷ length(δx), height ÷ length(δy), 3, length(δx), length(δy)))
    summation = summation |> a -> permutedims(a, (4, 1, 5, 2, 3))
    summation = summation |> a -> reshape(a, length(RGB3) ÷ 3, 3)
    RGB3 .= summation

    map!(brightness -> clamp(brightness, 0, 1), RGB3, RGB3)

    RGB .= RGBf.(RGB3[:, 1], RGB3[:, 2], RGB3[:, 3])
end


## Experimental

function shade_tex2(
    adr::ADRay,
    n,
    t,
    first_diffuse_index,
    δx,
    δy,
    Λ,
    tex::AbstractArray{Float32},
    retina_factor
)
    # NB disregards in_medium
    # evolve to hit a diffuse surface
    out = zero(RGBf)
    for i_λ in 1:length(Λ)
        λ = Λ[i_λ]
        r = expand(adr, λ, δx, δy)
        d, n, t = get_hit((n, t), r, unsafe = true)
        # WARNING inconsistent ray defined
        r = FastRay(r.pos + r.dir * d, r.dir, r.ignore_tri)

        @inbounds if n >= first_diffuse_index
            # compute the position in the new triangle, set dir to zero
            u, v = tex_uv(r.pos, t)

            CUDA.@assert !isnan(u) && !isnan(v)
            # it's theoretically possible u, v could come back as zero
            i = clamp(Int(ceil(u * (size(tex)[1]))), 1, size(tex)[1])
            j = clamp(Int(ceil(v * (size(tex)[2]))), 1, size(tex)[2])
            intensity = tex[i, j, i_λ] * cosine_shading(r, t)
            R = retina_factor[1, 1, i_λ] * intensity
            G = retina_factor[1, 2, i_λ] * intensity
            B = retina_factor[1, 3, i_λ] * intensity
            out += RGBf(R, G, B)
        end
    end
    out
end


function continuum_shade!(imager::ExperimentalImager;
    tracer,
    RGB3,
    RGB,
    tris,
    rays,
    n_tris,
    spectrum,
    first_diffuse,
    retina_factor,
    intensity = 7.0f-2,
    dλ,
    tex,
    width,
    height,
    kwargs...,
)
    RGB3 .= 0.0f0
    tri_view = @view tris[tracer.hit_idx]
    # putting tex in a Ref prevents it from being broadcast over
    # TODO don't alloc on fly - change made as micro-opt to avoid allocs

    s(args...) = shade_tex2(args..., spectrum, tex, retina_factor)
    δx = tracer.δ |> a -> reshape(a, (1, length(a)))
    δy = tracer.δ |> a -> reshape(a, (1, 1, length(a)))
    # TODO: remove alloc
    upres_rgb = s.(rays, tracer.hit_idx, tri_view, first_diffuse, δx, δy)
    upres_rgb = reshape(upres_rgb, width ÷ length(δx), height ÷ length(δy), length(δx), length(δy))
    upres_rgb = permutedims(upres_rgb, (3, 1, 4, 2))
    upres_rgb = reshape(upres_rgb, size(RGB))
    RGB .= upres_rgb

end
