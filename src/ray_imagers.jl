
## Datastruct manipulation

function final_evolution(r, i, t, ∂t∂λ, ∂t∂x, ∂t∂y)
    return ADRay(
        next_p(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, r.λ, r.x, r.y),
        ForwardDiff.derivative(λ -> next_p(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, λ, r.x, r.y), r.λ),
        ForwardDiff.derivative(x -> next_p(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, r.λ, x, r.y), r.x),
        ForwardDiff.derivative(y -> next_p(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, r.λ, r.x, y), r.y),
        r.dir,
        r.∂d∂λ,
        r.∂d∂x,
        r.∂d∂y,
        r.in_medium,
        i,
        r.dest,
        r.λ,
        r.x,
        r.y,
        RAY_STATUS_DIFFUSE,
    )
end

function final_evolution(r, i, T)
    """
    Evolves rays that are pointing at diffuse surfaces into an expansion of
    the hit position in that surface. The resulting ray has the same direction
    as went in, which is useful for shading. Further evolution is not well
    defined.
    """
    if r.status == RAY_STATUS_DIFFUSE
        (t, ∂t∂λ, ∂t∂x, ∂t∂y), i, T = get_hit((i, T), r)
        return final_evolution(r, i, t, ∂t∂λ, ∂t∂x, ∂t∂y)
    else
        return r
    end
end


## Light mapping

function atomic_spectrum_kernel(
    rays::AbstractArray{ADRay},
    tris,
    first_diffuse_index,
    spectrum,
    δx,
    δy,
    tex,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    adr = rays[idx]
    t = tris[idx]
    if adr.status == RAY_STATUS_DIFFUSE
        for (n_λ, λ) in enumerate(spectrum)
            for x in δx
                for y in δy
                    r = expand(adr, λ, adr.x + x, adr.y + y)
                    u, v = tex_uv(r, t)

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
    # TODO: remove alloc
    rays .= final_evolution.(rays, tracer.hit_idx, tri_view)
    @cuda blocks = length(rays) ÷ 256 threads = 256 atomic_spectrum_kernel(
        rays,
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
    i,
    T,
    first_diffuse_index,
    λ, δx, δy,
    i_λ,
    tex::AbstractArray{Float32},
)::Float32
    # NB disregards in_medium
    # evolve to hit a diffuse surface
    r = expand(adr, λ, δx, δy)
    t, i, T = get_hit((i, T), r, unsafe = true)
    # WARNING inconsistent ray defined
    r = FastRay(r.pos + r.dir * t, r.dir, r.ignore_tri)

    @inbounds if i >= first_diffuse_index
        # compute the position in the new triangle, set dir to zero
        u, v = tex_uv(r.pos, T)

        CUDA.@assert !isnan(u) && !isnan(v)
        # it's theoretically possible u, v could come back as zero
        i = clamp(Int(ceil(u * (size(tex)[1]))), 1, size(tex)[1])
        j = clamp(Int(ceil(v * (size(tex)[2]))), 1, size(tex)[2])
        return tex[i, j, i_λ] * cosine_shading(r, T)
    end
    if isinf(t)
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
    dλ,
    tex,
    width,
    height,
    intensity = 1.0f0,
    kwargs...,
)
    @info "stable imager"
    RGB3 .= 0.0f0
    tri_view = @view tris[tracer.hit_idx]
    # putting tex in a Ref prevents it from being broadcast over
    # TODO don't alloc on fly - change made as micro-opt to avoid allocs
    i_Λ = CuArray(1:length(spectrum)) |> a -> reshape(a, size(spectrum))
    s(args...) = shade_tex(args..., tex)
    δx = tracer.δ |> a -> reshape(a, (1, 1, 1, length(a)))
    δy = tracer.δ |> a -> reshape(a, (1, 1, 1, 1, length(a)))
    broadcast = @~ s.(rays, tracer.hit_idx, tri_view, first_diffuse, spectrum, δx, δy, i_Λ) .* retina_factor .* intensity
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
    T,
    first_diffuse_index,
    intensity,
    δx,
    δy,
    Λ,
    tex::AbstractArray{Float32},
    retina_factor
)
    # NB disregards in_medium
    # evolve to hit a diffuse surface
    out = zero(RGBf)
    if adr.status == RAY_STATUS_DIFFUSE
        for i_λ in 1:length(Λ)
        λ = Λ[i_λ]
        r = expand(adr, λ, adr.x + δx, adr.y + δy)

            # compute the position in the new triangle, set dir to zero
            u, v = tex_uv(r, T)
            if  !isnan(u) && !isnan(v)
                # it's theoretically possible u, v could come back as zero
                i = clamp(Int(ceil(u * (size(tex)[1]))), 1, size(tex)[1])
                j = clamp(Int(ceil(v * (size(tex)[2]))), 1, size(tex)[2])
                intensity_λ = intensity * tex[i, j, i_λ] * cosine_shading(r, T)
                R = retina_factor[1, 1, i_λ] * intensity_λ
                G = retina_factor[1, 2, i_λ] * intensity_λ
                B = retina_factor[1, 3, i_λ] * intensity_λ
                out += RGBf(R, G, B)
            end
        end
    end
    RGBf(clamp(out.r, 0, 1), clamp(out.g, 0, 1), clamp(out.b, 0, 1))
end


# bilinear tex filter
function shade_tex3(
    adr::ADRay,
    T,
    first_diffuse_index,
    intensity,
    δx,
    δy,
    Λ,
    tex::AbstractArray{Float32},
    retina_factor
)
    # NB disregards in_medium
    # evolve to hit a diffuse surface
    out = zero(RGBf)
    if adr.status == RAY_STATUS_INFINITY
        x = (normalize(adr.dir)[2] * 0.5f0 + 0.5f0) * 0.2f0
        if isnan(x)
            x = 0.0f0
        end
        return RGBf(x,x,x)
    end
    if adr.status == RAY_STATUS_DIFFUSE
        for i_λ in 1:length(Λ)
        λ = Λ[i_λ]
        r = expand(adr, λ, adr.x + δx, adr.y + δy)

            # compute the position in the new triangle, set dir to zero
            _u, _v = tex_uv(r, T)
            if  !isnan(_u) && !isnan(_v)
                # it's theoretically possible u, v could come back as zero
                u, v = _u * size(tex)[1], _v * size(tex)[2]
                i₊ = clamp(Int(ceil(u)), 1, size(tex)[1])
                j₊ = clamp(Int(ceil(v)), 1, size(tex)[2])
                i₋ = clamp(Int(floor(u)), 1, size(tex)[1])
                j₋ = clamp(Int(floor(v)), 1, size(tex)[2])
                I₊₊ = intensity * tex[i₊, j₊, i_λ] * cosine_shading(r, T)
                I₊₋ = intensity * tex[i₊, j₋, i_λ] * cosine_shading(r, T)
                I₋₊ = intensity * tex[i₋, j₊, i_λ] * cosine_shading(r, T)
                I₋₋ = intensity * tex[i₋, j₋, i_λ] * cosine_shading(r, T)
                s₊₊ = abs((u - i₋) * (v - j₋))
                s₊₋ = abs((u - i₋) * (v - j₊))
                s₋₊ = abs((u - i₊) * (v - j₋))
                s₋₋ = abs((u - i₊) * (v - j₊))

                R = retina_factor[1, 1, i_λ] * (s₊₊ * I₊₊ + s₋₊ * I₋₊ + s₊₋ * I₊₋ + s₋₋ * I₋₋)
                G = retina_factor[1, 2, i_λ] * (s₊₊ * I₊₊ + s₋₊ * I₋₊ + s₊₋ * I₊₋ + s₋₋ * I₋₋)
                B = retina_factor[1, 3, i_λ] * (s₊₊ * I₊₊ + s₋₊ * I₋₊ + s₊₋ * I₊₋ + s₋₋ * I₋₋)
                out += RGBf(R, G, B)
            end
        end
    end
    RGBf(clamp(out.r, 0, 1), clamp(out.g, 0, 1), clamp(out.b, 0, 1))
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
    dλ,
    tex,
    width,
    height,
    intensity=1.0f0,
    kwargs...,
)
    """
    Uses shade_tex2 and final_evolution to refactor a final hit calculation
    by a factor of (δx * δy) compared to StableImager
    """
    RGB3 .= 0.0f0
    tri_view = @view tris[tracer.hit_idx]
    s(args...) = shade_tex3(args..., spectrum, tex, retina_factor)
    δx = tracer.δ |> a -> reshape(a, (length(a)))
    δy = tracer.δ |> a -> reshape(a, (1, length(a)))
    rays .= final_evolution.(rays, tracer.hit_idx, tri_view)
    R = length(rays)
    # reshape so expansions are first dims, and put in a common block for better IO patterns
    upres_rgb = s.(reshape(rays, (1, 1, R)), reshape(tri_view, (1,1,R)), first_diffuse, intensity, δx, δy)
    upres_rgb = permutedims(upres_rgb, (3, 1, 2))
    # next 3 lines expand into final image
    upres_rgb = reshape(upres_rgb, width ÷ length(δx), height ÷ length(δy), length(δx), length(δy))
    upres_rgb = permutedims(upres_rgb, (3, 1, 4, 2))
    upres_rgb = reshape(upres_rgb, size(RGB))
    RGB .= upres_rgb
end
