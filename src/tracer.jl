include("material.jl")
include("rgb_spectrum.jl")

using ForwardDiff
using Makie
using ProgressMeter
using Serialization

function get_hit(n_t::Tuple{Int32,T}, r::AbstractRay)::Tuple{Float32,Int32,T} where {T}
    n, t = n_t
    d = distance_to_plane(r.pos, r.dir, t[2], t[1])
    p = r.pos + r.dir * d
    if in_triangle(p, t[2], t[3], t[4]) && d > 0 && r.ignore_tri != n
        return (d, n, t)
    else
        return (Inf32, typemax(Int32), t)
    end
end

function get_hit(n_t::Tuple{Int32,T}, r::ADRay)::Tuple{Tuple{Float32,Float32},Int32,T} where {T}
    n, t = n_t
    # for n = 1, the degenerate triangle, this will be NaN, which fails d0 > 0 below
    d0 = distance_to_plane(r.pos, r.dir, t[2], t[1])
    d(λ) = distance_to_plane(r.pos + r.pos′ * (λ - r.λ), r.dir + r.dir′ * (λ - r.λ), t[2], t[1])
    p = r.pos + r.dir * d0
    if in_triangle(p, t[2], t[3], t[4]) && d0 > 0 && r.ignore_tri != n
        return ((d0, ForwardDiff.derivative(d, r.λ)), n, t)
    else
        return ((Inf32, Inf32), n, t)
    end
end

## Hit computers for AD  Rays


function next_hit!(dest :: AbstractArray{I}, rays, n_tris:: AbstractArray{Tuple{I, T}}, override=false) where {I, T}
    for i in 1:length(dest)
        if !rays[i].retired || override
            dest[i] = minimum(n_tri -> get_hit(n_tri, rays[i]), n_tris, init=((Inf32, Inf32), typemax(Int32), zero(T)))[2]
            if dest[i] == typemax(I)
                dest[i] = one(I)
            end
        else
            dest[i] = one(Int32)
        end
    end
    return nothing
end

## Ray evolvers


function p(r, d, d′, λ::N) where N
    r.pos + # origin constant
    r.pos′ * (λ - r.λ) +  #origin linear
    (r.dir + # direction constant
    r.dir′ * (λ - r.λ)) * # ... plus direction linear
    (d + d′ * (λ - r.λ)) # times constant + linear distance
end

function handle_optics(r, d, d′, n, N, n1 :: N1, n2::N2, rndm) where {N1, N2}
    refracts = can_refract(r.dir, N(r.λ), n1(r.λ), n2(r.λ)) && rndm > reflectance(r.dir, N(r.λ), n1(r.λ), n2(r.λ))

    if refracts
        return ADRay(p(r, d, d′, r.λ),
                     ForwardDiff.derivative(λ->p(r, d, d′, λ), r.λ),
                     refract(r.dir, N(r.λ), n1(r.λ), n2(r.λ)),
                     ForwardDiff.derivative(λ -> refract(r.dir + r.dir′ * (λ - r.λ), N(r.λ), n1(λ), n2(λ)), r.λ),
                     !r.in_medium, n, r.dest, r.λ, false)

    else
        return ADRay(p(r, d, d′, r.λ),
                     ForwardDiff.derivative(λ->p(r, d, d′, λ), r.λ),
                     reflect(r.dir, N(r.λ)),
                     ForwardDiff.derivative(λ -> reflect(r.dir + r.dir′ * (λ - r.λ), N(λ)), r.λ),
                     r.in_medium, n, r.dest, r.λ, false)
    end
end

function evolve_ray(r::ADRay, n, t, rndm, first_diffuse_index)::ADRay

    (d, d′), n, t = get_hit((n, t), r)
    if n >= first_diffuse_index
        # compute the position in the new triangle, set dir to zero
        return  ADRay(p(r, d, d′, r.λ),
                     ForwardDiff.derivative(λ->p(r, d, d′, λ), r.λ),
                     zero(V3),
                     zero(V3),
                     r.in_medium, n, r.dest, r.λ, true)
    end
    if isinf(d)
        return retired(r)
    end

    N(λ) = optical_normal(t, p(r, d, d′, λ))
    if r.in_medium
        return handle_optics(r, d, d′, n, N, glass, air, rndm)
    else
        return handle_optics(r, d, d′, n, N, air, glass, rndm)
    end
end


## Wrap it all up

function ad_frame_matrix(
    camera_generator::Function,
    width::Int,
    height::Int,
    hit_tris::AbstractArray{Tri},
    tris::AbstractArray{T},
    skys,
    dλ,
    depth,
    ITERS,
    phi,
    random,
    A, # type: either Array or CuArray
    sort_optimization,
    title,
    first_diffuse,
) where T
    camera = camera_generator(1, 1)

    #out .= RGBf(0, 0, 0)
    λ_min = 400.0f0
    λ_max = 700.0f0
    intensity = Float32(1 / ITERS) * 2

    function init_ray(x, y, λ, dv)::AbstractRay
        _x, _y = x - width / 2, y - height / 2
        scale = width * _COS_45 / camera.FOV_half_sin
        _x /= scale
        _y /= scale

        _z = sqrt(1 - _x^2 - _y^2)
        dir =
            _x * camera.right +
            _y * camera.up +
            _z * camera.dir +
            dv * 0.25f0 / max(width, height)
        dir = normalize(dir)
        idx = (x - 1) * height + y
        polarization = normalize(cross(camera.up, dir))
        return ADRay(camera.pos, zero(V3), dir, zero(V3), false, 1, idx, λ, false)
    end

    n_tris = collect(zip(map(Int32, collect(1:length(tris))), hit_tris)) |> A |> m -> reshape(m, 1, length(m)) |> A
    tris = A(tris)
    hit_tris = A(hit_tris)
    row_indices = A(1:width)
    col_indices = reshape(A(1:height), 1, height)
    rays = A{ADRay}(undef, height * width)
    hit_idx = A(zeros(Int32, length(rays)))
    dv = A{V3}(undef, width) # make w*h
    s0 = A{Float32}(undef, length(rays), 3)


    # Datastruct init
    hits = A{Tuple{Tuple{Float32, Float32}, Int32, T}}(undef, (height* width))
    rndm = random(Float32, width * height)

    # use host to compute constants used in turning spectra into colors
    spectrum = collect(λ_min:dλ:λ_max) |> a -> reshape(a, 1, 1, length(a))
    retina_factor = Array{Float32}(undef, 1, 3, length(spectrum))
    map!(retina_red,begin @view retina_factor[1, 1, :] end, spectrum)
    map!(retina_green,begin @view retina_factor[1, 2, :] end, spectrum)
    map!(retina_blue,begin @view retina_factor[1, 3, :] end, spectrum)

    retina_factor=A(retina_factor)
    spectrum = A(spectrum)


    synchronize()
    @info "tracing depth = $depth"
    @time for iter = 1:ITERS

        dv .=
            V3.(
                random(Float32, width),
                random(Float32, width),
                random(Float32, width),
            )
        rays .= reshape(init_ray.(row_indices, col_indices, 550.0, dv), width * height)
        cutoff = length(rays)

        for iter = 1:depth
            # compute hits
            @info cutoff
            h_view = @view hit_idx[1:cutoff]
            r_view = @view rays[1:cutoff]
            @info "hits..."
            next_hit!(h_view, r_view, n_tris, false)

            # evolve rays optically
            rndm .= random(Float32, length(rays))
            tri_view = @view tris[hit_idx]
            # I need to pass a scalar arg - this closure seems necessary since map! freaks at scalar args
            evolve_closure(rays, hit_idx, tri_view, rndm) = evolve_ray(rays, hit_idx, tri_view, rndm, first_diffuse)
            map!(evolve_closure, rays, rays, hit_idx, tri_view, rndm)

            # retire appropriate rays
            if sort_optimization
                @info "retirement sort..."
                sort!(r_view, by=ray->ray.retired)
                cutoff = count(ray->!ray.retired, r_view)
                cutoff = min(length(rays), cutoff + 256 - cutoff % 256)
            end
        end
    end

    if sort_optimization
        # restore original order so we can use simple broadcasts to color RGB
        sort!(rays, by=r->r.dest)
    end

    frame_n = 18
    @info "images"
    # output array
    RGB = A{Float32}(undef, length(rays), 3)


    @time for frame_i in 1:frame_n
        RGB .= 0.0f0
        for (i, sky) in enumerate(skys)
            # WARNING deleted `r.in_medium ? 0.0f0 : `
            function shade(r, λ, rf, tri)
                if r.ignore_tri >= first_diffuse && r.ignore_tri != 1
                    u, v = reverse_uv(r.pos, tri)
                    if xor(u % 0.1f0 > 0.05f0, v % 0.1f0 > 0.05f0)
                        return 1.0f0
                    else
                        return 0.0f0
                    end
                else
                    return sky(r.dir + r.dir′ * (λ - r.λ), λ, Float32(2 * pi / 20 * frame_i / frame_n)) * rf * intensity * dλ
                end
            end
            broadcast = @~ shade.(rays, spectrum, retina_factor, tris[map(r->r.ignore_tri, rays)])
            RGB = sum(broadcast, dims=3) |> a -> reshape(a, length(rays), 3)
            map!(brightness -> clamp(brightness, 0, 1), RGB, RGB)
        end

        out = Dict()
        for s in keys(skys)

            img = RGBf.(map(a -> reshape(Array(a), height, width), (RGB[:, 1], RGB[:, 2], RGB[:, 3]))...)
            Makie.save("out/$title/$(lpad(frame_i, 3, "0")).png", img)

        end
    end
    return nothing
end
