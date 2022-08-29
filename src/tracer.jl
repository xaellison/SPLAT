include("material.jl")
include("ray_generators.jl")
include("ray_imagers.jl")
include("rgb_spectrum.jl")

using ForwardDiff
using Makie
using Serialization
using Tullio

import Random.rand!

function get_hit(n_s::Tuple{Int32,Sphere}, r::AbstractRay; kwargs...)
    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    n, s = n_s
    if s.radius <= 0
        return (Inf32, one(Int32), s)
    end
    d = distance_to_sphere(r.pos, r.dir, s)
    if isinf(d)
        return (Inf32, one(Int32), s)
    end
    return (d, n, s)
end

function get_hit(n_s::Tuple{Int32,Sphere}, r::ADRay; kwargs...)
    n, s = n_s
    if s.radius <= 0
        return ((Inf32, Inf32), one(Int32), s)
    end
    d0 = distance_to_sphere(r.pos, r.dir, s)
    d(λ) = distance_to_sphere(r.pos + (λ - r.λ) * r.pos′, r.dir + (λ - r.λ) * r.dir′, s)
    if isinf(d0)
        return ((Inf32, Inf32), one(Int32), s)
    end
    return ((d0, ForwardDiff.derivative(d, r.λ)), n, s)
end

function get_hit(
    n_t::Tuple{Int32,T},
    r::AbstractRay;
    unsafe = false,
)::Tuple{Float32,Int32,T} where {T}
    # unsafe = true will ignore the in triangle test: useful for continuum_shade
    n, t = n_t
    d = distance_to_plane(r.pos, r.dir, t[2], t[1])
    p = r.pos + r.dir * d
    if (unsafe || in_triangle(p, t[2], t[3], t[4])) && d > 0 && r.ignore_tri != n
        return (d, n, t)
    else
        return (Inf32, one(Int32), t)
    end
end

function get_hit(
    n_t::Tuple{Int32,T},
    r::ADRay;
    kwargs...,
)::Tuple{Tuple{Float32,Float32},Int32,T} where {T}
    n, t = n_t
    # for n = 1, the degenerate triangle, this will be NaN, which fails d0 > 0 below
    d0 = distance_to_plane(r.pos, r.dir, t[2], t[1])
    d(λ) = distance_to_plane(
        r.pos + r.pos′ * (λ - r.λ),
        r.dir + r.dir′ * (λ - r.λ),
        t[2],
        t[1],
    )
    p = r.pos + r.dir * d0
    if in_triangle(p, t[2], t[3], t[4]) && d0 > 0 && r.ignore_tri != n
        return ((d0, ForwardDiff.derivative(d, r.λ)), n, t)
    else
        return ((Inf32, Inf32), one(Int32), t)
    end
end

## Hit computers for AD  Rays
#"""

function typemax(::Type{Tuple{Tuple{Float32,Float32},Int32}})
    return ((Inf32, Inf32), one(Int32))
end

function typemax(::Type{Tuple{Float32,Int32}})
    return (Inf32, one(Int32))
end

function hit_argmin(n_t, r::ADRay)::Tuple{Float32,Int32}
    return hit_argmin(n_t, FastRay(r))
end

function hit_argmin(n_t, r::FastRay)::Tuple{Float32,Int32}
    return get_hit(n_t, r)[1:2]
end

function next_hit!(dest, tmp, rays, n_tris)
    # TODO: restore tmp as function arg?
    @tullio (min) tmp[i] = hit_argmin(n_tris[j], rays[i])
    d_view = @view dest[:]
    d_view = reshape(d_view, length(d_view))
    map!(x -> x[2], d_view, tmp)
    return
end

#"""
## Ray evolvers


function p(r, d, d′, λ::N) where {N}
    r.pos + # origin constant
    r.pos′ * (λ - r.λ) +  #origin linear
    (r.dir + # direction constant
     r.dir′ * (λ - r.λ)) * # ... plus direction linear
    (d + d′ * (λ - r.λ)) # times constant + linear distance
end

function handle_optics(r, d, d′, n, N, n1::N1, n2::N2, rndm) where {N1,N2}
    refracts =
        can_refract(r.dir, N(r.λ), n1(r.λ), n2(r.λ)) &&
        rndm > reflectance(r.dir, N(r.λ), n1(r.λ), n2(r.λ))

    if refracts
        return ADRay(
            p(r, d, d′, r.λ),
            ForwardDiff.derivative(λ -> p(r, d, d′, λ), r.λ),
            refract(r.dir, N(r.λ), n1(r.λ), n2(r.λ)),
            ForwardDiff.derivative(
                λ -> refract(r.dir + r.dir′ * (λ - r.λ), N(r.λ), n1(λ), n2(λ)),
                r.λ,
            ),
            !r.in_medium,
            n,
            r.dest,
            r.λ,
            RAY_STATUS_ACTIVE,
        )

    else
        return ADRay(
            p(r, d, d′, r.λ),
            ForwardDiff.derivative(λ -> p(r, d, d′, λ), r.λ),
            reflect(r.dir, N(r.λ)),
            ForwardDiff.derivative(λ -> reflect(r.dir + r.dir′ * (λ - r.λ), N(λ)), r.λ),
            r.in_medium,
            n,
            r.dest,
            r.λ,
            RAY_STATUS_ACTIVE,
        )
    end
end

function evolve_ray(r::ADRay, n, t, rndm, first_diffuse_index)::ADRay
    if r.status != RAY_STATUS_ACTIVE
        return r
    end
    (d, d′), n, t = get_hit((n, t), r)
    if n >= first_diffuse_index
        # compute the position in the new triangle, set dir to zero
        return retire(r, RAY_STATUS_DIFFUSE)
    end
    if isinf(d)
        return retire(r, RAY_STATUS_INFINITY)
    end

    N(λ) = optical_normal(t, p(r, d, d′, λ))
    if r.in_medium
        return handle_optics(r, d, d′, n, N, glass, air, rndm)
    else
        return handle_optics(r, d, d′, n, N, air, glass, rndm)
    end
end

## Wrap it all up

function run_evolution!(;
    height::Int,
    width::Int,
    dλ,
    depth,
    sort_optimization,
    first_diffuse,
    n_tris,
    tris,
    rays,
    hit_idx,
    tmp,
    rndm,
    kwargs...,
) where {T}
    intensity = 1.0f0
    cutoff = length(rays)

    for iter = 1:depth
        h_view = @view hit_idx[1:cutoff]
        r_view = @view rays[1:cutoff]
        #@info "$(length(r_view)) / $(length(rays)) = $(length(r_view) / length(rays))"
        tmp_view = @view tmp[1:cutoff]
        next_hit!(h_view, tmp_view, r_view, n_tris)
        # evolve rays optically
        rand!(rndm)
        tri_view = @view tris[h_view]
        rand_view = @view rndm[1:cutoff]
        # everything has to be a view of the same size to avoid allocs + be sort safe
        r_view .= evolve_ray.(r_view, h_view, tri_view, rand_view, first_diffuse)

        # retire appropriate rays
        if sort_optimization
            sort!(r_view, by = ray -> ray.status)
            cutoff = count(ray -> ray.status == RAY_STATUS_ACTIVE, r_view)
            cutoff = min(length(rays), cutoff + 256 - cutoff % 256)
        end
    end

    if sort_optimization
        # restore original order so we can use simple broadcasts to color RGB
        # TODO: make abstraction so synchronize() works on GPU (if quicksort)
        sort!(rays, by = r -> r.dest)
    end
    return
end
