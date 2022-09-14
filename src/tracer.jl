include("material.jl")
include("ray_generators.jl")
include("ray_imagers.jl")
include("rgb_spectrum.jl")
include("atomic_argmin.jl")

using ForwardDiff
using Makie
using Serialization
using Tullio

import Random.rand!

function p(r, t, t′, λ::N) where {N}
    r.pos + # origin constant
    r.pos′ * (λ - r.λ) +  #origin linear
    (r.dir + # direction constant
     r.dir′ * (λ - r.λ)) * # ... plus direction linear
    (t + t′ * (λ - r.λ)) # times constant + linear distance
end

function handle_optics(r, t, t′, i, N, n1::R1, n2::R2, rndm) where {R1,R2}
    refracts =
        can_refract(r.dir, N(r.λ), n1(r.λ), n2(r.λ)) &&
        rndm > reflectance(r.dir, N(r.λ), n1(r.λ), n2(r.λ))

    if refracts
        return ADRay(
            p(r, t, t′, r.λ),
            ForwardDiff.derivative(λ -> p(r, t, t′, λ), r.λ),
            refract(r.dir, N(r.λ), n1(r.λ), n2(r.λ)),
            ForwardDiff.derivative(
                λ -> refract(r.dir + r.dir′ * (λ - r.λ), N(r.λ), n1(λ), n2(λ)),
                r.λ,
            ),
            !r.in_medium,
            i,
            r.dest,
            r.λ,
            RAY_STATUS_ACTIVE,
        )

    else
        return ADRay(
            p(r, t, t′, r.λ),
            ForwardDiff.derivative(λ -> p(r, t, t′, λ), r.λ),
            reflect(r.dir, N(r.λ)),
            ForwardDiff.derivative(λ -> reflect(r.dir + r.dir′ * (λ - r.λ), N(λ)), r.λ),
            r.in_medium,
            i,
            r.dest,
            r.λ,
            RAY_STATUS_ACTIVE,
        )
    end
end

function evolve_ray(r::ADRay, i, T, rndm, first_diffuse_index)::ADRay
    if r.status != RAY_STATUS_ACTIVE
        return r
    end
    (t, t′), i, T = get_hit((i, T), r)
    if i >= first_diffuse_index
        # compute the position in the new triangle, set dir to zero
        return retire(r, RAY_STATUS_DIFFUSE)
    end
    if isinf(t)
        return retire(r, RAY_STATUS_INFINITY)
    end

    N(λ) = optical_normal(T, p(r, t, t′, λ))
    # TODO: replace glass/air with expansion terms
    if r.in_medium
        return handle_optics(r, t, t′, i, N, glass, air, rndm)
    else
        return handle_optics(r, t, t′, i, N, air, glass, rndm)
    end
end

## Wrap it all up

function run_evolution!(hitter;
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
        next_hit!(hitter, h_view, tmp_view, r_view, n_tris)
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
