include("material.jl")
include("ray_generators.jl")
include("ray_imagers.jl")
include("rgb_spectrum.jl")
include("atomic_argmin.jl")
include("utils.jl")
include("hitters.jl")

using ForwardDiff
using Makie
using Serialization

import Random.rand!


## Ray evolvers


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

function run_evolution!(;
    hitter::AbstractHitter,
    tracer::Tracer,
    height::Int,
    width::Int,
    dλ,
    depth,
    first_diffuse,
    n_tris,
    tris,
    rays,
    kwargs...,
) where {T}
    for iter = 1:depth
        next_hit!(tracer, hitter, rays, n_tris)
        # evolve rays optically
        rand!(tracer.rndm)
        tri_view = @view tris[tracer.hit_idx]
        # everything has to be a view of the same size to avoid allocs + be sort safe
        rays .= evolve_ray.(rays, tracer.hit_idx, tri_view, tracer.rndm, first_diffuse)
    end
    return
end

function trace!(
    hitter_type::Type{H};
    cam,
    lights,
    tex,
    tris,
    λ_min,
    dλ,
    λ_max,
    width,
    height,
    depth,
    first_diffuse,
) where {H<:AbstractHitter}

    # initialize rays for forward tracing
    rays = rays_from_lights(lights)
    hitter = H(CuArray, rays)
    tracer = Tracer(CuArray, rays)

    spectrum, retina_factor = _spectrum_datastructs(CuArray, λ_min:dλ:λ_max)

    basic_params = Dict{Symbol,Any}()
    @pack! basic_params = width, height, dλ, λ_min, λ_max, depth, first_diffuse
    n_tris = tuple.(Int32(1):Int32(length(tris)), tris) |> m -> reshape(m, 1, length(m))

    Λ = CuArray(collect(λ_min:dλ:λ_max))
    array_kwargs = Dict{Symbol,Any}()
    @pack! array_kwargs = tex, tris, n_tris, rays, spectrum, retina_factor

    array_kwargs = Dict(kv[1] => CuArray(kv[2]) for kv in array_kwargs)
    CUDA.@time run_evolution!(;
        hitter = hitter,
        tracer = tracer,
        basic_params...,
        array_kwargs...,
    )
    CUDA.@time continuum_light_map!(; tracer = tracer, basic_params..., array_kwargs...)

    # reverse trace image
    RGB3 = CuArray{Float32}(undef, width * height, 3)
    RGB3 .= 0
    RGB = CuArray{RGBf}(undef, width * height)

    ray_generator(x, y, λ, dv) = camera_ray(cam, height, width, x, y, λ, dv)
    rays = wrap_ray_gen(ray_generator, height, width)

    hitter = H(CuArray, rays)
    tracer = Tracer(CuArray, rays)
    array_kwargs = Dict{Symbol,Any}()

    @pack! array_kwargs = tex, tris, n_tris, rays, spectrum, retina_factor, RGB3, RGB
    array_kwargs = Dict(kv[1] => CuArray(kv[2]) for kv in array_kwargs)

    CUDA.@time run_evolution!(;
        hitter = hitter,
        tracer = tracer,
        basic_params...,
        array_kwargs...,
    )
    CUDA.@time continuum_shade!(; tracer = tracer, basic_params..., array_kwargs...)
    return array_kwargs
end
