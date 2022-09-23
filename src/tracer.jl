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

## get_hit methods - how to intersect rays with geometric elements

function get_hit(i_S::Tuple{Int32,Sphere}, r::AbstractRay; kwargs...)
    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    i, S = i_S
    if S.radius <= 0
        return (Inf32, one(Int32), S)
    end
    t = distance_to_sphere(r.pos, r.dir, S)
    if isinf(t)
        return (Inf32, one(Int32), S)
    end
    return (t, i, S)
end

function get_hit(i_S::Tuple{Int32,Sphere}, r::ADRay; kwargs...)
    i, S = i_S
    if S.radius <= 0
        return ((Inf32, Inf32), one(Int32), S)
    end
    t0 = distance_to_sphere(r.pos, r.dir, S)
    t(λ) = distance_to_sphere(r.pos + (λ - r.λ) * r.∂p∂λ, r.dir + (λ - r.λ) * r.∂d∂λ, S)
    if isinf(t0)
        return ((Inf32, Inf32), one(Int32), S)
    end
    return ((t0, ForwardDiff.derivative(t, r.λ)), i, S)
end

function get_hit(
    i_T::Tuple{Int32,AbstractTri},
    r::AbstractRay;
    unsafe = false,
)::Tuple{Float32,Int32,AbstractTri} where {AbstractTri}
    # unsafe = true will ignore the in triangle test: useful for continuum_shade
    i, T = i_T
    t = distance_to_plane(r, T)
    p = r.pos + r.dir * t
    if (unsafe || in_triangle(p, T)) && t > 0 && r.ignore_tri != i
        return (t, i, T) # formerly d, n, t
    else
        return (Inf32, one(Int32), T)
    end
end

function get_hit(
    i_T::Tuple{Int32,AbstractTri},
    r::ADRay;
    kwargs...,
)::Tuple{Tuple{Float32,Float32,Float32,Float32},Int32,AbstractTri} where {AbstractTri}
    i, T = i_T
    # In the case of i = 1, the degenerate triangle, this will be NaN.
    # t0 = NaN fails d0 > 0 below, which properly gives us i = 1 back
    t0 = distance_to_plane(r, T)
    p0 = r.pos + r.dir * t0
    if in_triangle(p0, T) && t0 > 0 && r.ignore_tri != i
        return ((t0,
                ForwardDiff.derivative(λ -> distance_to_plane(r, T, λ, r.x, r.y), r.λ),
                ForwardDiff.derivative(x -> distance_to_plane(r, T, r.λ, x, r.y), r.x),
                ForwardDiff.derivative(y -> distance_to_plane(r, T, r.λ, r.x, y), r.y),),
                i,
                T)
    else
        return ((Inf32, Inf32, Inf32, Inf32), one(Int32), T)
    end
end


## Ray evolvers



function next_p(r :: ADRay, t :: Float32, ∂t∂λ :: Float32, ∂t∂x :: Float32, ∂t∂y :: Float32, λ::N1, x::N2, y::N3) where {N1, N2, N3}
    p_expansion(r, λ, x, y) +
    d_expansion(r, λ, x, y) *
    (t +
    ∂t∂λ * (λ - r.λ) +
    ∂t∂x * (x - r.x) +
    ∂t∂y * (y - r.y)) # times constant + linear distance
end

function handle_optics(r, t, ∂t∂λ, i, N, n1::R1, n2::R2, rndm) where {R1,R2}
    refracts =
        can_refract(r.dir, N(r.λ, r.x, r.y), n1(r.λ), n2(r.λ)) &&
        rndm > reflectance(r.dir, N(r.λ, r.x, r.y), n1(r.λ), n2(r.λ))

    if refracts
        return ADRay(
            next_p(r, t, ∂t∂λ, 0.0f0, 0.0f0, r.λ, 0.0f0, 0.0f0),
            ForwardDiff.derivative(λ -> next_p(r, t, ∂t∂λ, 0.0f0, 0.0f0, λ, 0.0f0, 0.0f0), r.λ),
            zero(ℜ³),
            zero(ℜ³),
            refract(r.dir, N(r.λ, r.x, r.y), n1(r.λ), n2(r.λ)),
            ForwardDiff.derivative(
                λ -> refract(r.dir + r.∂d∂λ * (λ - r.λ), N(λ, r.x, r.y), n1(λ), n2(λ)),
                r.λ,
            ),
            zero(ℜ³),
            zero(ℜ³),
            !r.in_medium,
            i,
            r.dest,
            r.λ,
            0.0f0,
            0.0f0,
            RAY_STATUS_ACTIVE,
        )

    else
        return ADRay(
            next_p(r, t, ∂t∂λ, 0.0f0, 0.0f0, r.λ, 0.0f0, 0.0f0),
            ForwardDiff.derivative(λ -> next_p(r, t, ∂t∂λ, 0.0f0, 0.0f0, λ, 0.0f0, 0.0f0), r.λ),
            zero(ℜ³),
            zero(ℜ³),
            reflect(r.dir, N(r.λ, r.x, r.y)),
            ForwardDiff.derivative(λ -> reflect(r.dir + r.∂d∂λ * (λ - r.λ), N(λ, r.x, r.y)), r.λ),
            zero(ℜ³),
            zero(ℜ³),
            r.in_medium,
            i,
            r.dest,
            r.λ,
            0.0f0,
            0.0f0,
            RAY_STATUS_ACTIVE,
        )
    end
end

function evolve_ray(r::ADRay, i, T, rndm, first_diffuse_index)::ADRay
    if r.status != RAY_STATUS_ACTIVE
        return r
    end
    (t, ∂t∂λ, ∂t∂x, ∂t∂y), i, T = get_hit((i, T), r)
    if i >= first_diffuse_index
        # compute the position in the new triangle, set dir to zero
        return retire(r, RAY_STATUS_DIFFUSE)
    end
    if isinf(t)
        return retire(r, RAY_STATUS_INFINITY)
    end

    N(λ, x, y) = optical_normal(T, next_p(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, λ, x, y))
    # TODO: replace glass/air with expansion terms
    if r.in_medium
        return handle_optics(r, t, ∂t∂λ, i, N, glass, air, rndm)
    else
        return handle_optics(r, t, ∂t∂λ, i, N, air, glass, rndm)
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
