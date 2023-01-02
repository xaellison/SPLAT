#include("material.jl") # disable for nvvp
include("utils.jl")
include("ray_generators.jl")
#include("ray_imagers.jl") # disable for nvvp
#include("rgb_spectrum.jl") # disable for nvvp
include("atomic_argmin.jl")
include("hitters.jl")
#include("partition.jl") # this brings in geo.jl # disable for nvvp
include("bv.jl")

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

function handle_optics(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, i, N, n1::F1, n2::F2, rndm) where {F1,F2}
    refracts =
        can_refract(r.dir, N(r.λ, r.x, r.y), n1(r.λ), n2(r.λ)) &&
        rndm > reflectance(r.dir, N(r.λ, r.x, r.y), n1(r.λ), n2(r.λ))

    if refracts
        return ADRay(
            next_p(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, r.λ, r.x, r.y),
            ForwardDiff.derivative(λ -> next_p(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, λ, r.x, r.y), r.λ),
            ForwardDiff.derivative(x -> next_p(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, r.λ, x, r.y), r.x),
            ForwardDiff.derivative(y -> next_p(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, r.λ, r.x, y), r.y),
            refract(r.dir, N(r.λ, r.x, r.y), n1(r.λ), n2(r.λ)),
            ForwardDiff.derivative(
                λ -> refract(d_expansion(r, λ, r.x, r.y), N(r.λ, r.x, r.y), n1(λ), n2(λ)),
                r.λ,
            ),
            ForwardDiff.derivative(
                x -> refract(d_expansion(r, r.λ, x, r.y), N(r.λ, x, r.y), n1(r.λ), n2(r.λ)),
                r.x,
            ),
            ForwardDiff.derivative(
                y -> refract(d_expansion(r, r.λ, r.x, y), N(r.λ, r.x, y), n1(r.λ), n2(r.λ)),
                r.x,
            ),
            !r.in_medium,
            i,
            r.dest,
            r.λ,
            r.x,
            r.y,
            RAY_STATUS_ACTIVE,
        )

    else
        return ADRay(
            next_p(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, r.λ, r.x, r.y),
            ForwardDiff.derivative(λ -> next_p(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, λ, r.x, r.y), r.λ),
            ForwardDiff.derivative(x -> next_p(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, r.λ, x, r.y), r.x),
            ForwardDiff.derivative(y -> next_p(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, r.λ, r.x, y), r.y),
            reflect(r.dir, N(r.λ, r.x, r.y)),
            ForwardDiff.derivative(
                λ -> reflect(d_expansion(r, λ, r.x, r.y), N(λ, r.x, r.y)),
                r.λ,
            ),
            ForwardDiff.derivative(
                x -> reflect(d_expansion(r, r.λ, x, r.y), N(r.λ, x, r.y)),
                r.x,
            ),
            ForwardDiff.derivative(
                y -> reflect(d_expansion(r, r.λ, r.x, y), N(r.λ, r.x, y)),
                r.x,
            ),
            r.in_medium,
            i,
            r.dest,
            r.λ,
            r.x,
            r.y,
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
        return handle_optics(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, i, N, glass, air, rndm)
    else
        return handle_optics(r, t, ∂t∂λ, ∂t∂x, ∂t∂y, i, N, air, glass, rndm)
    end
end

## Wrap it all up

function run_evolution!(
    hitter::AbstractHitter,
    tracer::StableTracer;
    depth,
    first_diffuse,
    n_tris,
    tris,
    rays,
    force_rand=nothing,
    kwargs...,
) where {T}
    for iter = 1:depth
        next_hit!(tracer, hitter, rays, n_tris)
        # evolve rays optically
        rand!(tracer.rndm)
        if !isnothing(force_rand)
            tracer.rndm .= force_rand
        end

        # NOTE: this triggers a D2H memcpy & gap in compute usage
        tri_view = @view tris[tracer.hit_idx]
        # everything has to be a view of the same size to avoid allocs + be sort safe
        rays .= evolve_ray.(rays, tracer.hit_idx, tri_view, tracer.rndm, first_diffuse)
    end
    return
end


function run_evolution!(
    hitter::AbstractHitter,
    tracer::ExperimentalTracer;
    depth,
    first_diffuse,
    n_tris,
    tris,
    rays,
    reorder=false,
    force_rand=nothing,
    kwargs...,
) where {T}
    cap = length(rays)
    ray_view = @view rays[1:cap]
    for iter = 1:depth
        if cap == 0
            break
        end
        ray_view = @view rays[1:cap]

        next_hit!(tracer, hitter, ray_view, n_tris)
        # evolve rays optically
        rand!(tracer.rndm)
        if !isnothing(force_rand)
            tracer.rndm .= force_rand
        end
        tri_view = @view tris[tracer.hit_idx]
        # everything has to be a view of the same size to avoid allocs + be sort safe
        rays .= evolve_ray.(rays, tracer.hit_idx, tri_view, tracer.rndm, first_diffuse)

        cap = partition!(rays, tracer.ray_swap; by=(ignore_arg, r)->r.status!=RAY_STATUS_ACTIVE)
        if cap % 512 != 0
            # TODO - I think this lazy math is inefficient
            cap = min(length(rays), ((cap ÷ 512) + 1) * 512)
        end
    end

    if reorder
        indices = map(r->r.dest, rays)
        # `copy!` is faster but causes nvvprof to fail on windows
        #copy!(tracer.ray_swap, rays)
        tracer.ray_swap .= rays
        dest_view = @view rays[indices]
        #copy!(dest_view, tracer.ray_swap)
        dest_view .= tracer.ray_swap
    end
    # needed to realign hit_idx
    next_hit!(tracer, hitter, rays, n_tris)
    return
end

function trace!(
    tracer_type::Type{T},
    hitter_type::Type{H},
    imager_type::Type{I};
    cam,
    lights,
    forward_upscale,
    backward_upscale,
    tex_f,
    tris,
    λ_min,
    dλ,
    λ_max,
    width,
    height,
    depth,
    first_diffuse,
    force_rand=nothing, # 0 to force reflection, 1 to force refraction
    intensity=1.0f0,
    iterations_per_frame=1,
    reclaim_after_iter=false,
   
) where {T<:AbstractTracer,
         H<:AbstractHitter,
         I<:AbstractImager}
    out = nothing
    @time centers, members = cluster_fuck(tris, 16)
    for frame_iter in 1:iterations_per_frame
        @sync let
        tex = tex_f()
        # initialize rays for forward tracing
        @info "ray init"
        rays = rays_from_lights(lights, forward_upscale)
        
        
        hitter = BoundingVolumeHitter(CuArray, rays, centers, members)
        #hitter = H(CuArray, rays)
        tracer = T(CuArray, rays, forward_upscale)

        spectrum, retina_factor = _spectrum_datastructs(CuArray, λ_min:dλ:λ_max)

        basic_params = Dict{Symbol,Any}()
        @pack! basic_params = width, height, dλ, λ_min, λ_max, depth, first_diffuse, intensity, force_rand
        n_tris = tuple.(Int32(1):Int32(length(tris)), map(tri_from_ftri, tris)) |> m -> reshape(m, 1, length(m))

        Λ = CuArray(collect(λ_min:dλ:λ_max))
        array_kwargs = Dict{Symbol,Any}()
        @pack! array_kwargs = tex, tris, n_tris, rays, spectrum, retina_factor

        array_kwargs = Dict(kv[1] => CuArray(kv[2]) for kv in array_kwargs)
        run_evolution!(
            hitter,
            tracer;
            basic_params...,
            array_kwargs...,
        )
        continuum_light_map!(; tracer = tracer, basic_params..., array_kwargs...)

        # reverse trace image
        RGB3 = CuArray{Float32}(undef, width * height, 3)
        RGB3 .= 0
        RGB = CuArray{RGBf}(undef, width * height)

        ray_generator(x, y, λ, dv) = camera_ray(cam, height ÷ backward_upscale, width ÷ backward_upscale, x, y, λ, dv)
        rays = wrap_ray_gen(ray_generator, height ÷ backward_upscale, width ÷ backward_upscale)

        hitter = BoundingVolumeHitter(CuArray, rays, centers, members)
        #hitter = H(CuArray, rays)
        tracer = T(CuArray, rays, backward_upscale)
        array_kwargs = Dict{Symbol,Any}()

        @pack! array_kwargs = tex, tris, n_tris, rays, spectrum, retina_factor, RGB3, RGB, rays
        array_kwargs = Dict(kv[1] => CuArray(kv[2]) for kv in array_kwargs)

        CUDA.NVTX.@range "evolver" begin CUDA.@sync begin run_evolution!(
            hitter,
            tracer;
            reorder=true,
            basic_params...,
            array_kwargs...,
        ) end end
        CUDA.NVTX.@range "shady" begin CUDA.@sync begin continuum_shade!(I(); tracer = tracer, basic_params..., array_kwargs...) end end
        @unpack RGB = array_kwargs
    	if frame_iter == 1
    		out = RGB
    	else
    		out .= out .* (frame_iter - 1) / frame_iter + RGB ./ frame_iter
    	end
        end
        if reclaim_after_iter
            # this costs ~10% in real time loop
            # but hitting OOM on large renders without
            CUDA.reclaim()
        end
    end
    return out
end
