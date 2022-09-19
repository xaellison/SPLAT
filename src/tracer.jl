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
    t(λ) = distance_to_sphere(r.pos + (λ - r.λ) * r.pos′, r.dir + (λ - r.λ) * r.dir′, S)
    if isinf(t0)
        return ((Inf32, Inf32), one(Int32), S)
    end
    return ((t0, ForwardDiff.derivative(t, r.λ)), i, S)
end

function get_hit(
    i_T::Tuple{Int32, AbstractTri},
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
)::Tuple{Tuple{Float32,Float32},Int32,AbstractTri} where {AbstractTri}
    i, T = i_T
    # In the case of i = 1, the degenerate triangle, this will be NaN.
    # t0 = NaN fails d0 > 0 below, which properly gives us i = 1 back
    t0 = distance_to_plane(r, T)
    t(λ) = distance_to_plane(r, T, λ)
    p = r.pos + r.dir * t0
    if in_triangle(p, T) && t0 > 0 && r.ignore_tri != i
        return ((t0, ForwardDiff.derivative(t, r.λ)), i, T)
    else
        return ((Inf32, Inf32), one(Int32), T)
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

function hit_argmin(i_T, r::ADRay)::Tuple{Float32,Int32}
    return hit_argmin(i_T, FastRay(r))
end

function hit_argmin(i_T, r::FastRay)::Tuple{Float32,Int32}
    return get_hit(i_T, r)[1:2]
end


function next_hit_kernel(rays, n_tris :: AbstractArray{X}, dest :: AbstractArray{UInt64}, default ) where {X}
    # TODO: rename everything
    shmem = @cuDynamicSharedMem(Tuple{Int32, Tri}, blockDim().x)

    dest_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    r = rays[dest_idx]

    arg_min = default
    min_val = Inf32

    if threadIdx().x + (blockIdx().y - 1) * blockDim().x <= length(n_tris)
        n, FT = n_tris[threadIdx().x + (blockIdx().y - 1) * blockDim().x]
        shmem[threadIdx().x] = n, Tri(FT[1], FT[2], FT[3], FT[4])
    end
    sync_threads()
    for scan = 1:min(blockDim().x, length(n_tris) - (blockIdx().y - 1) * blockDim().x)
        i, T = shmem[scan]
        t = distance_to_plane(r, T)
        p = r.pos + r.dir * t
        if in_triangle(p, T) && min_val > t > 0 && r.ignore_tri != i
            arg_min = i
            min_val = t
        end
    end

    if dest_idx <= length(rays)
        operand = unsafe_encode(min_val, UInt32(arg_min))
        dest[dest_idx] = min(dest[dest_idx], operand)
    end
    return nothing
end




function next_hit!(dest, tmp, rays, n_tris)

    my_args = rays, n_tris, tmp, Int32(1)

    kernel = @cuda launch=false next_hit_kernel(my_args...)
    tmp.=typemax(UInt64)
    get_shmem(threads) = threads * sizeof(Tuple{Int32, Tri})
    # TODO: this is running <= 50% occupancy. Need to put a cap on shmem smaller than block
    config = launch_configuration(kernel.fun, shmem=threads->get_shmem(threads))
    threads = config.threads
    blocks = (cld(length(rays), threads), cld(length(n_tris), threads))
    kernel(my_args...; blocks=blocks, threads=threads, shmem=get_shmem(threads))
    dest .= unsafe_decode.(tmp)
    return
end

function next_hit!(dest, tmp, rays, n_tris::AbstractArray{Tuple{N, Sphere}}) where N
    # TODO: restore tmp as function arg?
    @tullio (min) tmp[i] = hit_argmin(n_tris[j], rays[i])
    d_view = @view dest[:]
    d_view = reshape(d_view, length(d_view))
    map!(x -> x[2], d_view, tmp)
end

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

function trace!(;cam, rays, tex, tris, λ_min, dλ, λ_max, width, height, depth, sort_optimization, first_diffuse)
    basic_params = Dict{Symbol, Any}()
	@pack! basic_params = width, height, dλ, λ_min, λ_max, depth, sort_optimization, first_diffuse
    n_tris = tuple.(1:length(tris), tris) |> m -> reshape(m, 1, length(m))

    Λ = CuArray(collect(λ_min:dλ:λ_max))

	datastructs = forward_datastructs(CuArray, rays; basic_params...)
    array_kwargs = Dict{Symbol, Any}()

    @pack! array_kwargs = tex, tris, n_tris, rays
	array_kwargs = merge(datastructs, array_kwargs)

    array_kwargs = Dict(kv[1]=>CuArray(kv[2]) for kv in array_kwargs)
    CUDA.@time run_evolution!(;basic_params..., array_kwargs...)

	CUDA.@time continuum_light_map!(;basic_params..., array_kwargs...)

	# reverse trace image
	datastructs = scene_datastructs(CuArray; basic_params...)
	ray_generator2(x, y, λ, dv) = camera_ray(cam, height, width, x, y, λ, dv)
	rays = wrap_ray_gen(ray_generator2, height, width)
    array_kwargs = Dict{Symbol, Any}()

    @pack! array_kwargs = tex, tris, n_tris, rays
	array_kwargs = merge(datastructs, array_kwargs)

    array_kwargs = Dict(kv[1]=>CuArray(kv[2]) for kv in array_kwargs)

	CUDA.@time run_evolution!(;basic_params..., array_kwargs...)
	CUDA.@time continuum_shade!(;basic_params..., array_kwargs...)
    return array_kwargs
end
