include("material.jl")
include("rgb_spectrum.jl")

using ForwardDiff
using Makie
using Serialization
using Tullio

import Random.rand!

function get_hit(n_s::Tuple{Int32,Sphere}, r::AbstractRay)
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

function get_hit(n_s::Tuple{Int32,Sphere}, r::ADRay)
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

function get_hit(n_t::Tuple{Int32,T}, r::AbstractRay)::Tuple{Float32,Int32,T} where {T}
    n, t = n_t
    d = distance_to_plane(r.pos, r.dir, t[2], t[1])
    p = r.pos + r.dir * d
    if in_triangle(p, t[2], t[3], t[4]) && d > 0 && r.ignore_tri != n
        return (d, n, t)
    else
        return (Inf32, one(Int32), t)
    end
end

function get_hit(
    n_t::Tuple{Int32,T},
    r::ADRay,
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

function hit_argmin(n_t, r::ADRay) :: Tuple{Float32, Int32}
    hit = get_hit(n_t, r)
    return (hit[1][1], hit[2])
end

function hit_argmin(n_t, r::FastRay) :: Tuple{Float32, Int32}
    return get_hit(n_t, r)[1:2]
end

function next_hit!(dest, tmp, rays, n_tris)
    # TODO: restore tmp as function arg?
    @tullio (min) tmp[i] = hit_argmin(n_tris[j], rays[i])
    d_view = @view dest[:]
    d_view = reshape(d_view, length(d_view))
    map!(x->x[2], d_view, tmp)
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

function shade(r::FastRay, n, t, first_diffuse_index)::Float32
    # NB disregards in_medium
    # evolve to hit a diffuse surface
    d, n, t = get_hit((n, t), r)
    r = FastRay(r.pos + r.dir * d, zero(V3), r.ignore_tri)

    if n >= first_diffuse_index
        # compute the position in the new triangle, set dir to zero
        u, v = reverse_uv(r.pos, t)
        s = 0.1f0
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
        u, v = reverse_uv(r.pos, t)
        # it's theoretically possible u, v could come back as zero
        i = max(1, Int(ceil(u * (size(tex)[1]))))
        j = max(1, Int(ceil(v * (size(tex)[2]))))
        return tex[i, j]
    end
    if isinf(d)
        return 0.0f0# retire(r, RAY_STATUS_INFINITY)
    end
    return 0.0f0
end

## Wrap it all up

function ad_frame_matrix(
    ;camera_generator::Function,
    height::Int,
    width::Int,
    dλ,
    depth,
    ITERS,
    sort_optimization,
    first_diffuse,
    RGB3,
    RGB,
    n_tris,
    tris,
    row_indices,
    col_indices,
    rays,
    hit_idx,
    dv,
    s0,
    hits,
    tmp,
    rndm,
    expansion,
    spectrum,
    retina_factor,
    tex,
) where {T}
    camera = camera_generator(1, 1)
    intensity = Float32(1 / ITERS)

    function init_ray(x, y, λ, dv)::AbstractRay
        _x, _y = x - height / 2, y - width / 2
        scale = height * _COS_45 / camera.FOV_half_sin
        _x /= scale
        _y /= scale

        _z = sqrt(1 - _x^2 - _y^2)
        dir =
            _x * camera.right +
            _y * camera.up +
            _z * camera.dir +
            dv * 0.25f0 / max(height, width)
        dir = normalize(dir)
        idx = (y - 1) * height + x
        return ADRay(
            camera.pos,
            zero(V3),
            dir,
            zero(V3),
            false,
            1,
            idx,
            λ,
            RAY_STATUS_ACTIVE,
        )
    end


    #@info "Stage 1: AD tracing depth = $depth"
    begin
        # FIXME
        dv .= V3.(CUDA.rand(Float32, height, width), CUDA.rand(Float32, height, width), CUDA.rand(Float32, height, width))
        rays = reshape(rays, height, width)
        rays .= init_ray.(row_indices, col_indices, 550.0, dv)
        rays = reshape(rays, height * width)
        cutoff = length(rays)

        for iter = 1:depth
            # compute hits
            #@info cutoff
            h_view = @view hit_idx[1:cutoff]
            r_view = @view rays[1:cutoff]
            #@info "$(length(r_view)) / $(length(rays)) = $(length(r_view) / length(rays))"
            #@info "hits..."
            next_hit!(h_view, tmp, r_view, n_tris)

            # evolve rays optically
            rand!(rndm)
            tri_view = @view tris[hit_idx]
            # I need to pass a scalar arg - this closure seems necessary since map! freaks at scalar args
            evolve_closure(rays, hit_idx, tri_view, rndm) =
                evolve_ray(rays, hit_idx, tri_view, rndm, first_diffuse)
            map!(evolve_closure, rays, r_view, h_view, tri_view, rndm)

            # retire appropriate rays
            if sort_optimization
                #@info "retirement sort..."
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
    end

    #@info "Stage 2: Expansion (optimization then evaluation)"
    # NB: we should also expand rays that have been dispersed AND go to infinity - the may have fringe intersections
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
            α = s.(expansion, hits, tri_view, first_diffuse)
            broadcast = (α .* retina_factor[:, :, n] .* intensity .* dλ)
            RGB3 .+= broadcast |> a -> reshape(a, length(rays), 3)
        end
    end
    map!(brightness -> clamp(brightness, 0, 1), RGB3, RGB3)
    RGB .= RGBf.(RGB3[:, 1], RGB3[:, 2], RGB3[:, 3])
    return nothing
end
