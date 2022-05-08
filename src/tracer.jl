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

function evolve_ray(r::ADRay, n, t, rndm)::ADRay

    (d, d′), n, t = get_hit((n, t), r)

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
    skys,#::AbstractArray{Function},
    dλ,
    depth,
    ITERS,
    phi,
    random,
    A, # type: either Array or CuArray
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
        return ADRay(camera.pos, zero(V3), dir, zero(V3), false, 0, idx, λ, false)
    end

    n_tris = collect(zip(map(Int32, collect(1:length(tris))), hit_tris)) |> A |> m -> reshape(m, 1, length(m)) |> A
    tris = A(tris)
    hit_tris = A(hit_tris)
    row_indices = A(1:width)
    col_indices = reshape(A(1:height), 1, height)
    rays = A{ADRay}(undef, height * width)
    hit_idx = A(zeros(Int32, length(rays)))
    ray_dests = map(Int32, A(1:length(rays)))
    dv = A{V3}(undef, width) # make w*h
    s0 = nothing
    #@showprogress for (iter, λ) in [(iter, λ) for iter = 1:ITERS for λ = λ_min:dλ:λ_max]

    # Datastruct init
    #rays = CuArray()
    hits = A{Tuple{Tuple{Float32, Float32}, Int32, T}}(undef, (height* width))
    rndm = random(Float32, width * height)
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
            CUDA.@time CUDA.@sync next_hit!(h_view, r_view, n_tris, false)

            # evolve rays optically
            rndm .= random(Float32, length(rays))
            tri_view = @view tris[hit_idx]
            map!(evolve_ray, rays, rays, hit_idx, tri_view, rndm)

            # retire appropriate rays
            @info "retirement sort..."
            CUDA.@time CUDA.@sync sort!(rays, by=ray->ray.retired)
            cutoff = count(ray->!ray.retired, rays)
            cutoff = min(length(rays), cutoff + 256 - cutoff % 256)
        end
    end

    frame_n = 18
    @info "images"
    R = A(zeros(Float32, height, width))
    G = A(zeros(Float32, height, width))
    B = A(zeros(Float32, height, width))
    @time for frame_i in 1:frame_n
        R .= 0.0f0
        G .= 0.0f0
        B .= 0.0f0
        for λ in λ_min:dλ:λ_max
            r0, g0, b0 = retina_red(λ), retina_green(λ), retina_blue(λ)
            for (i, sky) in enumerate(skys)
                # WARNING deleted `r.in_medium ? 0.0f0 : `
                g = r -> sky(r.dir + r.dir′ * (λ - r.λ), λ, Float32(2 * pi / 20 * frame_i / frame_n))

            @assert size(ray_dests) == size(rays)
            map!(r->r.dest, ray_dests, rays)

            @assert size(ray_dests) == size(rays)
                if isnothing(s0)
                    s0 = g.(rays)
                else
                    map!(g, s0, rays)
                end

                R[ray_dests] .+= intensity * s0 * r0 * dλ
                G[ray_dests] .+= intensity * s0 * g0 * dλ
                B[ray_dests] .+= intensity * s0 * b0 * dλ
            end
        end

        out = Dict()
        for s in keys(skys)
            _R, _G, _B = map(Array, (R, G, B))
            img = Array{RGBf}(undef, size(_R)...)
            @simd for i = 1:length(img)
                if isnan(_R[i]) || isnan(_G[i]) || isnan(_B[i])
                    img[i] = RGBf(1.0f0, 0.0f0, 0.0f0)
                else
                    r, g, b = clamp(_R[i], 0, 1), clamp(_G[i], 0, 1), clamp(_B[i], 0, 1)
                    img[i] = RGBf(r, g, b)
                end
            end

            Makie.save("out/lion/may/$(lpad(frame_i, 3, "0")).png", img)

        end
    end
    return nothing
end
