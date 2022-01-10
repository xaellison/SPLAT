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
        return (Inf32, n, t)
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

## !! Hit computers for AD and non-AD Rays

function next_hit(rays :: AbstractArray{R}, n_tris :: AbstractArray{Tuple{I, T}}) where {R<:AbstractRay, I, T}
    dest = Array{Tuple{Float32, Int32, T}}(undef, size(rays))
    next_hit!(dest, rays, n_tris)
    return dest
end

function next_hit!(dest :: Array{Tuple{Float32, Int32, T}}, rays, n_tris:: AbstractArray{Tuple{I, T}}) where {I, T}
    @simd for i in 1:length(dest)
        dest[i] = minimum(n_tri -> get_hit(n_tri, rays[i]), n_tris, init=(Inf32, typemax(Int32), zero(T)))
    end
    return nothing
end


function next_hit(rays :: AbstractArray{ADRay}, n_tris :: AbstractArray{Tuple{I, T}}, override) where {I, T}
    dest = Array{Tuple{Tuple{Float32, Float32}, Int32, T}}(undef, size(rays))
    next_hit!(dest, rays, n_tris, override)
    return dest
end

function next_hit!(dest :: Array{Tuple{Tuple{Float32, Float32}, Int32, T}}, rays, n_tris:: AbstractArray{Tuple{I, T}}, override=false) where {I, T}
    for i in 1:length(dest)
        if !rays[i].retired || override
            dest[i] = minimum(n_tri -> get_hit(n_tri, rays[i]), n_tris, init=((Inf32, Inf32), typemax(Int32), zero(T)))
        else
            dest[i] = ((Inf32, Inf32), typemax(Int32), zero(T))
        end
    end
    return nothing
end

## Ray evolvers

function evolve_ray(r::Ray, d_n_t, rndm)::Ray
    d, n, t = d_n_t
    if isinf(d)
        return r
    end
    p = r.pos + r.dir * d

    n1, n2 = 1.0f0, glass(r.λ)
    if r.in_medium
        n2, n1 = n1, n2
    end
    N = optical_normal(t, p)

    s_polarization = project(r.polarization, N)
    p_polarization = r.polarization - s_polarization

    if can_refract(r.dir, N, n1, n2)
        # if we can reflect, scale probability between two polarizations
        # NB T_p + R_p + T_s + T_p = 2
        r_s = reflectance_s(r.dir, N, n1, n2)
        r_p = reflectance_p(r.dir, N, n1, n2)
        if rndm <= (1 - r_s) / 2.0f0
            s_polarization = normalize(s_polarization)
            return Ray(
                p,
                refract(r.dir, N, n1, n2),
                s_polarization,
                !r.in_medium,
                n,
                r.dest,
                r.λ,
            )
        elseif rndm <= (2 - r_s - r_p) / 2.0f0
            p_polarization = normalize(p_polarization)
            return Ray(
                p,
                refract(r.dir, N, n1, n2),
                p_polarization,
                !r.in_medium,
                n,
                r.dest,
                r.λ,
            )
        end
    end
    # NB this guard clause setup allows us to reflect if it is necessary (transmission = 0)
    # or if it was simply selected probabalistically.
    reflected_direction = reflect(r.dir, N)
    reflected_polarization = normalize(cross(reflected_direction, p_polarization))
    return Ray(p, reflected_direction, reflected_polarization, r.in_medium, n, r.dest, r.λ)

end

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

function evolve_ray(r::ADRay, d_n_t :: Tuple{Tuple{R, R}, I, T}, rndm)::ADRay where {R<:Real, I<:Integer, T}
    (d, d′), n, t = d_n_t
    if r.retired
        return r
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
    tris,
    skys,#::AbstractArray{Function},
    dλ,
    depth,
    ITERS,
    phi,
    random,
    A, # type: either Array or CuArray
)
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

    I = map(Int32, collect(1:length(tris)))
    n_tris = collect(zip(I, tris)) |> A |> m -> reshape(m, 1, length(m)) |> A

    row_indices = A(1:width)
    col_indices = reshape(A(1:height), 1, height)
    has_run = false
    rays = nothing
    hits = nothing
    grid = nothing
    rndm = nothing
    dv = nothing
    I = nothing
    s0 = nothing
    #@showprogress for (iter, λ) in [(iter, λ) for iter = 1:ITERS for λ = λ_min:dλ:λ_max]
    @info "tracing depth = $depth"
    @time for iter = 1:ITERS

        if has_run
            dv .=
                V3.(
                    random(Float32, width),
                    random(Float32, width),
                    random(Float32, width),
                )
            rays .= reshape(init_ray.(row_indices, col_indices, 550.0, dv), width * height)
        else
            rndm = random(Float32, width * height)
            dv =
                V3.(
                    random(Float32, width),
                    random(Float32, width),
                    random(Float32, width),
                )
            rays = reshape(init_ray.(row_indices, col_indices, 550.0, dv), width * height)
        end

        rndm .= random(Float32, length(rays))
        if has_run
            next_hit!(hits, rays, n_tris, false)
        else
            CUDA.@time CUDA.@sync hits = next_hit(rays, n_tris, true)
        end
        #
        for iter = 2:depth
            rndm .= random(Float32, length(rays))
            #map!(d_r -> d_r[2], rays, hits)
            map!(evolve_ray, rays, rays, hits, rndm)
            #if iter == 2
            CUDA.@time CUDA.@sync     sort!(rays, by=ray->ray.retired)
            cutoff = count(ray->!ray.retired, rays)
            @info "cutoff $cutoff"
            cutoff = min(length(rays), cutoff + 256 - cutoff % 256)
                @info "cutoff $cutoff"
            #end
            h_view = @view hits[1:cutoff]
            r_view = @view rays[1:cutoff]
            CUDA.@time CUDA.@sync next_hit!(h_view, r_view, n_tris, false)
        end
        rndm .= random(Float32, width * height)
        map!(evolve_ray, rays, rays, hits, rndm)
        has_run = true
    end

    frame_n = 180
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

                if isnothing(I)
                    I = map(r->r.dest, rays)
                else
                    map!(r->r.dest, I, rays)
                end

                if isnothing(s0)
                    s0 = g.(rays)
                else
                    map!(g, s0, rays)
                end

                R[I] .+= intensity * s0 * r0 * dλ
                G[I] .+= intensity * s0 * g0 * dλ
                B[I] .+= intensity * s0 * b0 * dλ
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

            Makie.save("out/tiger/_/$(lpad(frame_i, 3, "0")).png", img)

        end
    end
    return nothing
end
