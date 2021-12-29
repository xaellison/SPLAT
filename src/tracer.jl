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


function next_hit(rays :: AbstractArray{ADRay}, n_tris :: AbstractArray{Tuple{I, T}}) where {I, T}
    dest = Array{Tuple{Tuple{Float32, Float32}, Int32, T}}(undef, size(rays))
    next_hit!(dest, rays, n_tris)
    return dest
end

function next_hit!(dest :: Array{Tuple{Tuple{Float32, Float32}, Int32, T}}, rays, n_tris:: AbstractArray{Tuple{I, T}}) where {I, T}
    @simd for i in 1:length(dest)
        dest[i] = minimum(n_tri -> get_hit(n_tri, rays[i]), n_tris, init=((Inf32, Inf32), typemax(Int32), zero(T)))
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


function evolve_ray(r::ADRay, d_n_t, rndm)::ADRay
    (d, d′), n, t = d_n_t
    if isinf(d)
        return r
    end
    p(λ) = r.pos + # origin constant
           r.pos′ * (λ - r.λ) +  #origin linear
           (r.dir + # direction constant
           r.dir′ * (λ - r.λ)) * # ... plus direction linear
           (d + d′ * (λ - r.λ)) # times constant + linear distance

    n1, n2 = λ -> 1.0f0, glass
    if r.in_medium
        n2, n1 = n1, n2
    end
    N(λ) = optical_normal(t, p(r.λ))
    in_medium = false

    refracts = can_refract(r.dir, N(r.λ), n1(r.λ), n2(r.λ)) && rndm > reflectance(r.dir, N(r.λ), n1(r.λ), n2(r.λ))
    in_medium = refracts ? !r.in_medium : r.in_medium

    if refracts
        return ADRay(p(r.λ),
                     ForwardDiff.derivative(p, r.λ),
                     refract(r.dir, N(r.λ), n1(r.λ), n2(r.λ)),
                     ForwardDiff.derivative(λ -> refract(r.dir + r.dir′ * (λ - r.λ), N(r.λ), n1(λ), n2(λ)), r.λ),
                     in_medium, n, r.dest, r.λ)

    else
        return ADRay(p(r.λ),
                     ForwardDiff.derivative(p, r.λ),
                     reflect(r.dir, N(r.λ)),
                     ForwardDiff.derivative(λ -> reflect(r.dir + r.dir′ * (λ - r.λ), N(λ)), r.λ),
                     in_medium, n, r.dest, r.λ)
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
    R = [zeros(Float32, height, width) for s in skys]
    G = [zeros(Float32, height, width) for s in skys]
    B = [zeros(Float32, height, width) for s in skys]
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
        return ADRay(camera.pos, zero(V3), dir, zero(V3), false, 0, idx, λ)
    end

    I = map(Int32, collect(1:length(tris)))
    n_tris = collect(zip(I, tris)) |> A |> m -> reshape(m, 1, length(m))

    row_indices = A(1:width)
    col_indices = reshape(A(1:height), 1, height)
    has_run = false
    rays = nothing
    hits = nothing
    grid = nothing
    rndm = nothing
    host_rays = nothing
    dv = nothing
    I = nothing
    s0 = nothing
    #@showprogress for (iter, λ) in [(iter, λ) for iter = 1:ITERS for λ = λ_min:dλ:λ_max]
    for iter = 1:ITERS

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

        rndm .= random(Float32, width * height)
        if has_run
            next_hit!(hits, rays, n_tris)
        else
            hits = next_hit(rays, n_tris)
        end
        #	@info "tada"
        for iter = 2:depth
            rndm .= random(Float32, width * height)
            #map!(d_r -> d_r[2], rays, hits)
            map!(evolve_ray, rays, rays, hits, rndm)
            hits = next_hit(rays, n_tris)
        end
        rndm .= random(Float32, width * height)
        map!(evolve_ray, rays, rays, hits, rndm)

        if has_run
            copyto!(host_rays, rays)
        else
            host_rays = Array(rays)
        end


        for λ in λ_min:dλ:λ_max
            r0, g0, b0 = retina_red(λ), retina_green(λ), retina_blue(λ)
            for (i, sky) in enumerate(skys)

                g = r -> sky(r.dir + r.dir′ * (λ - r.λ), λ, phi)

                if isnothing(I)
                    I = map(r->r.dest, host_rays)
                else
                    map!(r->r.dest, I, host_rays)
                end

                if isnothing(s0)
                    s0 = g.(host_rays)
                else
                    map!(g, s0, host_rays)
                end

                R[i][I] .+= intensity * s0 * r0 * dλ
                G[i][I] .+= intensity * s0 * g0 * dλ
                B[i][I] .+= intensity * s0 * b0 * dλ
            end
        end

        has_run = true
    end

    out = Dict()
    for s in keys(skys)
        _R, _G, _B = R[s], G[s], B[s]
        img = Array{RGBf}(undef, size(_R)...)
        @simd for i = 1:length(img)
            if isnan(_R[i]) || isnan(_G[i]) || isnan(_B[i])
                img[i] = RGBf(1.0f0, 0.0f0, 0.0f0)
            else
                r, g, b = clamp(_R[i], 0, 1), clamp(_G[i], 0, 1), clamp(_B[i], 0, 1)
                img[i] = RGBf(r, g, b)
            end
        end
        out[s] = img
    end
    return out
end



function frame_matrix(
    camera_generator::Function,
    width::Int,
    height::Int,
    tris,
    skys,
    dλ,
    depth,
    ITERS,
    phi,
    random,
    A, # type: either Array or CuArray
)
    camera = camera_generator(1, 1)
    R = [zeros(Float32, height, width) for s in skys]
    G = [zeros(Float32, height, width) for s in skys]
    B = [zeros(Float32, height, width) for s in skys]
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
        return Ray(camera.pos, dir, polarization, false, 0, idx, λ)
    end

    I = map(Int32, collect(1:length(tris)))
    n_tris = collect(zip(I, tris)) |> A |> m -> reshape(m, 1, length(m))

    row_indices = A(1:width)
    col_indices = reshape(A(1:height), 1, height)
    has_run = false
    rays = nothing
    hits = nothing
    grid = nothing
    rndm = nothing
    host_rays = nothing
    dv = nothing
    I = nothing
    s0 = nothing
    @showprogress for (iter, λ) in [(iter, λ) for iter = 1:ITERS for λ = λ_min:dλ:λ_max]

        if has_run
            dv .=
                V3.(
                    random(Float32, width),
                    random(Float32, width),
                    random(Float32, width),
                )
            rays .= reshape(init_ray.(row_indices, col_indices, λ, dv), width * height)
        else
            rndm = random(Float32, width * height)
            dv =
                V3.(
                    random(Float32, width),
                    random(Float32, width),
                    random(Float32, width),
                )
            rays = reshape(init_ray.(row_indices, col_indices, λ, dv), width * height)
        end

        rndm .= random(Float32, width * height)
        if has_run
            next_hit!(hits, rays, n_tris)
        else
            hits = next_hit(rays, n_tris)#minimum(Broadcast.broadcasted(get_hit, n_tris, rays), dims=2, init=(Inf32, typemax(Int32), zero(STri)))
        end
        #	@info "tada"
        for iter = 2:depth
            rndm .= random(Float32, width * height)
            #map!(d_r -> d_r[2], rays, hits)
            map!(evolve_ray, rays, rays, hits, rndm)
            hits = next_hit(rays, n_tris)#hits = minimum(Broadcast.broadcasted(get_hit, n_tris, rays), dims=2, init=(Inf32, typemax(Int32), zero(STri)))
        end
        rndm .= random(Float32, width * height)
        map!(evolve_ray, rays, rays, hits, rndm)

        if has_run
            copyto!(host_rays, rays)
        else
            host_rays = Array(rays)
        end

        r0, g0, b0 = retina_red(λ), retina_green(λ), retina_blue(λ)
        for (i, sky) in enumerate(skys)

            g = r -> sky(r.dir, λ, phi)

            if isnothing(I)
                I = map(r->r.dest, host_rays)
            else
                map!(r->r.dest, I, host_rays)
            end

            if isnothing(s0)
                s0 = g.(host_rays)
            else
                map!(g, s0, host_rays)
            end

            R[i][I] .+= intensity * s0 * r0 * dλ
            G[i][I] .+= intensity * s0 * g0 * dλ
            B[i][I] .+= intensity * s0 * b0 * dλ
        end


        has_run = true

    end

    out = Dict()
    for s in keys(skys)
        _R, _G, _B = R[s], G[s], B[s]
        img = Array{RGBf}(undef, size(_R)...)
        @simd for i = 1:length(img)
            if isnan(_R[i]) || isnan(_G[i]) || isnan(_B[i])
                img[i] = RGBf(1.0f0, 0.0f0, 0.0f0)
            else
                r, g, b = clamp(_R[i], 0, 1), clamp(_G[i], 0, 1), clamp(_B[i], 0, 1)
                img[i] = RGBf(r, g, b)
            end
        end
        out[s] = img
    end
    return out
end
