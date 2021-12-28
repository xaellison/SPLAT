include("material.jl")
include("rgb_spectrum.jl")

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

function next_hit(rays, n_tris :: Array{Tuple{I, T}}) where {I, T}
    dest = Array{Tuple{Float32, Int32, T}}(undef, size(rays))
    @simd for i in 1:length(dest)
        dest[i] = minimum(n_tri -> get_hit(n_tri, rays[i]), n_tris, init=(Inf32, typemax(Int32), zero(T)))
    end

    dest
end

function next_hit!(dest, rays, n_tris:: Array{Tuple{I, T}}) where {I, T}
    @simd for i in 1:length(dest)
        dest[i] = minimum(n_tri -> get_hit(n_tri, rays[i]), n_tris, init=(Inf32, typemax(Int32), zero(T)))
    end
    return nothing
end

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
        if rndm <= r_s / 2.0f0
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
        elseif rndm <= (r_s + r_p) / 2.0f0
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


function evolve_ray(r::FastRay, d_n_t, rndm)::FastRay
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


    if can_refract(r.dir, N, n1, n2)
        # if we can reflect, scale probability between two polarizations
        # NB T_p + R_p + T_s + T_p = 2
        R = reflectance(r.dir, N, n1, n2)
        if rndm <= R
            return FastRay(
                p,
                refract(r.dir, N, n1, n2),
                !r.in_medium,
                n,
                r.dest,
                r.λ,
            )
        end
    end

    reflected_direction = reflect(r.dir, N)
    return FastRay(p, reflected_direction, r.in_medium, n, r.dest, r.λ)
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
    R = Dict(s => zeros(Float32, height, width) for s in keys(skys))
    G = Dict(s => zeros(Float32, height, width) for s in keys(skys))
    B = Dict(s => zeros(Float32, height, width) for s in keys(skys))
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

        for r in host_rays
            for s in keys(skys)
                R[s][r.dest] += intensity * skys[s](r.dir, r.λ, phi) * retina_red(r.λ) * dλ
                G[s][r.dest] +=
                    intensity * skys[s](r.dir, r.λ, phi) * retina_green(r.λ) * dλ
                B[s][r.dest] += intensity * skys[s](r.dir, r.λ, phi) * retina_blue(r.λ) * dλ
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
