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
    dest = Array{Tuple{Float32, Int32, T}}(undef, size(rays))
    next_hit!(dest, rays, n_tris, override)
    return dest
end

function next_hit!(dest :: AbstractArray{Tuple{Float32, Int32, T}}, rays, n_tris:: AbstractArray{Tuple{I, T}}, override=false) where {I, T}

    for i in 1:length(dest)
        if !rays[i].retired || override
            dest[i] = minimum(n_tri -> get_hit(n_tri, rays[i]), n_tris, init=(Inf32, typemax(Int32), zero(T)))
        else
            dest[i] = (Inf32, typemax(Int32), zero(T))
        end
    end
    return nothing
end

## Ray evolvers

function p(r, t, λ::T1, x::T2, y::T3) where {T1, T2, T3}
    r.pos + # origin constant
    r.pos′ * (λ - r.λ) +  #origin linear
    r.pos_x′ * x +
    r.pos_y′ * y +
    (r.dir + # direction constant
    r.dir′ * (λ - r.λ) +# ... plus direction linear
    r.dir_x′ * x +
    r.dir_y′ * y
    ) * distance_to_plane(r.pos+ # origin constant
    r.pos′ * (λ - r.λ) +  #origin linear
    r.pos_x′ * x +
    r.pos_y′ * y , r.dir + # direction constant
    r.dir′ * (λ - r.λ) +# ... plus direction linear
    r.dir_x′ * x +
    r.dir_y′ * y, t[2], t[1])
end

function handle_optics(r,  n, t, N, n1 :: N1, n2::N2, rndm) where {N1, N2}
    refracts = can_refract(r.dir, N(r.λ, 0.0f0, 0.0f0), n1(r.λ), n2(r.λ)) && rndm > reflectance(r.dir, N(r.λ, 0.0f0,0.0f0), n1(r.λ), n2(r.λ))

    if refracts
        return ADRay(p(r, t, r.λ, 0.0f0, 0.0f0),
                     ForwardDiff.derivative(λ->p(r, t, λ, 0.0f0, 0.0f0), r.λ),
                     ForwardDiff.derivative(δx->p(r, t, r.λ, δx, 0.0f0), 0.0f0),
                     ForwardDiff.derivative(δy->p(r, t, r.λ, 0.0f0, δy), 0.0f0),
                     refract(r.dir, N(r.λ, 0.0f0, 0.0f0), n1(r.λ), n2(r.λ)),
                     ForwardDiff.derivative(λ -> refract(r.dir + r.dir′ * (λ - r.λ), N(λ, 0.0f0, 0.0f0), n1(λ), n2(λ)), r.λ),
                     ForwardDiff.derivative(δx -> refract(r.dir + r.dir_x′ * δx, optical_normal(t, p(r, t, r.λ, δx, 0.0f0)), n1(r.λ), n2(r.λ)), 0.0f0),
                     ForwardDiff.derivative(δy -> refract(r.dir + r.dir_y′ * δy, optical_normal(t, p(r, t, r.λ, 0.0f0, δy)), n1(r.λ), n2(r.λ)), 0.0f0),
                     !r.in_medium, n, r.dest, r.λ, N(r.λ,0.0f0,0.0f0), false)

    else
        return ADRay(p(r, t, r.λ, 0.0f0, 0.0f0),
                     ForwardDiff.derivative(λ->p(r, t, λ, 0.0f0, 0.0f0), r.λ),
                     ForwardDiff.derivative(δx->p(r, t, r.λ, δx, 0.0f0), 0.0f0),
                     ForwardDiff.derivative(δy->p(r, t, r.λ, 0.0f0, δy), 0.0f0),
                     reflect(r.dir, N(r.λ,0.0f0,0.0f0)),
                     ForwardDiff.derivative(λ -> reflect(r.dir + r.dir′ * (λ - r.λ), N(λ, 0.0f0, 0.0f0)), r.λ),
                     ForwardDiff.derivative(δx -> reflect(r.dir + r.dir_x′ * δx, N(r.λ, δx, 0.0f0)), 0.0f0),
                     ForwardDiff.derivative(δy -> reflect(r.dir + r.dir_y′ * δy, N(r.λ, 0.0f0, δy)), 0.0f0),
                     r.in_medium, n, r.dest, r.λ, N(r.λ,0.0f0,0.0f0), false)
    end
end

function evolve_ray(r::ADRay, d_n_t :: Tuple{R, I, T}, rndm)::ADRay where {R<:Real, I<:Integer, T}
    d, n, t = d_n_t
    if r.retired
        return r
    end

    if isinf(d)
        return retired(r)
    end

    N(λ, x, y) = optical_normal(t, p(r, t, λ, x, y))
    if r.in_medium
        return handle_optics(r,  n, t, N, glass, air, rndm)
    else
        return handle_optics(r, n, t, N, air, glass, rndm)
    end
end


## Wrap it all up

function diffdir(camera::Cam, _x::T1, _y::T2) where {T1, T2}
    _z = 1.0f0#sqrt(1 - _x^2 - _y^2)
    dir =
        _x * camera.right +
        _y * camera.up +
        _z * camera.dir #+
    dir |> normalize
    #dir
end

function ad_frame_matrix(
    camera_generator::Function,
    width::Int,
    height::Int,
    tris,
    sky :: S,
    dλ,
    depth,
    ITERS,
    phi,
    random,
    zeros,
    A, # type: either Array or CuArray
) where S
    camera = camera_generator(1, 1)
    CUDA.memory_status()
    #out .= RGBf(0, 0, 0)
    λ_min = 400.0f0
    λ_max = 700.0f0
    intensity = Float32(1 / ITERS) #*(1/0.06125)
    _COS_45 :: Float32 = Float32(1 / sqrt(2))
    function init_ray(x, y, λ, dv)::AbstractRay
        _x, _y = Float32(x - width / 2), Float32(y - height / 2)
        scale = Float32(width * _COS_45 / camera.FOV_half_sin)
        __x = Float32(_x / scale)
        __y = Float32(_y / scale)


            #dv * 0.25f0 / max(width, height)
        idx = (x - 1) * height + y
        dir = diffdir(camera, __x, __y)
        x̂ = ForwardDiff.derivative(X->diffdir(camera, X, __y), 0.0f0)# cross(dir, camera.up) |> normalize
        ŷ = ForwardDiff.derivative(Y->diffdir(camera, __x, Y), 0.0f0)#cross(dir, x̂) |> normalize

        return ADRay(camera.pos , zero(V3),zero(V3),zero(V3), dir, zero(V3),x̂, ŷ,  false, 0, idx, λ, dir, false)
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
    ray_iters = []
    @time for noise_iter = 1:ITERS

        if has_run
            dv .=
                V3.(
                    random(Float32, width, height),
                    random(Float32, width, height),
                    random(Float32, width, height),
                )
            rays .= reshape(init_ray.(row_indices, col_indices, 550.0, dv), width * height)
        else
            rndm = random(Float32, width * height)
            dv =
                V3.(
                    random(Float32, width, height),
                    random(Float32, width, height),
                    random(Float32, width, height),
                )
            rays = reshape(init_ray.(row_indices, col_indices, 550.0, dv), width * height)
        end
        rndm = random(Float32, length(rays)) .* (1 / ITERS) .+ ((noise_iter - 1) / ITERS)
        if has_run
            next_hit!(hits, rays, n_tris, false)
        else
            hits = next_hit(rays, n_tris, true)
        end
        #
        for iter = 2:depth
            CUDA.memory_status()
            rndm = random(Float32, length(rays)) .* (1 / ITERS) .+ ((noise_iter - 1) / ITERS)
            #map!(d_r -> d_r[2], rays, hits)
            println(eltype(hits))
            map!(evolve_ray, rays, rays, hits, rndm)
            #if iter == 2
            #CUDA.@sync sort!(rays, by=ray->ray.retired, alg=CUDA.QuickSortAlg())

            #cutoff = length(rays)
            #cutoff = count(ray->!ray.retired, rays)
            #@info "cutoff $cutoff"
            #cutoff = min(length(rays), cutoff + 256 - cutoff % 256)
            #@info "cutoff $cutoff"

            #h_view = @view hits[1:cutoff]
            #r_view = @view rays[1:cutoff]
            next_hit!(hits, rays, n_tris, false)
        end
        rndm = random(Float32, width * height) .* (1 / ITERS) .+ ((noise_iter - 1) / ITERS)
        map!(evolve_ray, rays, rays, hits, rndm)
        has_run = true
        push!(ray_iters, map(x->(x.dir, x.dir′, x.λ, x.dest), rays))
    end
    @info "syncing..."
    CUDA.synchronize()
    @info "sync'ed"
    frame_n = 180
    @info "images"
    CUDA.memory_status()
    R = zeros(Float32, height, width)

    G = zeros(Float32, height, width)
    B = zeros(Float32, height, width)

    I_s = []
    for rays in ray_iters
        @time push!(I_s, map(x->x[4], rays))
    end

    hits = reshape(hits, length(hits))
    @time for frame_i in 1:frame_n
        R .= 0.0f0
        G .= 0.0f0
        B .= 0.0f0
        for (rays, I) in zip(ray_iters, I_s)

            for λ in λ_min:dλ:λ_max
                r0, g0, b0 = retina_red(λ), retina_green(λ), retina_blue(λ)


                    # WARNING deleted `r.in_medium ? 0.0f0 : `
                    ϕ = Float32(2 * pi / 20 * frame_i / frame_n)
                    #@info "x"


                #    @info "y $(typeof(I))"
                    #if isnothing(s0)
                #    println(eltype(hits))
                        s0 = shade.(rays, sky, λ, ϕ)
                    #else
                    #    map!(r->shade(r, sky, λ, ϕ), s0, rays)
                    #end

                #    @info "z"
                    @assert length(ray_iters ) == ITERS
                    R_v = @view R[I]
                    G_v = @view G[I]
                    B_v = @view B[I]
                    R_v .+= intensity * s0 * r0 * dλ
                    G_v .+= intensity * s0 * g0 * dλ
                    B_v .+= intensity * s0 * b0 * dλ
                #    @info "zz"

            end
        end

        out = Dict()
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

        Makie.save("out/lion/_/$(lpad(frame_i, 3, "0")).png", img)
    #    print(Array(map(r->r.last_normal|>norm, rays)))
    end
    return nothing
end
