using FileIO
using LinearAlgebra
using MeshIO
using Revise
using Rotations
using StaticArrays
using Test

import Base: rand, typemax, isless, one, zero


const ℜ³ = SVector{3,Float32}
const _COS_45 = 1 / sqrt(2)

# element 1 is normal
const Tri = SVector{4,ℜ³}
const STri = SVector{7,ℜ³}
const FTri = SVector{10,ℜ³}

tri_from_ftri(ftri) = Tri(ftri[1], ftri[2], ftri[3], ftri[4])

struct Sphere
    origin::ℜ³
    radius::Float32
end

struct Cam
    pos::ℜ³
    look_at::ℜ³
    dir::ℜ³
    up::ℜ³
    right::ℜ³
    FOV_half_sin::Float32
end

abstract type AbstractRay end

struct ADRay <: AbstractRay
    pos::ℜ³
    ∂p∂λ::ℜ³
    ∂p∂x::ℜ³
    ∂p∂y::ℜ³
    dir::ℜ³
    ∂d∂λ::ℜ³
    ∂d∂x::ℜ³
    ∂d∂y::ℜ³
    in_medium::Bool
    ignore_tri::Int
    dest::Int
    λ::Float32
    x::Float32
    y::Float32
    status::UInt8
end

const RAY_STATUS_ACTIVE = UInt8(1)
const RAY_STATUS_DIFFUSE = UInt8(0)
const RAY_STATUS_INFINITY = UInt8(2)

function retire(ray::ADRay, status)
    return ADRay(
        ray.pos,
        ray.∂p∂λ,
        ray.∂p∂x,
        ray.∂p∂y,
        ray.dir,
        ray.∂d∂λ,
        ray.∂d∂x,
        ray.∂d∂y,
        ray.in_medium,
        ray.ignore_tri,
        ray.dest,
        ray.λ,
        ray.x,
        ray.y,
        status,
    )
end

struct FastRay <: AbstractRay
    pos::ℜ³
    dir::ℜ³
    ignore_tri::Int
end

FastRay(adray::ADRay) = FastRay(adray.pos, adray.dir, adray.ignore_tri)

Base.zero(::ℜ³) = ℜ³(0.0f0, 0.0f0, 0.0f0)

Base.zero(::Type{FastRay}) = FastRay(zero(ℜ³), zero(ℜ³), 1)
Base.zero(::Type{ADRay}) = ADRay(
    zero(ℜ³),
    zero(ℜ³),
    zero(ℜ³),
    zero(ℜ³),
    zero(ℜ³),
    zero(ℜ³),
    zero(ℜ³),
    zero(ℜ³),
    false,
    1,
    -1,
    zero(Float32),
    zero(Float32),
    zero(Float32),
    zero(UInt8),
)

translate(t::Tri, v) = Tri(t[1], t[2] + v, t[3] + v, t[4] + v)
translate(t::STri, v) = STri(t[1], t[2] + v, t[3] + v, t[4] + v, t[5], t[6], t[7])
translate(t::FTri, v) =
    FTri(t[1], t[2] + v, t[3] + v, t[4] + v, t[5], t[6], t[7], t[8], t[9], t[10])
rotate(t::FTri, R) = FTri(
    R * t[1],
    R * t[2],
    R * t[3],
    R * t[4],
    R * t[5],
    R * t[6],
    R * t[7],
    t[8],
    t[9],
    t[10],
)


function expand(r::ADRay, λ::Float32)::FastRay
    return FastRay(r.pos + r.∂p∂λ * (λ - r.λ), r.dir + r.∂d∂λ * (λ - r.λ), r.ignore_tri)
end


function expand(r::ADRay, λ::Float32, x::Float32, y::Float32)::FastRay
    return FastRay(p_expansion(r, λ, x, y), d_expansion(r, λ, x, y), r.ignore_tri)
end

function rand(::ℜ³)
    return ℜ³((rand(Float32, 3) .- 0.5) .* 2...)
end

function get_camera(pos, lookat, true_up, fov)
    # return a camera with normalized direction vectors and all fields populated
    # fov in radians
    dir = normalize(lookat - pos)

    right = normalize(cross(dir, true_up))
    up = normalize(cross(right, dir))

    return Cam(pos, lookat, dir, up, right, sin(fov / 2))
end

function optical_normal(s::Sphere, p)
    return normalize(p - s.origin)
end

function optical_normal(t::Tri, p)
    t[1]
end

function optical_normal(t::STri, pos::V)::V where {V}
    #return t[1]
    a, b, c = t[2], t[3], t[4]
    n_a, n_b, n_c = t[5], t[6], t[7]
    u, v = reverse_uv(pos, t)
    return normalize(n_a * u + n_b * v + n_c * (1 - u - v))
end

function optical_normal(t::FTri, pos::V)::V where {V}
    optical_normal(STri(t[1], t[2], t[3], t[4], t[5], t[6], t[7]), pos)
end

function tex_uv(r::R, t) where {R<:AbstractRay}
    tex_uv(r.pos, t)
end

function tex_uv(P, t::Sphere)
    return reverse_uv(P, t)
end

function tex_uv(P, t)
    u, v = reverse_uv(P, t)
    w = (1 - u - v)
    uvw = (u, v, w)
    U = (t[8][1], t[9][1], t[10][1])
    V = (t[8][2], t[9][2], t[10][2])
    return sum(U .* uvw), sum(V .* uvw)
end


function reverse_uv(P, t)
    # returns barycentric coords describing position of P in t. Not to be
    # confused with texture uv coords
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
    v0, v1, v2 = t[2], t[3], t[4]

    v0v1 = v1 - v0
    v0v2 = v2 - v0
    #// no need to normalize
    N = cross(v0v1, v0v2)
    denom = dot(N, N)

    edge1 = v2 - v1
    vp1 = P - v1
    C = cross(edge1, vp1)
    u = dot(N, C)

    #// edge 2
    edge2 = v0 - v2
    vp2 = P - v2
    C = cross(edge2, vp2)
    v = dot(N, C)

    u /= denom
    v /= denom
    return u, v
end

function reverse_uv(pos::ℜ³, s::Sphere) #:: Pair{Float32, Float32}
    # lazy test like: min/max [1]/[2]
    # maximum(reverse_uv(rand(ℜ³), Sphere(ℜ³(0.5,0.5,0.5), 1))[1] for i in 1:10000)
    x, y, z = normalize(pos - s.origin)
    ϕ = atan(sqrt(x^2 + y^2), z)
    θ = atan(y, x)
    return θ / (2 * pi) + 0.5f0, ϕ / pi
end

function texel_scaling(r, t::FTri)
    uv_area = norm(cross(t[8] - t[9], t[10] - t[8]))
    spatial_area = norm(cross(t[2] - t[4], t[3] - t[4]))
    # consider two tris, same uv area, areas = 1, 2: 2x as many rays
    # will hit the larger, but they should be the same color, so each
    # ray hitting the larger should be scaled by 1/2
    return uv_area / spatial_area
end

function texel_scaling(r, s::Sphere)
    u, v = reverse_uv(r.pos, s)
    return abs(1 / sin(v * pi)) / 100.0f0
end

function cosine_shading(r, t)
    n = optical_normal(t, r.pos)
    abs(dot(n, normalize(r.dir)))
end


function process_face(face, triangle_dest)
    # face is a face of vertex indices which may be a polygon of degree > 3.
    # this function breaks that face into triangles and inserts them to tri_dest.
    if length(face) < 3
        return error("degenerate face")
    elseif length(face) == 3
        v1, v2, v3 = face
        push!(triangle_dest, Tri(cross(v1 - v2, v2 - v3), v1, v2, v3))
    else
        # we triangulate in a fan pattern
        for i = 3:length(face)
            v1, v2, v3 = face[1], face[i-1], face[i]
            push!(triangle_dest, Tri(cross(v1 - v2, v2 - v3), v1, v2, v3))
        end
    end
end


function mesh_to_Tri(mesh)::Array{Tri}
    out = []
    for face in mesh
        process_face(face, out)
    end
    prepend!(out, [Tri(zero(ℜ³), zero(ℜ³), zero(ℜ³), zero(ℜ³))])
    out
end

function mesh_to_STri(mesh)::Array{STri}
    out = []
    for face in mesh
        (v1, v2, v3) = map(p -> ℜ³(p.position), face.points)
        (n1, n2, n3) = map(p -> ℜ³(p.normals), face.points)
        push!(out, STri(cross(v1 - v2, v2 - v3), v1, v2, v3, n1, n2, n3))
    end
    # prepend degenerate triangle which will alway fail hit tests
    prepend!(
        out,
        [STri(zero(ℜ³), zero(ℜ³), zero(ℜ³), zero(ℜ³), zero(ℜ³), zero(ℜ³), zero(ℜ³))],
    )
    out
end

function mesh_to_FTri(mesh)::Array{FTri}
    out = []
    for face in mesh
        (v1, v2, v3) = map(p -> ℜ³(p.position), face.points)
        (n1, n2, n3) = map(p -> ℜ³(p.normals), face.points)
        (t1, t2, t3) = map(p -> ℜ³(p.uv..., 0), face.points)
        push!(out, FTri(cross(v1 - v2, v2 - v3), v1, v2, v3, n1, n2, n3, t1, t2, t3))
    end
    # prepend degenerate triangle which will alway fail hit tests
    prepend!(
        out,
        [
            FTri(
                zero(ℜ³),
                zero(ℜ³),
                zero(ℜ³),
                zero(ℜ³),
                zero(ℜ³),
                zero(ℜ³),
                zero(ℜ³),
                zero(ℜ³),
                zero(ℜ³),
                zero(ℜ³),
            ),
        ],
    )
    out
end


function model_box(vertices)
    min_x = minimum(v[1] for v in vertices)
    min_y = minimum(v[2] for v in vertices)
    min_z = minimum(v[3] for v in vertices)
    max_x = maximum(v[1] for v in vertices)
    max_y = maximum(v[2] for v in vertices)
    max_z = maximum(v[3] for v in vertices)
    return (min_x, max_x, min_y, max_y, min_z, max_z)
end

function model_box(tris::Array{T}) where {T<:Union{Tri,STri,FTri}}
    min_x = minimum(map(t -> minimum(map(v -> v[1], t[2:4])), tris))
    min_y = minimum(map(t -> minimum(map(v -> v[2], t[2:4])), tris))
    min_z = minimum(map(t -> minimum(map(v -> v[3], t[2:4])), tris))
    max_x = maximum(map(t -> maximum(map(v -> v[1], t[2:4])), tris))
    max_y = maximum(map(t -> maximum(map(v -> v[2], t[2:4])), tris))
    max_z = maximum(map(t -> maximum(map(v -> v[3], t[2:4])), tris))
    return (min_x, max_x, min_y, max_y, min_z, max_z)
end

function _centroid(tris)::ℜ³
    avg(i) = sum(sum(v[i] for v in t[2:4]) for t in tris) / (3 * length(tris))
    return ℜ³(map(avg, [1, 2, 3])...)
end

function centroidish(tris)::ℜ³
    v = model_box(tris)
    lo = ℜ³(v[1:2:end]...)
    hi = ℜ³(v[2:2:end]...)
    return (lo + hi) .* 0.5f0
end

function print_model_box(vertices)
    (min_x, max_x, min_y, max_y, min_z, max_z) = model_box(vertices)
    println(min_x, " <= x <= ", max_x)
    println(min_y, " <= y <= ", max_y)
    println(min_z, " <= z <= ", max_z)
end

function rotation_matrix(axis, θ)
    x, y, z = normalize(axis)
    M = Matrix{Float32}(undef, 3, 3)
    M[1, 1] = cos(θ) + x^2 * (1 - cos(θ))
    M[1, 2] = x * y * (1 - cos(θ)) - z * sin(θ)
    M[1, 3] = x * z * (1 - cos(θ)) + y * sin(θ)
    M[2, 1] = x * y * (1 - cos(θ)) + z * sin(θ)
    M[2, 2] = cos(θ) + y^2 * (1 - cos(θ))
    M[2, 3] = z * y * (1 - cos(θ)) - x * sin(θ)
    M[3, 1] = x * z * (1 - cos(θ)) - y * sin(θ)
    M[3, 2] = z * y * (1 - cos(θ)) + x * sin(θ)
    M[3, 3] = cos(θ) + z^2 * (1 - cos(θ))
    return RotMatrix{3,Float32}(M)
end

@inline function p_expansion(r, λ, x, y)
    r.pos + # origin constant
    r.∂p∂λ * (λ - r.λ) +  #origin linear
    r.∂p∂x * (x - r.x) +  #origin linear
    r.∂p∂y * (y - r.y)
end

@inline function d_expansion(r, λ, x, y)
    r.dir + # direction constant
    r.∂d∂λ * (λ - r.λ) +# ... plus direction linear
    r.∂d∂x * (x - r.x) +
    r.∂d∂y * (y - r.y)
 end

function distance_to_sphere(r_pos, r_dir, s::Sphere)
    # schwartz inequality: radical can't be positive if radius = 0
    radical = dot(r_dir, r_pos - s.origin)^2 - (norm(r_pos - s.origin)^2 - s.radius^2)
    # if it exactly hits at 1 point, discard this infinitesimal edge case
    if radical <= 0
        return Inf32
    end
    # empirically, a scalar of 1f-7 becomes problematic
    ϵ = s.radius * 1.0f-6
    d1 = dot(s.origin - r_pos, r_dir) - sqrt(radical)
    if d1 > ϵ
        return d1
    end

    d2 = dot(s.origin - r_pos, r_dir) + sqrt(radical)
    if d2 > ϵ
        return d2
    end
    return Inf32
end

@inline function distance_to_plane(origin, dir, plane_point, normal)
    @fastmath (dot(normal, plane_point) - dot(normal, origin)) / dot(normal, dir)
end

@inline function distance_to_plane(r, T)
    @inbounds return distance_to_plane(r.pos, r.dir, T[2], T[1])
end

function distance_to_plane(r, T, λ, x, y)
    @inbounds distance_to_plane(p_expansion(r, λ, x, y), d_expansion(r, λ, x, y), T[2], T[1])
end


@inline function in_triangle(p, a, b, c)
    # https://blackpawn.com/texts/pointinpoly/
    v0 = c - a
    v1 = b - a
    v2 = p - a

    #// Compute dot products
    dot00 = dot(v0, v0)
    dot01 = dot(v0, v1)
    dot02 = dot(v0, v2)
    dot11 = dot(v1, v1)
    dot12 = dot(v1, v2)

    #// Compute barycentric coordinates
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    #// Check if point is in triangle
    return (u >= 0) && (v >= 0) && (u + v < 1)
end

function in_triangle(p, T)
    return @inbounds @fastmath in_triangle(p, T[2], T[3], T[4])
end

function vector_cosine(a::ℜ³, b::ℜ³)::Float32
    return dot(a, b) / (norm(a) * norm(b))
end

function project(a::ℜ³, b::ℜ³)::ℜ³
    # this is the result of projecting a onto b
    return b * (dot(a, b) / (norm(b)))
end

function reflect(v, normal)
    normal = normalize(normal)
    temp = v - normal * dot(normal, v) * 2
    #vector_cosine(v, normal) ≈ -1*vector_cosine(temp, normal)
    return temp
end

function reflectance(v, normal, n1, n2)
    # https://en.wikipedia.org/wiki/Schlick%27s_approximation
    # fascinatingly, schlick gives non-zero reflectance if the media have same n
    # so we resort to using the average of s and p polarizations
    # https://en.wikipedia.org/wiki/Schlick%27s_approximation
    n = normalize(normal)
    if dot(n, v) < 0
        n = n * -1
    end
    c1 = dot(normalize(v), n)

    s1 = sqrt(1 - c1^2)
    s2 = sqrt(1 - (n1 / n2)^2 * s1^2)
    return (((n1 + n2) / 2 * c1 - (n1 + n2) / 2 * s2) / (n1 * c1 + n2 * s2))^2
end

function reflectance_s(v::ℜ³, normal::ℜ³, n1, n2)
    n = normalize(normal)
    if dot(n, v) < 0
        n = n * -1
    end
    c1 = dot(normalize(v), n)

    s1 = sqrt(1 - c1^2)
    s2 = sqrt(1 - (n1 / n2)^2 * s1^2)
    return ((n1 * c1 - n2 * s2) / (n1 * c1 + n2 * s2))^2
end

function reflectance_p(v::ℜ³, normal::ℜ³, n1, n2)
    n = normalize(normal)
    if dot(n, v) < 0
        n = n * -1
    end
    c1 = dot(normalize(v), n)

    s1 = sqrt(1 - c1^2)
    s2 = sqrt(1 - (n1 / n2)^2 * s1^2)
    return ((n1 * s2 - n2 * c1) / (n1 * s2 + n2 * c1))^2
end

function can_refract(v, normal, n1, n2)::Bool
    # use the normal pointing up from the plane if dir is coming down into it
    n = normalize(normal)
    if dot(n, v) > 0.0f0
        n = -n
    end
    c1 = dot(normalize(v), n)
    ϵ = 1.0f-9
    if c1^2 > 1 - ϵ
        return false
    end
    s1 = sqrt(1.0f0 - c1 * c1)
    s2 = (n1 / n2) * s1
    return abs(s2) < 1 - ϵ
end

function can_refract_λ(v, normal, n1, n2, λ::N) where {N}
    return can_refract(v(λ), normal(λ), n1(λ), n2(λ))
end

function refract(v, normal, n1, n2)
    # use the normal pointing up from the plane if dir is coming down into it
    # see https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form for sign logic
    n = normalize(normal)

    if dot(v, n) > 0
        n = n * -1
    end
    c1 = dot(normalize(v), n)
    s1 = sqrt(1 - c1 * c1)
    s2 = (n1 / n2) * s1
    c2 = sqrt(1 - s2 * s2)
    return normalize(v * (n1 / n2) + n * ((n1 / n2) * c1 - c2))
end

function refract_λ(v, n, n1, n2, λ::N) where {N}
    return refract(v(λ), normal(λ), n1(λ), n2(λ))
end
