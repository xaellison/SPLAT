using FileIO
using LinearAlgebra
using MeshIO
using Revise
using Rotations
using StaticArrays
using Test

import Base: rand, typemax, isless, one, zero


const V3 = SVector{3,Float32}
const INVALID_V3 = SVector(Inf, Inf, Inf)
const _COS_45 = 1 / sqrt(2)

# element 1 is normal
const Tri = SVector{4,V3}

const STri = SVector{7,V3}
const FTri = SVector{10, V3}

struct Sphere
    origin :: V3
    radius :: Float32
end

struct Cam
    pos::V3
    look_at::V3
    dir::V3
    up::V3
    right::V3
    FOV_half_sin::Float32
end

abstract type AbstractRay end

struct ADRay <: AbstractRay
    pos::V3
    pos′::V3
    dir::V3
    dir′::V3
    in_medium::Bool
    ignore_tri::Int
    dest::Int
    λ::Float32
    status::UInt8
end

const RAY_STATUS_ACTIVE = UInt8(1)
const RAY_STATUS_DIFFUSE = UInt8(0)
const RAY_STATUS_INFINITY = UInt8(2)

function retire(ray::ADRay, status)
    return ADRay(ray.pos, ray.pos′, ray.dir, ray.dir′,
                 ray.in_medium, ray.ignore_tri, ray.dest, ray.λ, status)
end

struct FastRay <: AbstractRay
    pos::V3
    dir::V3
    ignore_tri::Int
end

Base.zero(::V3) = V3(0.0f0, 0.0f0, 0.0f0)

Base.zero(::Type{FastRay}) = FastRay(zero(V3), zero(V3), 1)
Base.zero(::Type{ADRay}) = ADRay(zero(V3), zero(V3), zero(V3), zero(V3), false, 1, -1, 0.0f0, zero(UInt8))

function expand(r :: ADRay, λ :: Float32) :: FastRay
    return FastRay(
        r.pos + r.pos′ * (λ - r.λ),
        r.dir + r.dir′ * (λ - r.λ),
        r.ignore_tri,
    )
end

function rand(::V3)
    return V3((rand(Float32, 3) .- 0.5) .* 2...)
end

function get_camera(pos, lookat, true_up, fov)
    # return a camera with normalized direction vectors and all fields populated
    # fov in radians
    dir = normalize(lookat - pos)
    up = rotation_matrix(cross(dir, true_up), pi / 2) * dir
    right = cross(dir, up)
    right = normalize(right)

    return Cam(pos, lookat, dir, up, right, sin(fov / 2))
end

function optical_normal(s::Sphere, p)
    return normalize(p - s.origin)
end

function optical_normal(t::Tri, p)
    t[1]
end

function optical_normal(t::STri, pos :: V) :: V where V
    #return t[1]
    a, b, c = t[2], t[3], t[4]
    n_a, n_b, n_c = t[5], t[6], t[7]
    det = (b[2] - c[2]) * (a[1] - c[1]) + (c[1] - b[1]) * (a[2] - c[2])
    lambda1 = (b[2] - c[2]) * (pos[1] - c[1]) + (c[1] - b[1]) * (pos[2] - c[2])
    lambda1 /= det
    lambda2 = (c[2] - a[2]) * (pos[1] - c[1]) + (a[1] - c[1]) * (pos[2] - c[2])
    lambda2 /= det
    lambda3 = 1 - lambda1 - lambda2
    lambda1 = clamp(lambda1, 0, 1)
    lambda2 = clamp(lambda2, 0, 1)
    lambda3 = clamp(lambda3, 0, 1)
    return normalize(n_a * lambda1 + n_b * lambda2 + n_c * lambda3)
end

function optical_normal(t::FTri, pos::V) :: V where V
    optical_normal(STri(t[1], t[2], t[3], t[4], t[5], t[6], t[7]), pos)
end

function reverse_uv(r, t)
    reverse_uv(r.pos, t)
end

function reverse_uv(pos::V3, t::FTri) :: Pair{Float32, Float32}
    a, b, c = t[2], t[3], t[4]
    t_a, t_b, t_c = t[8], t[9], t[10]
    det = (b[2] - c[2]) * (a[1] - c[1]) + (c[1] - b[1]) * (a[2] - c[2])
    lambda1 = (b[2] - c[2]) * (pos[1] - c[1]) + (c[1] - b[1]) * (pos[2] - c[2])
    lambda1 /= det
    lambda2 = (c[2] - a[2]) * (pos[1] - c[1]) + (a[1] - c[1]) * (pos[2] - c[2])
    lambda2 /= det
    lambda3 = 1 - lambda1 - lambda2
    lambda1 = clamp(lambda1, 0, 1)
    lambda2 = clamp(lambda2, 0, 1)
    lambda3 = clamp(lambda3, 0, 1)
    t_vec = t_a * lambda1 + t_b * lambda2 + t_c * lambda3
    return Pair(t_vec[1], t_vec[2])
end

function reverse_uv(pos::V3, s::Sphere) :: Pair{Float32, Float32}
    x, y, z = normalize(pos - s.origin)
    y, z, x= x, y, z
    θ = atan(y, x)
    ϕ = atan(sqrt(x^2 + y^2), z)
    return Pair(θ / (2 * pi) + 0.5f0, ϕ / (2 * pi) + 0.5f0)
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
    prepend!(out, [Tri(zero(V3), zero(V3), zero(V3), zero(V3))])
    out
end

function mesh_to_STri(mesh)::Array{STri}
    out = []
    for face in mesh
        (v1, v2, v3) = map(p -> V3(p.position), face.points)
        (n1, n2, n3) = map(p -> V3(p.normals), face.points)
        push!(out, STri(cross(v1 - v2, v2 - v3), v1, v2, v3, n1, n2, n3))
    end
    # prepend degenerate triangle which will alway fail hit tests
    prepend!(out, [STri(zero(V3), zero(V3), zero(V3), zero(V3), zero(V3), zero(V3), zero(V3))])
    out
end

function mesh_to_FTri(mesh)::Array{FTri}
    out = []
    for face in mesh
        (v1, v2, v3) = map(p -> V3(p.position), face.points)
        (n1, n2, n3) = map(p -> V3(p.normals), face.points)
        (t1, t2, t3) = map(p -> V3(p.uv..., 0), face.points)
        push!(out, FTri(cross(v1 - v2, v2 - v3), v1, v2, v3, n1, n2, n3, t1, t2, t3))
    end
    # prepend degenerate triangle which will alway fail hit tests
    prepend!(out, [FTri(zero(V3), zero(V3), zero(V3), zero(V3), zero(V3), zero(V3), zero(V3), zero(V3), zero(V3), zero(V3))])
    out
end

function parse_obj(path)::AbstractArray{STri}
    vertices = []
    normals = []
    faces = []
    face_normals = []
    s = open(path) do file
        for line in readlines(file)
            words = split(line)
            if length(words) >= 4
                if words[1] == "v"
                    x, y, z = map(x -> parse(Float32, x), words[2:4])
                    vertex = V3(x, y, z)
                    push!(vertices, vertex)
                elseif words[1] == "f"
                    dests = Dict(1 => faces, 3 => face_normals)
                    for i in (1, 3)
                        integers = map(
                            x -> parse(Int32, split(x, "/")[i]),
                            words[2:length(words)],
                        )
                        naturals::Array{Int32} =
                            map(x -> x < 0 ? length(vertices) - x : x, integers)
                        face = Array{Int32,1}(naturals)
                        process_face(face, dests[i])
                    end
                elseif words[1] == "vn"
                    x, y, z = map(x -> parse(Float32, x), words[2:4])
                    normal = V3(x, y, z)
                    push!(normals, normal)
                end
            end
        end
    end
    # we require triangulated models
    @test all(map(length, faces)[i] == 3 for i = 1:length(faces))
    # Normal indices may not align with vertices'
    final_normals = Array{V3,1}(undef, length(normals))
    for (face_vertices, face_normals) in zip(faces, face_normals)
        for i = 1:3
            dest_index = face_vertices[i]
            src_index = face_normals[i]
            final_normals[dest_index] = normals[src_index]
        end
    end

    #plane_normals = compute_normals(vertices, faces)
    out = []
    for f in faces
        a, b, c = vertices[f[1]], vertices[f[2]], vertices[f[3]]
        q, r, s = final_normals[f[1]], final_normals[f[2]], final_normals[f[3]]
        push!(
            out,
            STri(
                cross(a - b, b - c),
                a,
                b,
                c,
                cross(a - b, b - c),
                cross(a - b, b - c),
                cross(a - b, b - c),
            ),
        )
    end
    return out
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

function model_box(tris::Array{T}) where {T <: Union{Tri, STri}}
    min_x = minimum(map(t -> minimum(map(v -> v[1], t[2:4])), tris))
    min_y = minimum(map(t -> minimum(map(v -> v[2], t[2:4])), tris))
    min_z = minimum(map(t -> minimum(map(v -> v[3], t[2:4])), tris))
    max_x = maximum(map(t -> maximum(map(v -> v[1], t[2:4])), tris))
    max_y = maximum(map(t -> maximum(map(v -> v[2], t[2:4])), tris))
    max_z = maximum(map(t -> maximum(map(v -> v[3], t[2:4])), tris))
    return (min_x, max_x, min_y, max_y, min_z, max_z)
end

function _centroid(tris)::V3
    avg(i) = sum(sum(v[i] for v in t[2:4]) for t in tris) / (3 * length(tris))
    return V3(map(avg, [1,2,3])...)
end

function centroidish(tris) :: V3
    v = model_box(tris)
    lo = V3(v[1:2:end]...)
    hi = V3(v[2:2:end]...)
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

function distance_to_sphere(r_pos, r_dir, s :: Sphere)
    # schwartz inequality: radical can't be positive if radius = 0
    radical = dot(r_dir, r_pos - s.origin) ^ 2 - (norm(r_pos - s.origin) ^ 2 - s.radius ^ 2)
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

@inline function distance_to_plane(
    origin,
    dir,
    plane_point,
    normal)
    normal = normalize(normal)
    dist = (dot(normal, plane_point) - dot(normal, origin)) / dot(normal, dir)
end

function same_side(p1, p2, _a, _b)
    # if t >= 0 then p1 and p2 are on the same side of segment a b
    cp1 = cross(_b - _a, p1 - _a)
    cp2 = cross(_b - _a, p2 - _a)
    return dot(cp1, cp2) >= 0
end

function in_triangle(p, a, b, c)
    if !same_side(p, a, b, c)
        return false
    end
    if !same_side(p, b, a, c)
        return false
    end
    if !same_side(p, c, a, b)
        return false
    end
    return true
end

function in_triangle(temp_hit, geometry, tri_id)
    f = geometry.faces[tri_id]
    a = geometry.vertices[f[1]]
    b = geometry.vertices[f[2]]
    c = geometry.vertices[f[3]]
    return in_triangle(temp_hit.hit, a, b, c)
end

function intersection(ray, geometry, tri_id)::Hit
    pos, dir = ray
    a = geometry.vertices[geometry.faces[tri_id]][1]
    normal = geometry.normals[tri_id]
    dist = distance_to_plane(pos, normalize(dir), a, normal)
    p = pos + normalize(dir) * dist

    f = geometry.faces[tri_id]
    a = geometry.vertices[f[1]]
    b = geometry.vertices[f[2]]
    c = geometry.vertices[f[3]]
    if in_triangle(p, a, b, c)
        return Hit(p, dist, tri_id)
    else
        return Hit(p, -1, -1)
    end
end

function vector_cosine(a::V3, b::V3)::Float32
    return dot(a, b) / (norm(a) * norm(b))
end

function project(a::V3, b::V3)::V3
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

function reflectance_s(v::V3, normal::V3, n1, n2)
    # https://en.wikipedia.org/wiki/Schlick%27s_approximation
    # fascinatingly, schlick gives non-zero reflectance if the media have same n
    # so we resort to using one of the fresnel's arbitrarily
    #https://en.wikipedia.org/wiki/Fresnel_equations
    # Additionally, we use the average of s and p polarizations
    # https://en.wikipedia.org/wiki/Schlick%27s_approximation
    n = normalize(normal)
    if dot(n, v) < 0
        n = n * -1
    end
    c1 = dot(normalize(v), n)

    s1 = sqrt(1 - c1^2)
    s2 = sqrt(1 - (n1 / n2)^2 * s1^2)
    return ((n1 * c1 - n2 * s2) / (n1 * c1 + n2 * s2))^2
end

function reflectance_p(v::V3, normal::V3, n1, n2)
    # https://en.wikipedia.org/wiki/Schlick%27s_approximation
    # fascinatingly, schlick gives non-zero reflectance if the media have same n
    # so we resort to using one of the fresnel's arbitrarily
    #https://en.wikipedia.org/wiki/Fresnel_equations
    # Additionally, we use the average of s and p polarizations
    # https://en.wikipedia.org/wiki/Schlick%27s_approximation
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

function can_refract_λ(v, normal, n1, n2, λ::N) where N
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

function refract_λ(v, n, n1, n2, λ::N) where N
    return refract(v(λ), normal(λ), n1(λ), n2(λ))
end
