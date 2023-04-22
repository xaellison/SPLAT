"""
Structs and their constructors for modular algorithm components of type
    1. Hitter (ray / geometry intersector)
    2. Imagers (turn rays into either light map or output image)
    3. Tracer (organize pipeline)
"""

abstract type AbstractHitter end

struct StableHitter <: AbstractHitter
    tmp::AbstractArray{Tuple{Float32,Int32}}
end

StableHitter(A, rays) = StableHitter(A{Tuple{Float32,Int32}}(undef, length(rays)))

struct ExperimentalHitter <: AbstractHitter
    tmp::AbstractArray{UInt64}
end

ExperimentalHitter(A, rays) = ExperimentalHitter(A{UInt64}(undef, size(rays)))

struct ExperimentalHitter2 <: AbstractHitter
    tmp::AbstractArray{UInt64}
end 

ExperimentalHitter2(A, rays) = ExperimentalHitter2(A{UInt64}(undef, size(rays)))

struct ExperimentalHitter3 <: AbstractHitter
    tmp::AbstractArray{UInt64}
end

ExperimentalHitter3(A, rays) = ExperimentalHitter3(A{UInt64}(undef, size(rays)))
ExperimentalHitter3(A, ray_length::Int) = ExperimentalHitter3(A{UInt64}(undef, ray_length))


struct ExperimentalHitter4 <: AbstractHitter
    tmp::AbstractArray{UInt64}
end

ExperimentalHitter4(A, rays) = ExperimentalHitter4(A{UInt64}(undef, size(rays)))
ExperimentalHitter4(A, ray_length::Int) = ExperimentalHitter4(A{UInt64}(undef, ray_length))


struct BoundingVolumeHitter{BV} <: AbstractHitter
    bvs :: AbstractArray{BV}
    bv_tris :: Dict{Int, AbstractArray{Int}}
    ray_queue_atomic_counters :: AbstractArray{Int}
    ray_queues :: AbstractArray{Int}
    queue_swap :: AbstractArray{Int}
    hitter :: ExperimentalHitter4
end

struct ExperimentalHitter6 <: AbstractHitter
    tmp::AbstractArray{UInt64}
end

ExperimentalHitter6(A, rays) = ExperimentalHitter6(A{UInt64}(undef, size(rays)))
ExperimentalHitter6(A, ray_length::Int) = ExperimentalHitter6(A{UInt64}(undef, ray_length))

BoundingVolumeHitter(A, ray_length::Int, bvs :: AbstractArray{BV}, memberships; concurrency::Int=4) where BV = begin
    BoundingVolumeHitter{BV}(bvs,
                         Dict(k => A(v) for (k, v) in memberships),
                         A(zeros(Int, concurrency)),
                         A(zeros(Int, (concurrency, ray_length))),
                         A(zeros(Int, (concurrency, ray_length))),
                         ExperimentalHitter4(A, ray_length)
                         )
end

# dynamic programming bounding volume
struct DPBVHitter{BV} <: AbstractHitter
    bvs :: AbstractArray{BV}
    bv_tri_count :: AbstractArray{Int}
    bv_tris :: AbstractArray{Int}
    ray_queue_atomic_counters :: AbstractArray{Int}
    ray_queues :: AbstractArray{Int}
    queue_swap :: AbstractArray{Int}
    tmp::AbstractArray{UInt64}
end

DPBVHitter(A, ray_length::Int, tris, bvs::AbstractArray{BV}, memberships; concurrency::Int=16) where BV = begin
    @assert concurrency <= length(bvs)
    DPBVHitter{BV}(bvs,
                pack_bv_tris(A, tris, bvs, memberships)...,
                A(zeros(Int, concurrency)),
                A(zeros(Int, (concurrency, ray_length))),
                A(zeros(Int, (concurrency, ray_length))),
                A{UInt64}(undef, ray_length),
                )
end


abstract type AbstractTracer end

struct StableTracer <: AbstractTracer
    hit_idx::AbstractArray{Int32}
    rndm::AbstractArray{Float32}
    δ
end

struct ExperimentalTracer <: AbstractTracer
    hit_idx::AbstractArray{Int32}
    rndm::AbstractArray{Float32}
    ray_swap::AbstractArray{ADRay}
    δ
end

upres(N::Int) = (Float32(1):Float32(N)) ./ N .-0.5f0 .- (1.0f0 / (2.0f0 * N))

StableTracer(A, rays, N) = StableTracer(A{Int32}(undef, length(rays)), A{Float32}(undef, length(rays)), upres(N))

ExperimentalTracer(A, rays, N) = ExperimentalTracer(A{Int32}(undef, length(rays)),
                                                    A{Float32}(undef, length(rays)),
                                                    similar(rays),
                                                    upres(N))


abstract type AbstractImager end

struct StableImager <: AbstractImager end

struct ExperimentalImager <: AbstractImager end

struct ExperimentalImager2 <: AbstractImager end
