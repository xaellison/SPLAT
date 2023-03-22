function _spectrum_datastructs(A, λ_range)
    spectrum = collect(λ_range) |> a -> reshape(a, 1, 1, length(a))
    retina_factor = Array{Float32}(undef, 1, 3, length(spectrum))
    map!(retina_red, begin
        @view retina_factor[1, 1, :]
    end, spectrum)
    map!(retina_green, begin
        @view retina_factor[1, 2, :]
    end, spectrum)
    map!(retina_blue, begin
        @view retina_factor[1, 3, :]
    end, spectrum)
    retina_factor = A(retina_factor)
    return spectrum, retina_factor
end

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

function pack_bv_tris(A, tris, bvs, memberships; max_overcount_factor=4.0) :: Tuple{AbstractArray{Int}, AbstractArray{Int}}
    # for 100k tris, 256 bvs, `out` will take up 195 MB
    out = A{Int}(undef, length(bvs), Int(ceil(length(tris) / length(bvs) * max_overcount_factor)))
    host_counts = zeros(Int, length(bvs))
    for (k, v) in memberships
        out_view = @view out[k, 1:length(v)] 
        out_view .= A(sort(v))
        host_counts[k] = length(v)
    end 
    return A(host_counts), out
end

function repack!(hitter, bvs, memberships)
    hitter.bvs .= bvs
    host_counts = zeros(Int, length(bvs))
    for (k, v) in memberships
        out_view = @view hitter.bv_tris[k, 1:length(v)]
        copy!(out_view, sort(v))
 #       out_view .= CuArray(sort(v))
        host_counts[k] = length(v)
    end
    hitter.bv_tri_count .= CuArray(host_counts)
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
