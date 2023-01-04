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


struct ExperimentalHitter4 <: AbstractHitter
    tmp::AbstractArray{UInt64}
end

ExperimentalHitter4(A, rays) = ExperimentalHitter4(A{UInt64}(undef, size(rays)))


struct BoundingVolumeHitter <: AbstractHitter
    bvs :: AbstractArray{Sphere}
    bv_tris :: Dict{Int, AbstractArray{Int}}
    ray_queue_atomic_counters :: AbstractArray{Int}
    ray_queues :: AbstractArray{Int}
    queue_swap :: AbstractArray{Int}
    hitter :: ExperimentalHitter4
end

BoundingVolumeHitter(A, rays, bvs, memberships, concurrency::Int=4) = begin
    BoundingVolumeHitter(bvs,
                         Dict(k => A(v) for (k, v) in memberships),
                         A(zeros(Int, concurrency)),
                         A(zeros(Int, (concurrency, length(rays)))),
                         A(zeros(Int, (concurrency, length(rays)))),
                         ExperimentalHitter4(A, rays)
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
