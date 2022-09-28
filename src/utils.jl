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



#abstract type AbstractForwardTracer end

struct Tracer
    hit_idx::AbstractArray{Int32}
    rndm::AbstractArray{Float32}
    δ::AbstractArray{Float32}
end

upres(A, N::Int) = A(Float32(1):Float32(N)) ./ N .-0.5f0 .- (1.0f0 / (2.0f0 * N))

Tracer(A, rays, N) = Tracer(A{Int32}(undef, length(rays)), A{Float32}(undef, length(rays)), upres(A, N))

abstract type AbstractImager end

struct StableImager <: AbstractImager end
