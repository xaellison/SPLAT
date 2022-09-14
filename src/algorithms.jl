abstract type AbstractHitter end
abstract type AbstractForwardTracer end
abstract type AbstractBackwardTracer end

struct StableHitter <: AbstractHitter
end

struct ExperimentalHitter <: AbstractHitter
end

struct ForwardContinuumTracer <: AbstractForwardTracer
    width
    height
    dλ
    λ_min
    λ_max
    depth
    sort_optimization
    first_diffuse
    datastructs :: Dict{Symbol, Any}
end

ForwardContinuumTracer(width, height, dλ, λ_min, λ_max, depth, sort_optimization, first_diffuse) = begin
    basic_params = Dict{Symbol, Any}()
	@pack! basic_params = width, height, dλ, λ_min, λ_max, depth, sort_optimization, first_diffuse
	datastructs = scene_datastructs(CuArray; basic_params...)
	return ForwardContinuumTracer(width, height, dλ, λ_min, λ_max, depth, sort_optimization, first_diffuse, datastructs)
end

struct ForwardExpansionTracer <: AbstractForwardTracer
    datastructs :: Dict{Symbol, Any}
end

struct BackwardContinuumTracer <: AbstractBackwardTracer
    datastructs :: Dict{Symbol, Any}
end

struct BackwardExpansionTracer <: AbstractBackwardTracer
    datastructs :: Dict{Symbol, Any}
end

# G is geometry type, sphere or tri
struct Renderer{G, H, T1 <: AbstractForwardTracer, T2 <: AbstractBackwardTracer}
    hitter :: H
    forward :: T1
    backward :: T2
end


function run(renderer)
    CUDA.@time run_evolution!(renderer.hitter; basic_params..., array_kwargs...)
    CUDA.@time continuum_light_map!(;basic_params..., array_kwargs...)

end
