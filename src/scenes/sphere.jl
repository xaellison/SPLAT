# RUN FROM /
using Revise, LazyArrays, Parameters, GLMakie, CUDA, KernelAbstractions , CUDAKernels

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../procedural_assets.jl")

function main()
	# Tracing params
    width = 1024
    height = 1024
    dλ = 25f0
    λ_min = 400.0f0
    λ_max = 700.0f0
    depth = 3
    ITERS = 1
	upscale = 4
	# Geometry

	obj_path = "objs/icos.obj"
	glass_sphere = mesh_to_FTri(load(obj_path))

	meshes = [[zero(FTri)], glass_sphere, stage()]
	first_diffuse = 1 + 1 + length(glass_sphere)
	tris = foldl(vcat, meshes)


	tex = checkered_tex(32, 32, length(λ_min:dλ:λ_max)) .* 3

	basic_params = Dict{Symbol, Any}()
	@pack! basic_params = width, height, dλ, λ_min, λ_max, depth, first_diffuse, upscale

    function my_moving_camera()
        camera_pos = ℜ³((0, 5, 5))
        look_at = zero(ℜ³)
        up = ℜ³((0.0, 0.0, -1.0))
        FOV = 45.0 * pi / 180.0

        return get_camera(camera_pos, look_at, up, FOV)
    end

    cam = my_moving_camera()

	lights = [
		RectLight(ℜ³(0, 0, 8), ℜ³(0, 0, -1), ℜ³(1, 0, 0), ℜ³(0, 1, 0), height, width),
	]

	trace_kwargs = Dict{Symbol, Any}()
	@pack! trace_kwargs = cam, lights, tex, tris, λ_min, dλ, λ_max
	trace_kwargs = merge(basic_params, trace_kwargs)
	array_kwargs = trace!(ExperimentalTracer,
						  ExperimentalHitter,
						  ExperimentalImager; trace_kwargs...)

#	@unpack RGB, viz_map = array_kwargs
#    RGB = Array(RGB)
    return array_kwargs#reshape(RGB, (height, width)), viz_map
end

CUDA.NVTX.@range "warmup" structs= main()
CUDA.NVTX.@range "real1" RGB = main();
CUDA.NVTX.@range "real2" RGB = main();
#
