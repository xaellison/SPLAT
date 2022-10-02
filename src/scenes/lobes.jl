# RUN FROM /
using Revise, LazyArrays, Parameters, GLMakie, CUDA, KernelAbstractions, CUDAKernels, Random

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../procedural_assets.jl")

function main()
	for frame in (collect(1:360))
	R = rotation_matrix(ℜ³(0, 0, 1), 2 * pi * (frame - 40) / 360)
	# Tracing params
    width = 2048
    height = 2048
    dλ = 25f0
    λ_min = 400.0f0
    λ_max = 700.0f0
    depth = 5
	forward_upscale = 2
	backward_upscale = 2
	# Geometry

	lobe1 = mesh_to_FTri(load("objs/lobe1.obj"))
	lobe2 = mesh_to_FTri(load("objs/lobe2_fix.obj"))
	lobe1 = map(t -> translate(t, ℜ³(0, 0, -0.25) ), lobe1)
	lobe1 = map(t -> rotate(t, R), lobe1)
	lobe2 = map(t -> translate(t, ℜ³(0, 0, -0.25) ), lobe2)
	lobe2 = map(t -> rotate(t, R), lobe2)
	meshes = [[zero(FTri)], lobe1, lobe2]
	first_diffuse = 1 + length(lobe1) + 1
	tris = foldl(vcat, meshes)

	Λ = CuArray(collect(λ_min:dλ:λ_max))
	tex = CUDA.zeros(Float32, 1024, 1024, length(Λ))

	basic_params = Dict{Symbol, Any}()
	@pack! basic_params = width, height, dλ, λ_min, λ_max, depth, first_diffuse, forward_upscale, backward_upscale

	# Forward Trace light map

	#ray_generator(x, y, λ, dv) = simple_light(ℜ³(0, 1, 0), ℜ³(0, -1, 0), ℜ³(0, 0, 1) * 0.3, ℜ³(1, 0, 0) * 0.3, height, width, x, y, λ, dv)

	lights = [
		RectLight(ℜ³(1, 0, 0), ℜ³(-1, 0, 0), ℜ³(0, 0, 1) * 0.3, ℜ³(0, 1, 0) * 0.3, 2048, 2048),
		RectLight(ℜ³(-1, 0, 0), ℜ³(1, 0, 0), ℜ³(0, 0, 1) * 0.3, ℜ³(0, 1, 0) * 0.3, 2048, 2048),
		RectLight(ℜ³(0, 0, 1), ℜ³(0, 0, -1), ℜ³(0, 1, 0) * 0.3, ℜ³(1, 0, 0) * 0.3, 2048, 2048),
		RectLight(ℜ³(0, 0, -1), ℜ³(0, 0, 1), ℜ³(0, 1, 0) * 0.3, ℜ³(1, 0, 0) * 0.3, 2048, 2048),
	]

	function my_moving_camera()
        camera_pos = ℜ³((0, 1, 0))
        look_at = ℜ³(0, 0, 0)
        up = ℜ³((0.0, 0.0, 1.0))
        FOV = 45.0 * pi / 180.0

        return get_camera(camera_pos, look_at, up, FOV)
    end

    cam = my_moving_camera()

	trace_kwargs = Dict{Symbol, Any}()
	@pack! trace_kwargs = cam, lights, tex, tris, λ_min, dλ, λ_max
	trace_kwargs = merge(basic_params, trace_kwargs)

	array_kwargs = trace!(ExperimentalTracer, ExperimentalHitter, ExperimentalImager; trace_kwargs...)

	@unpack RGB = array_kwargs
    RGB = Array(RGB)
    save("out/lobes/$(lpad(frame, 3, "0")).png", reshape(RGB, (height, width)))
	#break
	end
end
@time main()
