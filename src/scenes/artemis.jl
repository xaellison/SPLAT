# RUN FROM /
using Revise, LazyArrays, Parameters, GLMakie, CUDA, KernelAbstractions, CUDAKernels, Random

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../procedural_assets.jl")

function main()
	axis = ℜ³(0, 1, 0) + ℜ³(0.3, 0, 0)
	out = nothing
	width = 1024
	height = 1024
 	for frame in 1:1
	@sync CUDA.NVTX.@range "frame $frame" begin	# Tracing params
	    dλ = 25f0
	    λ_min = 400.0f0
	    λ_max = 700.0f0
	    depth = 2
		forward_upscale = 4
		backward_upscale = 4
		reclaim_after_iter=true
		iterations_per_frame=4
		# Geometry
		rips = ripples(frame / 100)
		rips = map(t -> translate(t, ℜ³(-0.5, -0.5, 0) ), rips)
		rips = map(t -> scale(t, 50), rips)
		rips = map(t -> translate(t, ℜ³(0, 25, 70) ), rips)

		arty = mesh_to_FTri(load("objs/artemis_smaller.obj"))
		meshes = [[zero(FTri)], rips, arty]
		first_diffuse = 1 + length(rips) + 1
		tris = foldl(vcat, meshes)

		tex_f() = checkered_tex(32, 16, length(λ_min:dλ:λ_max)) .*1#.* 12#CUDA.zeros(Float32, width ÷2, width÷2, length(Λ))

		basic_params = Dict{Symbol, Any}()
		@pack! basic_params = width, height, dλ, λ_min, λ_max, depth, first_diffuse, forward_upscale, backward_upscale, iterations_per_frame, reclaim_after_iter

		# Forward Trace light map

		#ray_generator(x, y, λ, dv) = simple_light(ℜ³(0, 1, 0), ℜ³(0, -1, 0), ℜ³(0, 0, 1) * 0.3, ℜ³(1, 0, 0) * 0.3, height, width, x, y, λ, dv)
		light_size = 1024
		lights = [
			RectLight(ℜ³(0, 25, 100), ℜ³(0, 0, -1), ℜ³(0, 25, 0) , ℜ³(25, 0, 0), light_size, light_size),
		]

		function my_moving_camera()
	        camera_pos = ℜ³((0, 25, 65))
	        look_at = ℜ³(0, 25, 0)
	        up = ℜ³((0.0, -1.0, 0.0))
	        FOV = 75.0 * pi / 180.0
	        return get_camera(camera_pos, look_at, up, FOV)
	    end

	    cam = my_moving_camera()

		trace_kwargs = Dict{Symbol, Any}()
		@pack! trace_kwargs = cam, lights, tex_f, tris, λ_min, dλ, λ_max
		trace_kwargs = merge(basic_params, trace_kwargs)
		@info "start..."
		CUDA.@time RGB = trace!(ExperimentalTracer, ExperimentalHitter2, ExperimentalImager; intensity=1f0, iterations_per_frame=4, force_rand=1.0f0, trace_kwargs...)

		save("out/arty/$(lpad(frame, 3, "0")).png", permutedims(reshape(Array(RGB), (height, width)), (2,1)))
	end
	end
	return nothing
end
#try

@time main()
#finally
#	CUDA.reclaim()
#end
