# RUN FROM /
using Revise, LazyArrays, Parameters, GLMakie, CUDA, KernelAbstractions, CUDAKernels, Random, NVTX

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../procedural_assets.jl")

function main()
	axis = ℜ³(0, 1, 0) + ℜ³(0.3, 0, 0)
	out = nothing
	width = 1024
	height = 1024
	frame_N = 180
 	for frame in 1:frame_N
	@sync NVTX.@range "frame $frame" begin	# Tracing params
	    dλ = 25f0
	    λ_min = 400.0f0
	    λ_max = 700.0f0
	    depth = 3
		forward_upscale = 4
		backward_upscale = 4
		iterations_per_frame=1
		intensity = 2
		# Geometry

		rips = mesh_to_FTri(load("objs/splash_scene_liquid.obj"))

		arty = mesh_to_FTri(load("objs/splash_scene_artemis.obj"))


		meshes = [[zero(FTri)], rips, arty]
		first_diffuse = 1 + 1 + length(rips)
		tris = foldl(vcat, meshes)

		R = rotation_matrix(ℜ³(0,1,0), 2 * pi * (frame / frame_N))
		tris = map(t -> rotate(t, R), tris)

		#bounding_volumes, bounding_volumes_members = cluster_fuck(tris, 64)

		@time bounding_volumes, bounding_volumes_members = bv_partition(tris, 6; verbose=true)
		
		tex_f() = checkered_tex(32, 16, length(λ_min:dλ:λ_max)) .*0#.* 12#CUDA.zeros(Float32, width ÷2, width÷2, length(Λ))

		basic_params = Dict{Symbol, Any}()
		@pack! basic_params = width, height, dλ, λ_min, λ_max, depth, first_diffuse, forward_upscale, backward_upscale, iterations_per_frame, intensity

		# Forward Trace light map

		function my_moving_camera()
			camera_pos = ℜ³((0, 0, -60))
			look_at = ℜ³(0, 0, 0)
			up = ℜ³((0.0, -1.0, 0.0))
			FOV = 75.0 * pi / 180.0
			return get_camera(camera_pos, look_at, up, FOV)
		end

		cam = my_moving_camera()

		#ray_generator(x, y, λ, dv) = simple_light(ℜ³(0, 1, 0), ℜ³(0, -1, 0), ℜ³(0, 0, 1) * 0.3, ℜ³(1, 0, 0) * 0.3, height, width, x, y, λ, dv)
		light_size = 512

		lights = [
			RectLight(ℜ³(-50, 50, 0), ℜ³(1, -1, 0), ℜ³(14, 14, 0), ℜ³(0, 0, 20), light_size, light_size),
			RectLight(ℜ³(50, 50, 0), ℜ³(-1, -1, 0), ℜ³(14, -14, 0), ℜ³(0, 0, 20), light_size, light_size),
		]
#		forward_hitter = DPBVHitter(CuArray, light_size ^ 2 ÷ (forward_upscale ^ 2) * length(lights), tris, bounding_volumes, bounding_volumes_members)
#		backward_hitter = DPBVHitter(CuArray, height * width ÷ (backward_upscale ^ 2), tris, bounding_volumes, bounding_volumes_members)
 
		forward_hitter = BoundingVolumeHitter(CuArray, light_size ^ 2 ÷ (forward_upscale ^ 2) * length(lights), bounding_volumes, bounding_volumes_members; concurrency=4)
		backward_hitter = BoundingVolumeHitter(CuArray, height * width ÷ (backward_upscale ^ 2), bounding_volumes, bounding_volumes_members; concurrency=4)

		trace_kwargs = Dict{Symbol, Any}()
		
		@pack! trace_kwargs = cam, lights, tex_f, tris, λ_min, dλ, λ_max, forward_hitter, backward_hitter
		trace_kwargs = merge(basic_params, trace_kwargs)
		@info "start..."
		
		CUDA.@time RGB = trace!(ExperimentalTracer, ExperimentalImager2; intensity=1f0, iterations_per_frame=4, force_rand=1.0f0, trace_kwargs...)

		save("out/splash/$(lpad(frame, 3, "0")).png", permutedims(reshape(Array(RGB), (height, width)), (2,1)))
	end
	end
	return nothing
end
#try

@time main()
#finally
#	CUDA.reclaim()
#end
