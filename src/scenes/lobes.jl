# RUN FROM /
using Revise, LazyArrays, Parameters, GLMakie, CUDA, KernelAbstractions, Random

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../procedural_assets.jl")

function main()
	axis = ℜ³(0, 1, 0) + ℜ³(0.3, 0, 0)
	out = nothing
	width = 1024
	height = 1024
 	for frame in 1:3
	@sync CUDA.NVTX.@range "frame $frame" begin	# Tracing params
	    dλ = 25f0
	    λ_min = 400.0f0
	    λ_max = 700.0f0
	    depth = 5
		forward_upscale = 4
		backward_upscale = 8
		reclaim_after_iter=true
		iterations_per_frame=1
		# Geometry

		lobe1 = mesh_to_FTri(load("C:/Users/ellis/Documents/Github/mcabbot/SHART/objs/lobe1_2.obj"))
		lobe2 = mesh_to_FTri(load("C:/Users/ellis/Documents/Github/mcabbot/SHART/objs/lobe2_2.obj"))
		R = rotation_matrix(ℜ³(1,0,0), 2 * pi * frame / 180) * rotation_matrix(axis, 2 * pi * 44 / 180) * rotation_matrix(ℜ³(0, 0, 1), pi / 4) * rotation_matrix(ℜ³(1, 0, 0), pi / 2) * rotation_matrix(ℜ³(0, 0, 1), pi / 4)
		lobe1 = map(t -> translate(t, ℜ³(0, 0, -0.25) ), lobe1)
		lobe1 = map(t -> rotate(t, R), lobe1)
		lobe2 = map(t -> translate(t, ℜ³(0, 0, -0.25) ), lobe2)
		lobe2 = map(t -> rotate(t, R), lobe2)
		meshes = [[zero(FTri)], lobe1, lobe2]
		first_diffuse = 1 + length(lobe1) + 1
		tris = foldl(vcat, meshes)
		bounding_volumes, bounding_volumes_members = cluster_fuck(tris, 64)

		Λ = CuArray(collect(λ_min:dλ:λ_max))
		tex_f() = checkered_tex(32, 16, length(λ_min:dλ:λ_max)) .*2#.* 12#CUDA.zeros(Float32, width ÷2, width÷2, length(Λ))

		basic_params = Dict{Symbol, Any}()
		@pack! basic_params = width, height, dλ, λ_min, λ_max, depth, first_diffuse, forward_upscale, backward_upscale, iterations_per_frame, reclaim_after_iter

		# Forward Trace light map

		#ray_generator(x, y, λ, dv) = simple_light(ℜ³(0, 1, 0), ℜ³(0, -1, 0), ℜ³(0, 0, 1) * 0.3, ℜ³(1, 0, 0) * 0.3, height, width, x, y, λ, dv)
		light_size = 1024
		lights = [
			RectLight(ℜ³(1, 0, 0), ℜ³(-1, 0, 0), ℜ³(0, 0, 1) * 0.3, ℜ³(0, 1, 0) * 0.3, light_size, light_size),
			RectLight(ℜ³(-1, 0, 0), ℜ³(1, 0, 0), ℜ³(0, 0, 1) * 0.3, ℜ³(0, 1, 0) * 0.3, light_size, light_size),
			RectLight(ℜ³(0, 0, 1), ℜ³(0, 0, -1), ℜ³(0, 1, 0) * 0.3, ℜ³(1, 0, 0) * 0.3, light_size, light_size),
			RectLight(ℜ³(0, 0, -1), ℜ³(0, 0, 1), ℜ³(0, 1, 0) * 0.3, ℜ³(1, 0, 0) * 0.3, light_size, light_size),
		]

		function my_moving_camera()
	        camera_pos = ℜ³((0, 0.8, 0))
	        look_at = ℜ³(0, 0, 0)
	        up = ℜ³((0.0, 0.0, 1.0))
	        FOV = 45.0 * pi / 180.0

	        return get_camera(camera_pos, look_at, up, FOV)
	    end

	    cam = my_moving_camera()

		trace_kwargs = Dict{Symbol, Any}()
		@pack! trace_kwargs = cam, lights, tex_f, tris, λ_min, dλ, λ_max, bounding_volumes, bounding_volumes_members
		trace_kwargs = merge(basic_params, trace_kwargs)
		@info "start..."
		CUDA.@time RGB = trace!(ExperimentalTracer, ExperimentalHitter2, ExperimentalImager; intensity=1f0, iterations_per_frame=4, trace_kwargs...)

		save("out/lobes/$(lpad(frame, 3, "0")).png", reshape(Array(RGB), (height, width)))
	end
	end
	return nothing
end
#try

@time main()
#finally
#	CUDA.reclaim()
#end
