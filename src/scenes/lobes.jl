# RUN FROM /
using Revise, LazyArrays, Parameters, GLMakie, CUDA, KernelAbstractions, Random, NVTX

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../procedural_assets.jl")

function main()
	axis = ℜ³(0, 1, 0) + ℜ³(0.3, 0, 0)
	out = nothing
	width = 1024
	height = 1024
 	begin	# Tracing params
	    dλ = 25f0
	    λ_min = 400.0f0
	    λ_max = 700.0f0
	    depth = 5
		forward_upscale = 4
		backward_upscale = 4
		iterations_per_frame=1
		# Geometry
		frame = 1
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

		bounding_volumes, bounding_volumes_members = bv_partition(tris, 6; verbose=true)

		Λ = CuArray(collect(λ_min:dλ:λ_max))
		tex_f() = checkered_tex(32, 16, length(λ_min:dλ:λ_max)) .*2#.* 12#CUDA.zeros(Float32, width ÷2, width÷2, length(Λ))

		basic_params = Dict{Symbol, Any}()
		@pack! basic_params = width, height, dλ, λ_min, λ_max, depth, first_diffuse, forward_upscale, backward_upscale, iterations_per_frame

		# Forward Trace light map

		#ray_generator(x, y, λ, dv) = simple_light(ℜ³(0, 1, 0), ℜ³(0, -1, 0), ℜ³(0, 0, 1) * 0.3, ℜ³(1, 0, 0) * 0.3, height, width, x, y, λ, dv)
		light_size = 1024
		lights = [
			RectLight(ℜ³(1, 0, 0), ℜ³(-1, 0, 0), ℜ³(0, 0, 1) * 0.3, ℜ³(0, 1, 0) * 0.3, light_size, light_size),
		]

		function my_moving_camera()
	        camera_pos = ℜ³((0, 0.8, 0))
	        look_at = ℜ³(0, 0, 0)
	        up = ℜ³((0.0, 0.0, 1.0))
	        FOV = 45.0 * pi / 180.0

	        return get_camera(camera_pos, look_at, up, FOV)
	    end

	    cam = my_moving_camera()
		#forward_hitter = ExperimentalHitter3(CuArray, light_size ^ 2 ÷ (forward_upscale ^ 2) * length(lights))#, tris, bounding_volumes, bounding_volumes_members)
		#backward_hitter = ExperimentalHitter3(CuArray, height * width ÷ (backward_upscale ^ 2))#, tris, bounding_volumes, bounding_volumes_members)
 		
		forward_hitter = DPBVHitter(CuArray, light_size ^ 2 ÷ (forward_upscale ^ 2) * length(lights), tris, bounding_volumes, bounding_volumes_members; concurrency=32)
		backward_hitter = DPBVHitter(CuArray, height * width ÷ (backward_upscale ^ 2), tris, bounding_volumes, bounding_volumes_members; concurrency=32)
 		#forward_hitter = BoundingVolumeHitter(CuArray, light_size ^ 2 ÷ (forward_upscale ^ 2) * length(lights), bounding_volumes, bounding_volumes_members)
		#backward_hitter = BoundingVolumeHitter(CuArray, height * width ÷ (backward_upscale ^ 2), bounding_volumes, bounding_volumes_members)
 
		
		runme(i) = begin
			trace_kwargs = Dict{Symbol, Any}()
			@pack! trace_kwargs = cam, lights, tex_f, tris, λ_min, dλ, λ_max, forward_hitter, backward_hitter
			trace_kwargs = merge(basic_params, trace_kwargs)
			RGB = trace!(StableTracer, ExperimentalImager2; intensity=1.0f0, trace_kwargs...)

			return reshape(RGB, (height, width))
		end


		matrix = Array{RGBf}(undef, height, width)
		fig, ax, hm = image(1:height, 1:width, matrix)


		# we use `record` to show the resulting video in the docs.
		# If one doesn't need to record a video, a normal loop works as well.
		# Just don't forget to call `display(fig)` before the loop
		# and without record, one needs to insert a yield to yield to the render task

		runme(1)
		runme(1)
		hm[3] = runme(1)
		display(fig)

		bvs2, bvms2 = copy(bounding_volumes), copy(bounding_volumes_members)
		@time for i in 1:40

			

		#    events(hm).mouseposition |> println
			recalc_task = Threads.@spawn begin
				tv = @view tris[2:end]
				oscillate(tv) = translate(tv, ℜ³(cos(i / 20) / 500, 0, sin(i / 20) / 500))
				tv .= oscillate.(tv)
				bvs2, bvms2 = bv_partition(tris, 6)
			end 
			
			hm[3] = runme(1) # update data
			wait(recalc_task)
			t1 = Threads.@spawn repack!(forward_hitter, bvs2, bvms2)
			t2 = Threads.@spawn repack!(backward_hitter, bvs2, bvms2)
			yield()
			
			wait(t1)
			wait(t2)
		end
	end
	return nothing
end
#try

@time main()
#finally
#	CUDA.reclaim()
#end
