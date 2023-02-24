# RUN FROM /
using Revise, LazyArrays, Parameters, CairoMakie, CUDA, KernelAbstractions , CUDAKernels, NVTX

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../procedural_assets.jl")

function main()
	# Tracing params
    width = 1024
    height = 1024
    dλ = 25.0f0
    λ_min = 400.0f0
    λ_max = 700.0f0
    depth = 3
    ITERS = 1
	forward_upscale = 8
	backward_upscale = 4
	# Geometry

	obj_path = "objs/icos.obj"
	
	glass_sphere = mesh_to_FTri(load(obj_path))

	meshes = [[zero(FTri)], glass_sphere, stage()]
	first_diffuse = 1 + 1 + length(glass_sphere)
	tris = CuArray(foldl(vcat, meshes))
	

	tex_f() = checkered_tex(48, 16, length(λ_min:dλ:λ_max)) .* 12

	basic_params = Dict{Symbol, Any}()
	@pack! basic_params = width, height, dλ, λ_min, λ_max, depth, first_diffuse, forward_upscale, backward_upscale

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

	bounding_volumes, bounding_volumes_members = cluster_fuck(Array(tris), 2)
	forward_hitter = DPBVHitter(CuArray, height * width ÷ (forward_upscale ^ 2) * length(lights), tris, bounding_volumes, bounding_volumes_members; concurrency=2)
    backward_hitter = DPBVHitter(CuArray, height * width ÷ (backward_upscale ^ 2), tris, bounding_volumes, bounding_volumes_members; concurrency=2)
    
	#forward_hitter = ExperimentalHitter3(CuArray, height * width ÷ (forward_upscale ^ 2) * length(lights))#, tris, bounding_volumes, bounding_volumes_members)
    #backward_hitter = ExperimentalHitter3(CuArray, height * width ÷ (backward_upscale ^ 2))#, tris, bounding_volumes, bounding_volumes_members)
    
	#forward_hitter = BoundingVolumeHitter(CuArray, height * width ÷ (forward_upscale ^ 2) * length(lights), bounding_volumes, bounding_volumes_members)
	#backward_hitter = BoundingVolumeHitter(CuArray, height * width ÷ (backward_upscale ^ 2), bounding_volumes, bounding_volumes_members)
 

	runme(i) = begin
		trace_kwargs = Dict{Symbol, Any}()
		@pack! trace_kwargs = cam, lights, tex_f, tris, λ_min, dλ, λ_max, forward_hitter, backward_hitter
		trace_kwargs = merge(basic_params, trace_kwargs)
		RGB = trace!(StableTracer, ExperimentalImager2; intensity=1.0f0, force_rand=1f0, trace_kwargs...)

		return reshape(RGB, (height, width))
	end


	matrix = Array{RGBf}(undef, height, width)
	fig, ax, hm = image(1:height, 1:width, matrix)


	# we use `record` to show the resulting video in the docs.
	# If one doesn't need to record a video, a normal loop works as well.
	# Just don't forget to call `display(fig)` before the loop
	# and without record, one needs to insert a yield to yield to the render task

	    

	if true
		# For nvvprof:
		NVTX.@range "warmup" runme(1)
		NVTX.@range "run 1" runme(1)
		NVTX.@range "run 2" runme(1)
	else
		runme(1)
		runme(1)
		hm[3] = runme(1)
		display(fig)
		@time for i in 1:40
		#    events(hm).mouseposition |> println
			tv = @view tris[2:first_diffuse-1]
		    hm[3] = runme(1) # update data
			#oscillate(tv) = translate(tv, ℜ³(cos(i / 20) / 50, 0, sin(i / 20) / 50))
			#tv .= oscillate.(tv)
		    yield()
		end
	end
end
CUDA.@profile main()
