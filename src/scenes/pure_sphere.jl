# RUN FROM /
using Revise, CUDA, LazyArrays

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../cuda.jl")
function main()
	frame_n = 5
	for frame_i in 1:frame_n

    width = 1280
    height = 720#Int(width * 3 / 4)


	function moving_camera(frame_i, frame_n)
		camera_pos = V3((7, 0, 0)) #+ centroid
		look_at = zero(V3)
		up = V3((0.0, 0.0, -1.0))
		FOV =  45.0 * pi / 180.0

		return get_camera(camera_pos, look_at, up, FOV)
	end

    depth = 2
    dλ = 25.0f0
    ITERS = 1

    skys = [sky_stripes_down]



		images = @time ad_frame_matrix(
            moving_camera,
            width,
            height,
			#hit_tris,
            [Sphere(zero(V3), 0.0f0), Sphere(V3(3, 0, 0), 1.0f0), Sphere(V3(-3, 0.4, 0.6), 1.0f0)],
            dλ,
            depth,
            ITERS,
            0,
            CUDA.rand,
            CuArray,
			false,
			"pure/may/$(lpad(frame_i, 3, "0"))",
			3,
        )
	end
    println("~fin")
end

@time main()
