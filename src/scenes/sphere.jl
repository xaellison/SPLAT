# RUN FROM /
using Revise
include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")

function main()

    obj_path = "objs/sphere.obj"
    tris = mesh_to_STri(load(obj_path))
	centroid = _centroid(tris)
	println(centroid)
	println(model_box(tris))
    #tris = parse_obj(obj_path)
    @info "$(length(tris)) triangles"
    width = 300
    height = Int(width * 3 / 4)
    frame_n = 60

	function moving_camera(frame_i, frame_n)
		camera_pos = V3((7, 0, 0)) + centroid
		look_at = zero(V3)
		up = V3((0.0, 0.0, -1.0))
		FOV =  45.0 * pi / 180.0

		return get_camera(camera_pos, look_at, up, FOV)
	end

    depth = 3
    dλ = 30
    ITERS = 16

    skys = [sky_stripes_down]
    Threads.@threads for i = 1:frame_n

		R = rotation_matrix(V3(1, 1, 1), 2 * pi * 0 / frame_n)
        #translate(t, v) = STri(t[1], t[2] - v, t[3] - v, t[4] - v, t[5:7]...)
		translate(t, v) = STri(t[1], t[2] - v, t[3] - v, t[4] - v, t[5], t[6], t[7])
		#translate(t, v) = Tri(t[1], t[2] - v, t[3] - v, t[4] - v)

		tris′ = map(t -> translate(t, -centroid), tris)

        tris′ = map(t -> map(v -> R * v, t), tris′)

        images = @time ad_frame_matrix(
            moving_camera,
            width,
            height,
            tris′,
            skys,
            dλ,
            depth,
            ITERS,
            Float32(2 * pi / 20 * i / frame_n),
            rand,
            Array
        )
        for s in keys(skys)
            Makie.save("out/sphere/$(s)/$(lpad(i, 3, "0")).png", images[s])
        end

    end
    println("~fin")
end

main()
