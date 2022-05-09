# RUN FROM /
using Revise, CUDA, LazyArrays

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../cuda.jl")
function main()

	translate(t, v) = STri(t[1], t[2] - v, t[3] - v, t[4] - v, t[5], t[6], t[7])

    obj_path = "objs/sphere.obj"
    tris = mesh_to_STri(load(obj_path))
	map!(t->translate(t, V3(-10, 0,0)), tris, tris)
	tris = vcat(tris, mesh_to_STri(load(obj_path)))
	map!(t->translate(t, V3(5,0, 0,)), tris, tris)

	centroid = _centroid(tris)
	println(centroid)
	println(model_box(tris))
    #tris = parse_obj(obj_path)
    @info "$(length(tris)) triangles"
    width = 512
    height = 512#Int(width * 3 / 4)
    frame_n = 720

	function moving_camera(frame_i, frame_n)
		camera_pos = V3((12, 0, 0)) + centroid
		look_at = zero(V3)
		up = V3((0.0, 0.0, -1.0))
		FOV =  45.0 * pi / 180.0

		return get_camera(camera_pos, look_at, up, FOV)
	end

    depth = 3
    dλ = 30
    ITERS = 1

    skys = [sky_stripes_down]


		R = rotation_matrix(V3(1, 1, 1), 2 * pi * 0 / frame_n)

		tris′ = map(t -> translate(t, -centroid ), tris)
		println(model_box(tris′))
        tris′ = map(t -> map(v -> R * v, t), tris′)
		hit_tris = map(t->Tri(t[1:4]...), tris′)

		images = @time ad_frame_matrix(
            moving_camera,
            width,
            height,
			hit_tris,
            tris′,
            skys,
            dλ,
            depth,
            ITERS,
            0,
            CUDA.rand,
            CuArray,
			false,
			"diffuse/may"
        )

    println("~fin")
end

main()
