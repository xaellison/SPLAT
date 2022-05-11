# RUN FROM /
using Revise, CUDA, LazyArrays

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../cuda.jl")
function main()
	frame_n = 120
	for frame_i in 1:frame_n
	translate(t :: Tri, v) = Tri(t[1], t[2] + v, t[3] + v, t[4] + v, t[5], t[6], t[7])
	translate(t :: STri, v) = STri(t[1], t[2] + v, t[3] + v, t[4] + v, t[5], t[6], t[7])
	translate(t :: FTri, v) = FTri(t[1], t[2] + v, t[3] + v, t[4] + v, t[5], t[6], t[7], t[8], t[9], t[10])

    obj_path = "objs/uvs.obj"
	tris = mesh_to_FTri(load(obj_path))
	c = _centroid(tris)
	map!(t->translate(t, V3(3.0, 0, 0.0) - c), tris, tris)
	V = V3(0, cos(2 * pi * frame_i / frame_n), sin(2 * pi * frame_i / frame_n)) * 0.6
	meshes = [tris]
	tris = mesh_to_FTri(load(obj_path))
	c = _centroid(tris)
	map!(t->translate(t, V3(-3.0, 0, 0.0) + V - c), tris, tris)
	push!(meshes, tris)
	tris = foldl(vcat, meshes)

	centroid = _centroid(tris)
	println(centroid)
	println(model_box(tris))
    #tris = parse_obj(obj_path)
    @info "$(length(tris)) triangles"
    width = 1024
    height = 1024#Int(width * 3 / 4)


	function moving_camera(frame_i, frame_n)
		camera_pos = V3((7, 0, 0)) #+ centroid
		look_at = zero(V3)
		up = V3((0.0, 0.0, -1.0))
		FOV =  45.0 * pi / 180.0

		return get_camera(camera_pos, look_at, up, FOV)
	end

    depth = 3
    dλ = 25.0f0
    ITERS = 1

    skys = [sky_stripes_down]


		R = rotation_matrix(V3(1, 1, 1), 2 * pi * 0 / frame_n)

		tris′ = map(t -> translate(t, V3(0,0,0) ), tris)
		println(model_box(tris′))
        #tris′ = map(t -> map(v -> R * v, t), tris′)
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
			"diffuse/may/$(lpad(frame_i, 3, "0"))",
			length(mesh_to_FTri(load(obj_path))) + 1
        )
	end
    println("~fin")
end

@time main()
