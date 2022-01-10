# RUN FROM /
using Revise

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../cuda.jl")
function main()

    obj_path = "objs/tiger_head.obj"
    tris = mesh_to_STri(load(obj_path))

	centroid = _centroid(tris)
	println(centroid)
	println(model_box(tris))
    #tris = parse_obj(obj_path)
    @info "$(length(tris)) triangles"
    width = 256
    height = 256#Int(width * 3 / 4)
    frame_n = 720

	function moving_camera(frame_i, frame_n)
		camera_pos = V3((30, 0, 0))
		look_at = zero(V3)
		up = V3((0.0, 0.0, -1.0))
		FOV =  45.0 * pi / 180.0

		return get_camera(camera_pos, look_at, up, FOV)
	end

    depth = 3
    dλ = 10
    ITERS = 1

    skys = [sky_curtains]

    #for i = 1:frame_n
		# R0 gets corner up
	#	R0 = rotation_matrix(cross(V3(1,1,0), V3(1, 1, 1)), pi/2-acos(vector_cosine(V3(1,1,1),V3(1,1,0))))
		R = rotation_matrix(V3(0, 1, 0),  -pi / 6 ) * rotation_matrix(V3(0, 0, 1),  -pi / 2 )
        #translate(t, v) = STri(t[1], t[2] - v, t[3] - v, t[4] - v, t[5:7]...)
		translate(t::STri, v) = STri(t[1], t[2] - v, t[3] - v, t[4] - v, t[5], t[6], t[7])
		translate(t::Tri, v) = Tri(t[1], t[2] - v, t[3] - v, t[4] - v)

		tris′ = map(t -> translate(t, -centroid + V3(0,0,-3)), tris)

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
            0,
            CUDA.rand,
            CuArray
        )

    println("~fin")
end

main()
