# RUN FROM /
using Revise
include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")

function main()

    obj_path = "objs/s0.obj"
    tris = mesh_to_Tri(load(obj_path))
	centroid = centroidish(tris)

    #tris = parse_obj(obj_path)
    @info "$(length(tris)) triangles"
    width = 512
    height = Int(width * 3 / 4)
    frame_n = 1

	function moving_camera(frame_i, frame_n)
		camera_pos = V3((4.6, 0, 0)) + centroid
		look_at = centroid
		up = V3((0.0, 0.0, -1.0))
		FOV =  45.0 * pi / 180.0

		return get_camera(camera_pos, look_at, up, FOV)
	end

    depth = 4
    dλ = 30
    ITERS = 16

    skys = Dict("grey" => sky_color, "rezz" => sky_color_rezz)#
    for i = 1:frame_n

        images = @time frame_matrix(
            moving_camera,
            width,
            height,
            tris,
            skys,
            dλ,
            depth,
            ITERS,
            2 * pi / 4 * i / frame_n,
            rand,
            Array
        )
        for s in keys(skys)
            Makie.save("out/bull/$(s)_$i.png", images[s])
        end

    end
    println("~fin")
end

main()
