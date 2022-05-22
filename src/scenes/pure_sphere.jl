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

	λ_min = 400.0f0
    λ_max = 700.0f0
    depth = 2
    dλ = 25.0f0
    ITERS = 1

    skys = [sky_stripes_down]


	RGB = CuArray{Float32}(undef, width*height, 3)

    tris = CuArray([Sphere(zero(V3), 0.0f0), Sphere(V3(3, 0, 0), 1.0f0), Sphere(V3(-3, 0.4, 0.6), 1.0f0)],)
	n_tris = collect(zip(map(Int32, collect(1:length(tris))), tris)) |> CuArray |> m -> reshape(m, 1, length(m))
    row_indices = CuArray(1:height)
    col_indices = reshape(CuArray(1:width), 1, width)
    rays = CuArray{ADRay}(undef, width * height)
    hit_idx = CuArray(zeros(Int32, length(rays)))
    dv = CuArray{V3}(undef, height) # make w*h
    s0 = CuArray{Float32}(undef, length(rays), 3)

    # Datastruct init
    hits = CuArray{Int32}(undef, (width* height))
    rndm = CUDA.rand(Float32, height * width)

    # use host to compute constants used in turning spectra into colors
    spectrum = collect(λ_min:dλ:λ_max) |> a -> reshape(a, 1, 1, length(a))
    retina_factor = Array{Float32}(undef, 1, 3, length(spectrum))

		@time ad_frame_matrix(
            moving_camera,
			height,
			width,
            dλ,
            depth,
            ITERS,
            0,
            CUDA.rand,
            CuArray,
			false,
			3;
			RGB = RGB,
			n_tris = n_tris,
		    tris = tris,
		    row_indices = row_indices,
		    col_indices = col_indices,
		    rays = rays,
		    hit_idx = hit_idx,
		    dv = dv,
		    s0 = s0,

		    # Datastruct init
		    hits = hits,
		    rndm = rndm,
		    # use host to compute constants used in turning spectra into colors
		    spectrum = spectrum,
		    retina_factor = retina_factor,
        )
		img = RGBf.(map(a -> reshape(Array(a), height, width), (RGB[:, 1], RGB[:, 2], RGB[:, 3]))...)
		title="pure/may/$(lpad(frame_i, 3, "0"))"
        Makie.save("out/$title.png", img)
	end
    println("~fin")
end

@time main()
