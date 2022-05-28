

# RUN FROM /
using Revise, CUDA, LazyArrays#, GLMakie
import CUDA.NVTX.@range
include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../cuda.jl")
function main()

    width = 896
    height = 896
    xmin = 1
    xmax = height
    ymin = 1
    ymax = width


	translate(t :: Tri, v) = Tri(t[1], t[2] + v, t[3] + v, t[4] + v, t[5], t[6], t[7])
	translate(t :: STri, v) = STri(t[1], t[2] + v, t[3] + v, t[4] + v, t[5], t[6], t[7])
	translate(t :: FTri, v) = FTri(t[1], t[2] + v, t[3] + v, t[4] + v, t[5], t[6], t[7], t[8], t[9], t[10])

    obj_path = "objs/uvs.obj"
	tris = mesh_to_FTri(load(obj_path))
	c = _centroid(tris)
	map!(t->translate(t, V3(3.0, 0, 0.0) - c), tris, tris)
	V = V3(0, cos(2 * pi ), sin(2 * pi )) * 0.6
	meshes = [tris]
	tris = mesh_to_FTri(load(obj_path))
	c = _centroid(tris)
	map!(t->translate(t, V3(-3.0, 0, 0.0) + V - c), tris, tris)
	push!(meshes, tris)
	tris = foldl(vcat, meshes)
	n_tris = collect(zip(map(Int32, collect(1:length(tris))), tris)) |>
		m -> reshape(m, 1, length(m))



    x = collect(xmin:xmax)#LinRange(-2, 1, 200)
    y = collect(ymin:ymax)#LinRange(-1.1, 1.1, 200)
    function init(x, y)
        RGBf(rand(), rand(), rand())
    end
    img = init.(x, y')
    #fig, ax, hm = image(x, y, img)
    #display(fig)
    dλ = 25.0f0
    λ_min = 400.0f0
    λ_max = 700.0f0
    RGB3 = CuArray{Float32}(undef, width * height, 3)
    RGB = CuArray{RGBf}(undef, width * height)

    row_indices = CuArray(1:height)
    col_indices = reshape(CuArray(1:width), 1, width)
    rays = CuArray{ADRay}(undef, width * height)
    hit_idx = CuArray(zeros(Int32, length(rays)))
    dv = CuArray{V3}(undef, height) # make w*h
    s0 = CuArray{Float32}(undef, length(rays), 3)


    # use host to compute constants used in turning spectra into colors
    spectrum = collect(λ_min:dλ:λ_max) |> a -> reshape(a, 1, 1, length(a))
    retina_factor = Array{Float32}(undef, 1, 3, length(spectrum))
    map!(retina_red, begin
        @view retina_factor[1, 1, :]
    end, spectrum)
    map!(retina_green, begin
        @view retina_factor[1, 2, :]
    end, spectrum)
    map!(retina_blue, begin
        @view retina_factor[1, 3, :]
    end, spectrum)

    retina_factor = CuArray(retina_factor)
    spectrum = CuArray(spectrum)

    # Datastruct init
    expansion = CuArray{FastRay}(undef, (length(rays), 1, length(spectrum)))
    hits = CuArray{Int32}(undef, size(expansion))
    tmp = CuArray{Tuple{Float32, Int32}}(undef, size(expansion))
    rndm = CUDA.rand(Float32, height * width)
    θ = 0.0f0
    host_RGB = nothing
    for ffff in 1:4
    @time begin
        θ += 2 * π / 360
        function moving_camera(frame_i, frame_n)
            camera_pos = V3((7, 0, 0)) #+ centroid
            look_at = zero(V3)
            up = V3((0.0, 0.0, -1.0))
            FOV = 45.0 * pi / 180.0

            return get_camera(camera_pos, look_at, up, FOV)
        end

        tris = CuArray(tris)
        n_tris = CuArray(n_tris)
        depth = 2
        ITERS = 1

        skys = [sky_stripes_down]
        ad_frame_matrix(
            moving_camera,
            height,
            width,
            dλ,
            depth,
            ITERS,
            0,
            CUDA.rand,
            false,
            3;
            RGB3 = RGB3,
            RGB=RGB,
            n_tris = n_tris,
            tris = tris,
            row_indices = row_indices,
            col_indices = col_indices,
            rays = rays,
            hit_idx = hit_idx,
            dv = dv,
            s0 = s0,

            # Datastruct init
            expansion = expansion,
            hits = hits,
            rndm = rndm,
            tmp=tmp,
            # use host to compute constants used in turning spectra into colors
            spectrum = spectrum,
            retina_factor = retina_factor,
        )
        if isnothing(host_RGB)
            host_RGB = Array(RGB)
        else
            copyto!(host_RGB, RGB)
        end
        #hm[3] = reshape(host_RGB, (height,width))
    end
    #    yield()
        #title = "pure/may/$(lpad(frame_i, 3, "0"))"
        #Makie.save("out/$title.png", img)
    end
    println("~fin")
end

@time main()
