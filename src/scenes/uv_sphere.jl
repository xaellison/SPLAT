# RUN FROM /
using Revise, LazyArrays, Parameters
# NOTE: for GPU need KernelAbstractions, CUDAKernels for efficient tullio
using CUDA, KernelAbstractions, CUDAKernels
using GLMakie

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")

function scene_parameters()
    width = 256
    height = 256
    xmin = 1
    xmax = height
    ymin = 1
    ymax = width

    dλ = 25.0f0
    λ_min = 400.0f0
    λ_max = 700.0f0

    depth = 2
    ITERS = 1

    x = collect(xmin:xmax)#LinRange(-2, 1, 200)
    y = collect(ymin:ymax)#LinRange(-1.1, 1.1, 200)
    function init(x, y)
        RGBf(rand(), rand(), rand())
    end
    img = init.(x, y')
    #fig, ax, hm = image(x, y, img)
    #display(fig)

    RGB3 = Array{Float32}(undef, width * height, 3)
    RGB = Array{RGBf}(undef, width * height)

    row_indices = Array(1:height)
    col_indices = reshape(Array(1:width), 1, width)
    rays = Array{ADRay}(undef, width * height)
    hit_idx = Array(zeros(Int32, length(rays)))
    dv = Array{V3}(undef, height, width) # make w*h
    s0 = Array{Float32}(undef, length(rays), 3)


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

    retina_factor = Array(retina_factor)
    spectrum = Array(spectrum)

    # Datastruct init
    expansion = Array{FastRay}(undef, length(rays))
    hits = Array{Int32}(undef, size(expansion))
    tmp = Array{Tuple{Float32, Int32}}(undef, size(expansion))
    rndm = rand(Float32, height * width)

    θ = 0.0f0

    θ += 2 * π / 360
    function my_moving_camera(frame_i, frame_n)
        camera_pos = V3((7, 0, 0)) #+ centroid
        look_at = zero(V3)
        up = V3((0.0, 0.0, -1.0))
        FOV = 45.0 * pi / 180.0

        return get_camera(camera_pos, look_at, up, FOV)
    end

    obj_path = "objs/sphere.obj"
	glass_sphere = mesh_to_FTri(load(obj_path))
	map!(t->translate(t, V3(3.0, 0, 0.0)), glass_sphere, glass_sphere)
	V = V3(0, cos(0), sin(0))
	diffuse_sphere = mesh_to_FTri(load(obj_path))
	map!(t->translate(t, V3(-3.0, 0, 0.0) + V), diffuse_sphere, diffuse_sphere)

	meshes = [[zero(FTri)], glass_sphere, diffuse_sphere]

	tris = CuArray(foldl(vcat, meshes))

    n_tris = collect(zip(map(Int32, collect(1:length(tris))), tris)) |>
        m -> reshape(m, 1, length(m))

	tris_per_sphere = length(glass_sphere)

    first_diffuse =  1 + 1 + tris_per_sphere

	tex = rand(Float32, 512, 512)

    sort_optimization = true
    camera_generator = my_moving_camera
    scalar_kwargs = Dict{Symbol, Any}()
    array_kwargs = Dict{Symbol, Any}()
    @pack! scalar_kwargs =  width,
                            height,
                            dλ,
                            depth,
                            ITERS,
                            first_diffuse,
                            sort_optimization,
                            camera_generator

    @pack! array_kwargs = RGB3,
                          RGB,
                          n_tris,
                          tris,
                          row_indices,
                          col_indices,
                          rays,
                          hit_idx,
                          dv,
                          s0,
                          expansion,
                          hits,
                          rndm,
                          tmp,
                          spectrum,
                          retina_factor,
						  tex

    return scalar_kwargs, array_kwargs
end

function main()
    skw, akw = scene_parameters()
	akw = Dict(kv[1]=>CuArray(kv[2]) for kv in akw)
    ad_frame_matrix(;skw..., akw...)
    @unpack RGB = akw
    @unpack height, width = skw
    return reshape(Array(RGB), (height,width))
end

CUDA.@time RGB= main()
image(RGB)
