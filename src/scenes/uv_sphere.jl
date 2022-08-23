# RUN FROM /
using Revise, LazyArrays, Parameters
# NOTE: for GPU need KernelAbstractions, CUDAKernels for efficient tullio
using CUDA, KernelAbstractions, CUDAKernels
using GLMakie

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../utils.jl")

function scene_parameters()
	width = 1024
    height = 1024
    dλ = 5.0f0
    λ_min = 400.0f0
    λ_max = 700.0f0
    depth = 3
    ITERS = 1

	basic_params = Dict{Symbol, Any}()
	@pack! basic_params = width, height, dλ, λ_min, λ_max, depth

	datastructs = scene_datastructs(;basic_params...)

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

	tris = foldl(vcat, meshes)

    n_tris = collect(zip(map(Int32, collect(1:length(tris))), tris)) |>
        m -> reshape(m, 1, length(m))

	tris_per_sphere = length(glass_sphere)

    first_diffuse =  1 + 1 + tris_per_sphere

	tex = rand(Float32, 64, 64)

    sort_optimization = false

	cam = my_moving_camera(1, 1)
	ray_generator(x, y, λ, dv) = camera_ray(cam, height, width, x, y, λ, dv)
    scalar_kwargs = Dict{Symbol, Any}()
    array_kwargs = Dict{Symbol, Any}()
    @pack! scalar_kwargs =  first_diffuse,
                            sort_optimization,
                            ray_generator

	scalar_kwargs = merge(scalar_kwargs, basic_params)

    @pack! array_kwargs = tex, tris, n_tris
	array_kwargs = merge(array_kwargs, datastructs)

    return scalar_kwargs, array_kwargs
end

function main()
    skw, akw = scene_parameters()
	akw = Dict(kv[1]=>CuArray(kv[2]) for kv in akw)
    CUDA.@time ad_frame_matrix(;skw..., akw...)
    @unpack RGB = akw
    @unpack height, width = skw
    return reshape(Array(RGB), (height,width))
end
main()
RGB = main()
image(RGB)
