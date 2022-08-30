# RUN FROM /
using Revise, LazyArrays, Parameters, GLMakie, CUDA, KernelAbstractions , CUDAKernels

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../utils.jl")
include("../procedural_assets.jl")

function main()
	# Tracing params
    width = 1024
    height = 1024
    dλ = 12.50f0
    λ_min = 400.0f0
    λ_max = 700.0f0
    depth = 3
    ITERS = 1
    sort_optimization = false

	# Geometry

	obj_path = "objs/icos.obj"
	glass_sphere = mesh_to_FTri(load(obj_path))

	meshes = [[zero(FTri)], glass_sphere, stage()]
	first_diffuse = 1 + 1 + length(glass_sphere)
	tris = foldl(vcat, meshes)

    n_tris = collect(zip(map(Int32, collect(1:length(tris))), tris)) |>
        m -> reshape(m, 1, length(m))

	Λ = CuArray(collect(λ_min:dλ:λ_max))
	tex = checkered_tex(16, 16, length(Λ)) * 20

	basic_params = Dict{Symbol, Any}()
	@pack! basic_params = width, height, dλ, λ_min, λ_max, depth, sort_optimization, first_diffuse

	datastructs = scene_datastructs(CuArray; basic_params...)

	# Forward Trace light map

    function my_moving_camera()
        camera_pos = V3((0, 5, 5))
        look_at = zero(V3)
        up = V3((0.0, 0.0, -1.0))
        FOV = 45.0 * pi / 180.0

        return get_camera(camera_pos, look_at, up, FOV)
    end

    cam = my_moving_camera()
	ray_generator(x, y, λ, dv) = simple_light(V3(0, 0, 8), V3(0, 0, -1), V3(1, 0, 0), V3(0, 1, 0), height, width, x, y, λ, dv)

	rays = wrap_ray_gen(ray_generator; datastructs...)
    array_kwargs = Dict{Symbol, Any}()

    @pack! array_kwargs = tex, tris, n_tris, rays
	array_kwargs = merge(array_kwargs, datastructs)

    array_kwargs = Dict(kv[1]=>CuArray(kv[2]) for kv in array_kwargs)
    CUDA.@time run_evolution!(;basic_params..., array_kwargs...)

	CUDA.@time expansion_light_map!(;basic_params..., array_kwargs...)

	# reverse trace image
	@unpack RGB3 = array_kwargs
	RGB3 .= 0
	ray_generator2(x, y, λ, dv) = camera_ray(cam, height, width, x, y, λ, dv)
	rays = wrap_ray_gen(ray_generator2; datastructs...)
    array_kwargs = Dict{Symbol, Any}()

    @pack! array_kwargs = tex, tris, n_tris, rays
	array_kwargs = merge(array_kwargs, datastructs)

    array_kwargs = Dict(kv[1]=>CuArray(kv[2]) for kv in array_kwargs)
    (;basic_params..., array_kwargs...)

	CUDA.@time run_evolution!(;basic_params..., array_kwargs...)
	CUDA.@time expansion_shade!(;basic_params..., array_kwargs...)

	# return image

	@unpack RGB = array_kwargs
    RGB = Array(RGB)
    return reshape(RGB, (height, width))
end
@time RGB = main()
image(RGB)
