# RUN FROM /
using Revise, LazyArrays, Parameters, GLMakie, CUDA, KernelAbstractions , CUDAKernels

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../utils.jl")

function main()
	# Tracing params
    width = 512
    height = 512
    dλ = 12.50f0
    λ_min = 400.0f0
    λ_max = 700.0f0
    depth = 3
    ITERS = 1
    sort_optimization = false

	# Geometry
	θ = 0.0f0
    θ += 2 * π / 360

	function stage()
		z_floor = -2
		a = V3(3, 1, z_floor)
		b = V3(-3, 1, z_floor)
		c = V3(0, -2, z_floor)
		d = V3(0, -2, 2)
		t_a = V3(0, 0, 0)
		t_b = V3(1, 0, 0)
		t_c = V3(0.5, 0.5, 0)
		t_d = V3(0.5, 1, 0)
		n1 = normalize(cross(a - b, a - c))
		T1 = FTri(n1, a, b, c, n1, n1, n1, t_a, t_b, t_c)
		n2 = normalize(cross(d - b, d - c))
		T2 = FTri(n2, b, c, d, n2, n2, n2, t_b, t_c, t_d)
		n3 = normalize(cross(d - a, d - c))
		T3 = FTri(n3, a, d, c, n3, n3, n3, t_a, t_d, t_c)
		return [T1, T2, T3]
	end



	obj_path = "objs/icos.obj"
	glass_sphere = mesh_to_FTri(load(obj_path))

	meshes = [[zero(FTri)], glass_sphere, stage()]
	first_diffuse = 1 + 1 + length(glass_sphere)
	tris = foldl(vcat, meshes)

    n_tris = collect(zip(map(Int32, collect(1:length(tris))), tris)) |>
        m -> reshape(m, 1, length(m))

	x = repeat(vcat(repeat([1.0f0], 16), repeat([0.0f0], 16)), 16) |> CuArray
	y = reshape(x, 1, length(x)) |> CuArray
	f(x, y, λ) = (x + y) % 2
	Λ = CuArray(collect(λ_min:dλ:λ_max))
	tex = f.(x, y, reshape(Λ, 1, 1, length(Λ))) * 20
	#tex = CUDA.zeros(Float32, 1024, 1024, length(Λ))



	basic_params = Dict{Symbol, Any}()
	@pack! basic_params = width, height, dλ, λ_min, λ_max, depth, sort_optimization, first_diffuse

	@time datastructs = scene_datastructs(CuArray; basic_params...)

	# Forward Trace light map

    function my_moving_camera(frame_i, frame_n)
        camera_pos = V3((0, 5, 5)) #+ centroid
        look_at = zero(V3)
        up = V3((0.0, 0.0, -1.0))
        FOV = 45.0 * pi / 180.0

        return get_camera(camera_pos, look_at, up, FOV)
    end

    cam = my_moving_camera(1, 1)

	#ray_generator(x, y, λ, dv) = simple_light(V3(7, 7, 0), V3(-1/sqrt(2), -1/sqrt(2), 0), V3(1 / sqrt(2), -1/sqrt(2), 0), V3(0, 0, 1), height, width, x, y, λ, dv)
	ray_generator(x, y, λ, dv) = simple_light(V3(0, 0, 8), V3(0, 0, -1), V3(1, 0, 0), V3(0, 1, 0), height, width, x, y, λ, dv)

	rays = wrap_ray_gen(ray_generator; datastructs...)
    array_kwargs = Dict{Symbol, Any}()

    @pack! array_kwargs = tex, tris, n_tris, rays
	array_kwargs = merge(array_kwargs, datastructs)

    array_kwargs = Dict(kv[1]=>CuArray(kv[2]) for kv in array_kwargs)
    run_evolution(;basic_params..., array_kwargs...)

	spectral_light_map!(;basic_params..., array_kwargs...)

	# reverse trace image
	@unpack RGB3 = array_kwargs
	RGB3.=0
	ray_generator2(x, y, λ, dv) = camera_ray(cam, height, width, x, y, λ, dv)
	rays = wrap_ray_gen(ray_generator2; datastructs...)
    array_kwargs = Dict{Symbol, Any}()

    @pack! array_kwargs = tex, tris, n_tris, rays
	array_kwargs = merge(array_kwargs, datastructs)

    array_kwargs = Dict(kv[1]=>CuArray(kv[2]) for kv in array_kwargs)
    (;basic_params..., array_kwargs...)

	run_evolution(;basic_params..., array_kwargs...)
	expansion_shade2(;basic_params..., array_kwargs...)

	# return image

	@unpack RGB = array_kwargs
    RGB = Array(RGB)
    return reshape(RGB, (height, width))
end
#main()
@time RGB = main()
image(RGB)
