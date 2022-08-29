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
    dλ = 25.0f0
    λ_min = 400.0f0
    λ_max = 700.0f0
    depth = 3
    ITERS = 1
    sort_optimization = false

	# Geometry
	θ = 0.0f0
    θ += 2 * π / 360
	tris = [
        Sphere(zero(V3), 0.0f0),
        #Sphere(V3(2.5, 3, 0), 1.0f0),
		Sphere(V3(4, 0, 0), 1.0f0),
        Sphere(V3(0, 0, 0), 1.0f0),
    ]
    n_tris = collect(zip(map(Int32, collect(1:length(tris))), tris)) |>
        m -> reshape(m, 1, length(m))

	x = repeat(vcat(repeat([1.0f0], 16), repeat([0.0f0], 16)), 16) |> CuArray
	y = reshape(x, 1, length(x)) |> CuArray
	f(x, y, λ) = (x + y ) % 2
	Λ = CuArray(collect(λ_min:dλ:λ_max))
	tex = f.(x, y, reshape(Λ, 1, 1, length(Λ)))
#    tex = CUDA.ones(Float32, 1024, 1024, length(λ_min:dλ:λ_max))

    first_diffuse = 3


	basic_params = Dict{Symbol, Any}()
	@pack! basic_params = width, height, dλ, λ_min, λ_max, depth, sort_optimization, first_diffuse

	@time datastructs = scene_datastructs(CuArray; basic_params...)

	# Forward Trace light map

    function my_moving_camera(frame_i, frame_n)
        camera_pos = V3((8, 0, 0)) #+ centroid
        look_at = zero(V3)
        up = V3((0.0, 0.0, -1.0))
        FOV = 45.0 * pi / 180.0

        return get_camera(camera_pos, look_at, up, FOV)
    end

    cam = my_moving_camera(1, 1)

#	ray_generator(x, y, λ, dv) = simple_light(V3(7, 7, 0), V3(-1/sqrt(2), -1/sqrt(2), 0), V3(1 / sqrt(2), -1/sqrt(2), 0), V3(0, 0, 1), height, width, x, y, λ, dv)
#	ray_generator(x, y, λ, dv) = simple_light(V3(2, -2, -2), V3(-1, 0, 0), V3(0, 0, 4), V3(0, 4, 0), height, width, x, y, λ)

	#rays = wrap_ray_gen(ray_generator; datastructs...)



	# reverse trace image

	ray_generator2(x, y, λ, dv) = camera_ray(cam, height, width, x, y, λ, dv)
	rays = wrap_ray_gen(ray_generator2; datastructs...)
	@info eltype(rays)
    array_kwargs = Dict{Symbol, Any}()

    @pack! array_kwargs = tex, tris, n_tris, rays
	array_kwargs = merge(array_kwargs, datastructs)

    array_kwargs = Dict(kv[1]=>CuArray(kv[2]) for kv in array_kwargs)
    @time run_evolution!(;basic_params..., array_kwargs...)

	@time continuum_shade!(;basic_params..., array_kwargs...)

	# return image

	@unpack RGB = array_kwargs
    RGB = Array(RGB)
    return reshape(RGB, (height, width))
end
main()
@time RGB = main()
image(RGB)
