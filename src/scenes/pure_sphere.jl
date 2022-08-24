# RUN FROM /
using Revise, LazyArrays, Parameters, GLMakie, CUDA, KernelAbstractions , CUDAKernels

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../utils.jl")

function main()
    width = 512
    height = 512
    dλ = 5.0f0
    λ_min = 400.0f0
    λ_max = 700.0f0
    depth = 3
    ITERS = 1

	basic_params = Dict{Symbol, Any}()
	@pack! basic_params = width, height, dλ, λ_min, λ_max, depth

	datastructs = scene_datastructs(CuArray ;basic_params...)

    θ = 0.0f0

    θ += 2 * π / 360
    function my_moving_camera(frame_i, frame_n)
        camera_pos = V3((7, 0, 0)) #+ centroid
        look_at = zero(V3)
        up = V3((0.0, 0.0, -1.0))
        FOV = 45.0 * pi / 180.0

        return get_camera(camera_pos, look_at, up, FOV)
    end

    tris = [
        Sphere(zero(V3), 0.0f0),
        Sphere(V3(3, 0, 0), 1.0f0),
        Sphere(V3(-1, cos(θ), sin(θ)), 1.0f0),
    ]
    n_tris = collect(zip(map(Int32, collect(1:length(tris))), tris)) |>
        m -> reshape(m, 1, length(m))

    tex = let
        x = collect(1:64)
        y = collect(1:64) |> a -> reshape(a, 1, length(a))
        z = CuArray(map(Float32, (x .+ y) .% 2))
    end
    #    CUDA.rand(Float32, 64, 64)

    first_diffuse = 3
    sort_optimization = false
    cam = my_moving_camera(1, 1)
	ray_generator(x, y, λ, dv) = camera_ray(cam, height, width, x, y, λ, dv)

	rays = wrap_ray_gen(ray_generator; datastructs...)

    scalar_kwargs = Dict{Symbol, Any}()
    array_kwargs = Dict{Symbol, Any}()
    @pack! scalar_kwargs =  first_diffuse,
                            sort_optimization,
                            ray_generator

	scalar_kwargs = merge(scalar_kwargs, basic_params)

    @pack! array_kwargs = tex, tris, n_tris, rays
	array_kwargs = merge(array_kwargs, datastructs)

    array_kwargs = Dict(kv[1]=>CuArray(kv[2]) for kv in array_kwargs)
    CUDA.@time run_evolution(;scalar_kwargs..., array_kwargs...)

	light_map2!(;scalar_kwargs..., array_kwargs...)
	@unpack RGB = array_kwargs
    RGB = Array(RGB)
    return reshape(RGB, (height, width))
end
main()
RGB= main()
image(RGB)
