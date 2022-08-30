# RUN FROM /
using Revise, LazyArrays, Parameters, GLMakie, CUDA, KernelAbstractions, CUDAKernels

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../utils.jl")
include("../procedural_assets.jl")

function main()
    # Tracing params
    width = 1024
    height = 1024
    dλ = 5.0f0
    λ_min = 400.0f0
    λ_max = 700.0f0
    depth = 2
    ITERS = 1
    sort_optimization = false

    # Geometry
    rips = ripples()
    sq = square()
    map!(t -> translate(t, V3(-0.5, -0.5, 0)), rips, rips)

    @info model_box(rips)
    map!(t -> translate(t, V3(-0.5, -0.5 - 0.5, -3)), sq, sq)

    meshes = [[zero(FTri)], rips, sq]

    first_diffuse = 1 + length(rips) + 1
    tris = foldl(vcat, meshes)

    n_tris =
        collect(zip(map(Int32, collect(1:length(tris))), tris)) |>
        m -> reshape(m, 1, length(m))

    Λ = collect(λ_min:dλ:λ_max)
    tex = CUDA.ones(Float32, 512, 512, length(Λ))

    basic_params = Dict{Symbol,Any}()
    @pack! basic_params =
        width, height, dλ, λ_min, λ_max, depth, sort_optimization, first_diffuse

    @time datastructs = scene_datastructs(CuArray; basic_params...)

    # Forward Trace light map

    ray_generator(x, y, λ, dv) = simple_light(
        V3(0, 0, 5),
        V3(0, -0.25, -1),
        V3(1, 0, 0),
        V3(0, 1, 0),
        height,
        width,
        x,
        y,
        λ,
        dv,
    )

    rays = wrap_ray_gen(ray_generator; datastructs...)
    array_kwargs = Dict{Symbol,Any}()

    @pack! array_kwargs = tex, tris, n_tris, rays
    array_kwargs = merge(array_kwargs, datastructs)
    array_kwargs = Dict(kv[1] => CuArray(kv[2]) for kv in array_kwargs)

    run_evolution!(; basic_params..., array_kwargs...)
    continuum_light_map!(; basic_params..., array_kwargs...)

    # return image

    @unpack tex, retina_factor = array_kwargs
    RGB3 = Array(
        sum(reshape(tex, length(tex) ÷ length(Λ), 1, length(Λ)) .* retina_factor, dims = 3),
    )

    RGB = RGBf.(RGB3[:, 1], RGB3[:, 2], RGB3[:, 3])
    return reshape(RGB, (size(tex)[1:2]))
end

@time RGB = main()
image(RGB)
