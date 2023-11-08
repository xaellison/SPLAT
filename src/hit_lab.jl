using Revise, LazyArrays, CUDA, Random, Parameters, BenchmarkTools, Serialization
include("geo.jl")
include("tracer.jl")
let 

    function my_moving_camera()
        camera_pos = ℜ³((0, 25, 65))
        look_at = ℜ³(0, 25, 0)
        up = ℜ³((0.0, -1.0, 0.0))
        FOV = 75.0 * pi / 180.0
        return get_camera(camera_pos, look_at, up, FOV)
    end

    cam = my_moving_camera()
    ray_generator(x, y, λ, dv) = camera_ray(cam, 512, 512, x, y, λ, dv)
    rays = wrap_ray_gen(ray_generator, 512, 512)
    rays = rays#[1:length(rays) - rand(1:31)]
    index_view = CuArray((1:length(rays))[1:length(rays)-rand(1:32)])
   # @info index_view
    tris = mesh_to_FTri(load("C:/Users/sandy/Documents/Github/WWRTRT/objs/artemis_smaller.obj"))
    center = tris |> centroidish
    centers, members = nothing, nothing
    try
        centers, members = open(deserialize, "hit_lab_bvs.jls")
        @info "loading cached bvs"
    catch
        @info "recomputing bvs"
        cm = cluster_fuck(tris, 256)
        open(f -> serialize(f, cm), "hit_lab_bvs.jls", "w")
        centers, members =  cm
    end
    #N_rays = 256^2
    #rays = FastRay.(Ref(center), normalize.(CUDA.rand(ℜ³, N_rays) .- CUDA.rand(ℜ³, N_rays)), 1)    
    
    #hitter = BoundingVolumeHitter(CuArray, rays, centers, members, 2)
    hitter = ExperimentalHitter4(CuArray, rays)
    tracer = StableTracer(CuArray, rays, 1)
    n_tris = tuple.(Int32(1):Int32(length(tris)), map(tri_from_ftri, tris)) |> m -> reshape(m, 1, length(m)) |> CuArray
    next_hit!(tracer, hitter, rays, index_view, n_tris)
    #next_hit!(tracer, hitter, rays, n_tris)
    
    @benchmark begin CUDA.@sync next_hit!($tracer, $hitter, $rays, $index_view, $n_tris) end evals=1 samples=20 seconds=15
    #for iter in 1:4
    #    CUDA.NVTX.@range "iter $iter" next_hit!(tracer, hitter, rays, n_tris)
    #end
    
end