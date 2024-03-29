# RUN FROM /
using Revise, LazyArrays, Parameters, GLMakie, CUDA, NVTX

include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../procedural_assets.jl")

function main()
    # Tracing params
    width = 1280
    height = 720
    dλ = 25.0f0
    λ_min = 400.0f0
    λ_max = 700.0f0
    depth = 3
    ITERS = 1
    forward_upscale = 2
    backward_upscale = 2
    # Geometry

    obj_path = "objs/icos.obj"

    glass_sphere = mesh_to_FTri(load(obj_path))


    solid_sphere = mesh_to_FTri(load(obj_path))
    solid_sphere = map(t -> translate(t, ℜ³(1, 0, -4)), solid_sphere)

    glass_sphere2 = mesh_to_FTri(load(obj_path))
    glass_sphere2 = map(t -> translate(t, ℜ³(0, 2, 0)), glass_sphere2)

    solid_sphere2 = mesh_to_FTri(load(obj_path))
    solid_sphere2 = map(t -> translate(t, ℜ³(1, 2, -4)), solid_sphere2)

    meshes = [[zero(FTri)], glass_sphere, glass_sphere2, solid_sphere, solid_sphere2]
    first_diffuse = 1 + 1 + length(glass_sphere) * 2
    host_tris = foldl(vcat, meshes)
    tris = CuArray(host_tris)


    tex_f() = checkered_tex(32, 16, length(λ_min:dλ:λ_max)) .* 12

    basic_params = Dict{Symbol,Any}()
    @pack! basic_params = width,
    height,
    dλ,
    λ_min,
    λ_max,
    depth,
    first_diffuse,
    forward_upscale,
    backward_upscale

    function my_moving_camera()
        camera_pos = ℜ³((0, 5, 5))
        look_at = zero(ℜ³)
        up = ℜ³((0.0, 0.0, -1.0))
        FOV = 45.0 * pi / 180.0

        return get_camera(camera_pos, look_at, up, FOV)
    end

    cam = my_moving_camera()

    lights = [RectLight(ℜ³(0, 0, 8), ℜ³(0, 0, -1), ℜ³(1, 0, 0), ℜ³(0, 1, 0), height, width)]

    #bounding_volumes, bounding_volumes_members = cluster_fuck(Array(tris), 4)

    bounding_volumes, bounding_volumes_members = bv_partition(tris, 5; verbose = true)

    forward_hitter_A = DPBVHitter(
        CuArray,
        height * width ÷ (forward_upscale^2) * length(lights),
        tris,
        bounding_volumes,
        bounding_volumes_members;
        concurrency = 32,
    )
    backward_hitter_A = DPBVHitter(
        CuArray,
        height * width ÷ (backward_upscale^2),
        tris,
        bounding_volumes,
        bounding_volumes_members;
        concurrency = 32,
    )
    forward_hitter_B = DPBVHitter(
        CuArray,
        height * width ÷ (forward_upscale^2) * length(lights),
        tris,
        bounding_volumes,
        bounding_volumes_members;
        concurrency = 32,
    )
    backward_hitter_B = DPBVHitter(
        CuArray,
        height * width ÷ (backward_upscale^2),
        tris,
        bounding_volumes,
        bounding_volumes_members;
        concurrency = 32,
    )

    #forward_hitter = ExperimentalHitter6(CuArray, height * width ÷ (forward_upscale ^ 2) * length(lights))#, tris, bounding_volumes, bounding_volumes_members)
    #backward_hitter = ExperimentalHitter6(CuArray, height * width ÷ (backward_upscale ^ 2))#, tris, bounding_volumes, bounding_volumes_members)

    #forward_hitter = BoundingVolumeHitter(CuArray, height * width ÷ (forward_upscale ^ 2) * length(lights), bounding_volumes, bounding_volumes_members)
    #backward_hitter = BoundingVolumeHitter(CuArray, height * width ÷ (backward_upscale ^ 2), bounding_volumes, bounding_volumes_members)

    host_RGB_A, host_RGB_B =
        Array{RGBf}(undef, height, width), Array{RGBf}(undef, height, width)
    device_RGB_A, device_RGB_B = CuArray(host_RGB_A), CuArray(host_RGB_B)
    Mem.pin(host_RGB_A)
    Mem.pin(host_RGB_B)
    skydome = let 
        a = RGBf.(load("objs/skydome2.png"))
        R = map(rgb->rgb.r, a)
        G = map(rgb->rgb.g, a)
        B = map(rgb->rgb.b, a)
        RGB = Array{Float32}(undef, size(R)..., 3)
        RGB[:, :, 1] .= R
        RGB[:, :, 2] .= G
        RGB[:, :, 3] .= B
        CuTexture(CuTextureArray(RGB); interpolation=CUDA.LinearInterpolation())
    end 
    runme_A(i) = begin
        trace_kwargs = Dict{Symbol,Any}()
        forward_hitter, backward_hitter = forward_hitter_A, backward_hitter_A

        @pack! trace_kwargs =
            cam, lights, tex_f, tris, λ_min, dλ, λ_max, forward_hitter, backward_hitter
        trace_kwargs = merge(basic_params, trace_kwargs)
        RGB = trace!(
            StableTracer,
            ExperimentalImager2;
            intensity = 1.0f0,
            force_rand = 0.0f0,
            skydome=skydome,
            trace_kwargs...,
        )
        #return reshape(RGB, (height, width))
        #copyto!(host_RGB, RGB)

        device_RGB_A = (reshape(RGB, (height, width)))

    end


    runme_B(i) = begin
        trace_kwargs = Dict{Symbol,Any}()
        forward_hitter, backward_hitter = forward_hitter_B, backward_hitter_B

        @pack! trace_kwargs =
            cam, lights, tex_f, tris, λ_min, dλ, λ_max, forward_hitter, backward_hitter
        trace_kwargs = merge(basic_params, trace_kwargs)
        RGB = trace!(
            StableTracer,
            ExperimentalImager2;
            intensity = 1.0f0,
            #force_rand = 1.0f0,
            skydome=skydome,
            trace_kwargs...,
        )
        #return reshape(RGB, (height, width))
        #copyto!(host_RGB, RGB)

        device_RGB_B = (reshape(RGB, (height, width)))

    end


    matrix = Array{RGBf}(undef, height, width)
    fig, ax, hm = image(1:height, 1:width, matrix)


    # we use `record` to show the resulting video in the docs.
    # If one doesn't need to record a video, a normal loop works as well.
    # Just don't forget to call `display(fig)` before the loop
    # and without record, one needs to insert a yield to yield to the render task

    runme_A(1)
    runme_B(2)
    hm[3] = runme_A(1)
    display(fig)

    tasks = Dict()

    t0 = @async 1
    tasks["A"] = t0

    @time for i = 1:40
        #    events(hm).mouseposition |> println

        if i % 2 == 1
            # assume RBG_A ready, get B ready for next loop
            tasks["B"] = @async begin
                runme_B(i)
                copyto!(host_RGB_B, device_RGB_B)
            end#host_RGB_B = Array(device_RGB_B)

            wait(tasks["A"])
            #Threads.@spawn save("out/$i.png", host_RGB_A)	

        #    hm[3] = host_RGB_A
        else
            tasks["A"] = @async begin
                runme_A(i)
                copyto!(host_RGB_A, device_RGB_A)
            end#host_RGB_A = Array(device_RGB_A)

            wait(tasks["B"])
            #Threads.@spawn save("out/$i.png", host_RGB_B)
            #hm[3] = host_RGB_B
        end

        yield()
        #@async save("out/$i.png", fig )
    end
end
CUDA.@profile main()


