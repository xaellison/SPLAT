using Revise, LazyArrays, Parameters, GLMakie, CUDA, KernelAbstractions, CUDAKernels, Random
using Statistics, BenchmarkTools
include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../procedural_assets.jl")


function main()
    out = Dict()
    for R in map(x->x^2, [256, 512, 768, 1024, 1536, 2048])
        for N in map(x->1<<x, 0:12)
            B = @benchmark begin CUDA.@sync next_hit!(tracer, hitter, rays, n_tris) end evals=1 samples = 5 seconds=15 setup = begin

                #rays = FastRay.(CUDA.zeros(ℜ³, R), CUDA.rand(ℜ³, R), CUDA.rand(Int, R))
                rays = ADRay.(CUDA.rand(ℜ³, $R),
                              CUDA.zeros(ℜ³, $R),
                              CUDA.rand(ℜ³, $R),
                              CUDA.zeros(ℜ³, $R),
                              CUDA.rand(Bool, $R),
                              CUDA.rand(Int, $R),
                              CUDA.rand(Int, $R),
                              550.0f0,
                              RAY_STATUS_ACTIVE)
                shift(T, v) = Tri(T[1], T[2] .+ v, T[3] .+ v, T[4] .+ v)
                tris = rand(Tri, $N) #.* 0.02
                #tris = shift.(tris, rand(ℜ³, N))

                meshes = [[zero(Tri)], tris]
                all_tris = foldl(vcat, meshes)

                c_tris = CuArray(tris)
                n_tris = tuple.(Int32(1):Int32(length(tris)), c_tris) |> m -> reshape(m, 1, length(m))

                hitter = ExperimentalHitter(CuArray, rays)
                tracer = Tracer(CuArray, rays)
            end
            out[(R, N)] = median(B.times)
        end
    end

    rows = [(k..., out[k]) for k in keys(out)]
    sort!(rows)
    open("experimental_times_2.csv", "w") do f
                    write(f, "rays, tris, time\n")
                  for row in rows
                      R, N, t = row
                     write(f, "$R, $N, $t\n")
                  end
              end


end

main()
