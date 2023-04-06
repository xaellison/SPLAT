using Revise, LazyArrays, Parameters, CUDA, KernelAbstractions, CUDAKernels, Random
using Statistics, BenchmarkTools
include("../geo.jl")
include("../skys.jl")
include("../tracer.jl")
include("../procedural_assets.jl")


function main()
    out = Dict()
    for R in map(x->x^2, [256, 512, 768, 1024])
        for N in map(x->1<<x, 0:2:16)
            @info "$R $N"
            B = @benchmark begin CUDA.@sync next_hit!(tracer, hitter, rays, n_tris) end evals=1 samples = 5 seconds=15 setup = begin

                rays = FastRay.(CUDA.zeros(ℜ³, $R), CUDA.rand(ℜ³, $R), CUDA.rand(Int, $R))
                shift(T, v) = Tri(T[1], T[2] .+ v, T[3] .+ v, T[4] .+ v)
                tris = rand(Tri, $N) #.* 0.02
                #tris = shift.(tris, rand(ℜ³, N))

                meshes = [[zero(Tri)], tris]
                all_tris = foldl(vcat, meshes)

                c_tris = CuArray(tris)
                n_tris = tuple.(Int32(1):Int32(length(tris)), c_tris) |> m -> reshape(m, 1, length(m))

                hitter = ExperimentalHitter6(CuArray, rays)
                tracer = StableTracer(CuArray, rays, 1)
            end
            out[(R, N)] = median(B.times)
        end
    end

    rows = [(k..., out[k]) for k in keys(out)]
    sort!(rows)
    open("exp_6.csv", "w") do f
                    write(f, "rays, tris, time\n")
                  for row in rows
                      R, N, t = row
                     write(f, "$R, $N, $t\n")
                  end
              end


end

main()
