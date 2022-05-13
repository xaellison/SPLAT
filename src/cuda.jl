# GPU specific methods
using CUDA, Tullio, KernelAbstractions, CUDAKernels

function typemax(::Type{Tuple{Tuple{Float32,Float32},Int32}})
    return ((Inf32, Inf32), one(Int32))
end

function typemax(::Type{Tuple{Float32,Int32}})
    return (Inf32, one(Int32))
end

function hit_argmin(n_t, r::ADRay) :: Tuple{Tuple{Float32, Float32}, Int32}
    return get_hit(n_t, r)[1:2]
end

function hit_argmin(n_t, r::FastRay) :: Tuple{Float32, Int32}
    return get_hit(n_t, r)[1:2]
end

function next_hit!(dest :: AnyCuArray{I}, rays, n_tris:: AnyCuArray{Tuple{I, T}}, override) where {I, T}

  @tullio (min) tmp[i] := hit_argmin(n_tris[j], rays[i])
  d_view = @view dest[:]
  d_view = reshape(d_view, length(d_view))
  d_view .= map(x->x[2], tmp)
  return
end
