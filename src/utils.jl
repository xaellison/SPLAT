function _spectrum_datastructs(A, λ_range)
    spectrum = collect(λ_range) |> a -> reshape(a, 1, 1, length(a))
    retina_factor = Array{Float32}(undef, 1, 3, length(spectrum))
    map!(retina_red, begin
        @view retina_factor[1, 1, :]
    end, spectrum)
    map!(retina_green, begin
        @view retina_factor[1, 2, :]
    end, spectrum)
    map!(retina_blue, begin
        @view retina_factor[1, 3, :]
    end, spectrum)
    retina_factor = A(retina_factor)
    return spectrum, retina_factor
end

function scene_datastructs(A; width, height, dλ, λ_min, λ_max, depth, kwargs...)
    RGB3 = A{Float32}(undef, width * height, 3)
    RGB3 .= 0
    RGB = A{RGBf}(undef, width * height)

    row_indices = A(1:height)
    col_indices = reshape(A(1:width), 1, width)
    rays = A{ADRay}(undef, width * height)
    hit_idx = A(zeros(Int32, length(rays)))
    dv = A{ℜ³}(undef, height, width)

    # use host to compute constants used in turning spectra into colors
    spectrum, retina_factor = _spectrum_datastructs(A, λ_min:dλ:λ_max)

    # Datastruct init
    expansion = A{FastRay}(undef, (length(rays)))
    # for Sphere geometry:
    # tmp = A{Tuple{Float32, Int32}}(undef, size(expansion))
    tmp = A{UInt64}(undef, size(expansion))
    rndm = rand(Float32, height * width)
    out = Dict{Symbol,Any}()
    @pack! out = RGB,
    RGB3,
    row_indices,
    col_indices,
    rays,
    hit_idx,
    dv,
    spectrum,
    retina_factor,
    expansion,
    tmp,
    rndm
    return out
end


function forward_datastructs(A, rays; dλ, λ_min, λ_max, kwargs...)
    hit_idx = A(zeros(Int32, length(rays)))

    # use host to compute constants used in turning spectra into colors
    spectrum, retina_factor = _spectrum_datastructs(A, λ_min:dλ:λ_max)

    # Datastruct init
    expansion = A{FastRay}(undef, (length(rays)))
    # for Sphere geometry:
    # tmp = A{Tuple{Float32, Int32}}(undef, size(expansion))
    tmp = A{UInt64}(undef, size(rays))
    rndm = rand(Float32, length(rays))
    out = Dict{Symbol,Any}()
    @pack! out = hit_idx,
                 spectrum,
                 retina_factor,
                 expansion,
                 tmp,
                 rndm
    return out
end
