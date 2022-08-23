function scene_datastructs(;width, height, dλ, λ_min, λ_max, depth)
    RGB3 = Array{Float32}(undef, width * height, 3)
    RGB = Array{RGBf}(undef, width * height)

    row_indices = Array(1:height)
    col_indices = reshape(Array(1:width), 1, width)
    rays = Array{ADRay}(undef, width * height)
    hit_idx = Array(zeros(Int32, length(rays)))
    dv = Array{V3}(undef, height, width) # make w*h

    # use host to compute constants used in turning spectra into colors
    spectrum = collect(λ_min:dλ:λ_max) |> a -> reshape(a, 1, 1, length(a))
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


    # Datastruct init
    expansion = Array{FastRay}(undef, (length(rays)))
    tmp = Array{Tuple{Float32, Int32}}(undef, size(expansion))
    rndm = rand(Float32, height * width)
    out = Dict{Symbol, Any}()
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
