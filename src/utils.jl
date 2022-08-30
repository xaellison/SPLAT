function scene_datastructs(A; width, height, dλ, λ_min, λ_max, depth, kwargs...)
    RGB3 = A{Float32}(undef, width * height, 3)
    RGB3 .= 0
    RGB = A{RGBf}(undef, width * height)

    row_indices = A(1:height)
    col_indices = reshape(A(1:width), 1, width)
    rays = A{ADRay}(undef, width * height)
    hit_idx = A(zeros(Int32, length(rays)))
    dv = A{V3}(undef, height, width)

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
    retina_factor = A(retina_factor)

    # Datastruct init
    expansion = A{FastRay}(undef, (length(rays)))
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
