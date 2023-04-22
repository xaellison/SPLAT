using Makie

function piecewise(points)
    # generate a piecewise function with respect to an ascending x sorted iterable
    # of tuples like (x, y) which needs at most log2(N) conditionals to find the
    # domain to linearly interpolate over. Values outside the domain will be
    # interpolated with the nearest segment.
    L = length(points)
    function helper(lo, hi, i)
        # hi - inclusive upper bound
        # lo - exclusive lower bound
        # i - splitting index
        # base case
        if hi - lo < 2
            return error("bad")
        end
        if hi - lo == 2
            # interpolater
            y1 = points[hi][2]
            y2 = points[lo+1][2]
            x1 = points[hi][1]
            x2 = points[lo+1][1]
            return :(($y1 - $y2) / ($x1 - $x2) * (λ - $x1) + $y1)
        end
        left = helper(lo, i, Int(ceil((lo + i) / 2)))
        right = helper(i - 1, hi, Int(ceil((i + hi) / 2)))
        splitter = points[i][1]
        return :(λ < $splitter ? $left : $right)
    end
    f = helper(0, L, Int(ceil(L / 2)))
    return eval(:(λ -> $f))
end

_retina_red = piecewise([
    (399, 0),
    (400, 0.0647036),
    (420, 0.147492),
    (438, 0.195683),
    (460, 0.162951),
    (466, 0),
    (534, 0),
    (538, 0.243127),
    (571, 0.798209),
    (593, 0.980253),
    (602, 0.999977),
    (610, 0.98481),
    (660, 0.375838),
    (680, 0.210109),
    (700, 0.11005),
])

_retina_green = piecewise([
    (399, 0),
    (462, 0),
    (467, 0.165356),
    (474, 0.280228),
    (502, 0.587139),
    (521, 0.829692),
    (534, 0.924245),
    (551, 0.910167),
    (572, 0.777366),
    (600, 0.372516),
    (609, 0.158767),
    (612, 0),
    (700, 0),
])

_retina_blue = piecewise([
    (399, 0),
    (413, 0.23889),
    (418, 0.291473),
    (437, 0.437376),
    (456, 0.590128),
    (463, 0.72815),
    (476, 0.581812),
    (494, 0.452254),
    (515, 0.407615),
    (528, 0.299756),
    (536, 0),
    (608, 0),
    (611, 0.0407256),
    (614, 0.154091),
    (620, 0.220637),
    (635, 0.225269),
    (667, 0.116114),
    (700, 0.0424846),
])

function visualize_spectrum(file_name = "palette.png")
    _width = 300
    out = zeros(Float32, 3, _width, _width)

    r0 = sum(retina_red(400 + i) for i = 1:_width)
    g0 = sum(retina_green(400 + i) for i = 1:_width)
    b0 = sum(retina_blue(400 + i) for i = 1:_width)

    for i = 1:_width
        for j = 1:_width
            r = _retina_red(400 + i) #/ r0
            g = _retina_green(400 + i) #/ g0
            b = _retina_blue(400 + i) #/ b0

            out[1, i, j], out[2, i, j], out[3, i, j] = (r, g, b)
        end
    end
    img = Array{RGBf}(undef, _width, _width)
    for i = 1:_width
        for j = 1:_width
            #println(out[:, i, j])
            img[i, j] = RGBf(out[:, i, j]...)
        end
    end
    Makie.save(file_name, img)
end

_R0 = sum(_retina_red(x) for x = 400:700)
_G0 = sum(_retina_green(x) for x = 400:700)
_B0 = sum(_retina_blue(x) for x = 400:700)

function retina_red(λ)
    return _retina_red(λ) / _R0
end

function retina_green(λ)
    return _retina_green(λ) / _G0
end

function retina_blue(λ)
    return _retina_blue(λ) / _B0
end


export retina_red, retina_green, retina_blue

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