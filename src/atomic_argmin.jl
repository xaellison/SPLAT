function unsafe_encode_parts(f :: Float32) #:: UInt32
    if f == 0.0f0
        return UInt32(1 << 31), zero(UInt32), zero(UInt32)
    end
    if f == Inf32
        return typemax(UInt32)
    end
    if f == -Inf32
        return typemin(UInt32)
    end
    if f > 0
        sign = f > 0 ? UInt32(1 << 31) : zero(UInt32)
        # https://en.wikipedia.org/wiki/Single-precision_floating-point_format
        # See how exponent is interpretted as exponent - 127
        
        exp = (exponent(f) + 127) * 1 << 23 |> UInt32
        sig = significand(f) * 1 << 23 - 1 << 23 |> floor |> UInt32
    else
        # for negative numbers, larger exponent = more negative
        sign = f > 0 ? UInt32(1 << 31) : zero(UInt32)
        exp = (127 - exponent(f)) * 1 << 23 |> UInt32
        # significand is signed, again, effectively subtract int-string from max value
        sig = 1 << 23 + (significand(f) * 1 << 22) |> floor |> UInt32
    end

    return sign , exp , sig
end

function unsafe_encode(f :: Float32) #:: UInt32
    return sum(unsafe_encode_parts(f))
end

function test()
    A = vcat([-Inf32, 0.0f0, Inf32], rand(Float32, 10000) .* 16 .- 8)
    B = vcat([-Inf32, 0.0f0, Inf32], rand(Float32, 10000) .* 16 .- 8)
    @assert eltype(A) == Float32
    @assert eltype(B) == Float32

    for x in A
        for y in B
            e_x, e_y = zero(UInt32), zero(UInt32)
            try
                e_x = unsafe_encode(x)
                e_y = unsafe_encode(y)
            catch E
                # debug note: output may directly reparse as float64
                @error "Failed to parse $x or $y"
                return false
            end
            if (x < y) != (e_x < e_y)
                @error  "Failed for $x < $y)"
                return false
            end
        end
    end
    return true
end

@assert test()
