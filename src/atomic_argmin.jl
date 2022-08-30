# Linearly encode float32s into uint32 in a way that preserves comparison & equality
# for all normal non-NaN values

@inline function unsafe_encode_parts(f::Float32)::Tuple{UInt32,UInt32,UInt32}
    if f == 0.0f0
        # greater than any negative, but less than any positive since significand
        # for any non-zero number is 1.x
        return UInt32(1 << 31), zero(UInt32), zero(UInt32)
    end
    if f == Inf32
        return typemax(UInt32), zero(UInt32), zero(UInt32)
    end
    if f == -Inf32
        return typemin(UInt32), zero(UInt32), zero(UInt32)
    end
    if f > 0
        sign = UInt32(1 << 31)
        # https://en.wikipedia.org/wiki/Single-precision_floating-point_format
        # See how exponent is interpretted as exponent - 127
        # TODO: can handle subnormal nums by identifying when exponent(f) < -127
        exp = (exponent(f) + 127) * 1 << 23 |> UInt32
        sig = (significand(f) - 1) * 1 << 23 |> UInt32
    else
        # for negative numbers, larger exponent = more negative
        sign = zero(UInt32)
        exp = (127 - exponent(f)) * 1 << 23 |> UInt32
        # significand is signed. Again, effectively subtract int-string from max value
        sig = (2 + significand(f)) * 1 << 23 |> UInt32
    end

    return sign, exp, sig
end

function unsafe_encode(f::Float32)::UInt32
    (sign, exp, sig) = unsafe_encode_parts(f)
    return sign + exp + sig
end

function unsafe_encode(f::Float32, i::UInt32)::UInt64
    encoded_f = UInt64(unsafe_encode(f))
    return encoded_f * 1 << 32 + i
end

function unsafe_decode(i64::UInt64)::UInt32
    return UInt32(i64 % 1 << 32)
end

function test()
    A = vcat([-Inf32, 0.0f0, Inf32], rand(Float32, 100000) .* 16 .- 8)
    B = vcat([-Inf32, 0.0f0, Inf32], rand(Float32, 100000) .* 16 .- 8)
    try
        e_A = map(unsafe_encode, A)
        e_B = map(unsafe_encode, B)
        for (x, e_x) in zip(A, e_A)
            for (y, e_y) in zip(B, e_B)
                if (x < y) != (e_x < e_y)
                    @error "Failed for $x < $y)"
                    return false
                end
            end
        end
    catch E
        # debug note: output may directly reparse as float64
        @error E
        return false
    end

    return true
end

using BenchmarkTools
@benchmark unsafe_encode(f) setup = begin
    f = -rand(Float32) .- 0.5f0
end evals = 10 samples = 10
