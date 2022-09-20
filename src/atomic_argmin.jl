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
    #TODO rename to retreive or something
    return UInt32(i64 % 1 << 32)
end

function unsafe_decode_parts(i32::UInt32)
    sign = i32 ÷ (1 << 31)
    exp = i32 % (1 << 31) ÷ (1 << 23)
    sig = i32 % (1 << 23)
    return sign, exp, sig
end

function unsafe_decode(i32::UInt32)::Float32
    u_sign, u_exp, u_sig = unsafe_decode_parts(i32)
    if (u_sign, u_exp, u_sig) == (1, 0, 0)
        # special case avoids 0.0f0 -> 5.877472f-39
        return 0.0f0
    end
    if u_sign > 0
        exp = Int32(u_exp) - 127
        sig = Float32(u_sig) / (1 << 23) + 1
        return 2.0f0^exp * sig
    else
        exp = 127 - Int32(u_exp)
        sig = Float32(u_sig) / (1 << 23) - 2
        return 2.0f0^exp * sig
    end
end

function test_array()
    # numbers that we definitely want in the test
    special_numbers = [-Inf32, 0.0f0, Inf32]
    A = foldl(vcat, rand(Float32, 1000) * (2.0f0^i) .* 2 .- 1 for i = -100:5:100)
    A = vcat(A, special_numbers)
    A
end

function test_reversibility()
    A = test_array()
    if !all(unsafe_decode.(unsafe_encode.(A)) .== A)
        failure = findfirst(i -> unsafe_decode(unsafe_encode(A[i])) != A[i], 1:length(A))
        @error "Fails for float = $(A[failure])"
        return false
    end
    true
end

function test_comparison()
    A = test_array()
    B = test_array()
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


function test_comparison2()
    N = 100000
    A = UInt32.(rand(UInt32, N) .% (1 << 24) .+ rand(UInt32, N) .% (1 << 31))
    B = UInt32.(rand(UInt32, N) .% (1 << 24) .+ rand(UInt32, N) .% (1 << 31))
    try
        e_A = map(unsafe_decode, A)
        e_B = map(unsafe_decode, B)
        for (x, e_x) in zip(A, e_A)
            for (y, e_y) in zip(B, e_B)
                if (x < y) != (e_x < e_y)
                    @error "Failed for $x < $y != $e_x < $e_y"
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
