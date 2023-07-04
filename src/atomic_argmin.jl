# Linearly encode float32s into uint32 in a way that preserves comparison & equality
# for all normal non-NaN values


function monotonic_reinterpret(::Type{UInt32}, f::Float32)::UInt32
    # equivalent with float comparison
    #f < 0 ? ~reinterpret(UInt32, f) : reinterpret(UInt32, f) + UInt32(1<<31)
    uint = reinterpret(UInt32, f)
    uint > 1 << 31 ? ~uint : uint | UInt32(1 << 31)
end

function monotonic_reinterpret(::Type{UInt64}, f_i :: Tuple{Float32, UInt32})::UInt64
    f, i = f_i
    return UInt64(monotonic_reinterpret(UInt32, f)) << 32 | UInt64(i)
end

function retreive_arg(i64::UInt64)::UInt32
    # for an UInt64 encoding a (max/min, argmax/argmin), return argmax/argmin
    return UInt32(i64 ⊻ (i64 >> 32 << 32))
end

function monotonic_reinterpret(::Type{Float32}, i32::UInt32)::Float32
    flipped_sign = i32 >> 31 << 31
    if flipped_sign == 0
        # originally, sign 0 positive, so this is negative, restore
        return reinterpret(Float32, ~i32)
    else
        return reinterpret(Float32, i32 ⊻ UInt32(1 << 31))
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
    if !all(monotonic_reinterpret.(Float32, monotonic_reinterpret.(UInt32, A)) .== A)
        failure = findfirst(i -> monotonic_reinterpret(Float32, monotonic_reinterpret(UInt32, A[i])) != A[i], 1:length(A))
        @error "Fails for float = $(A[failure])"
        return false
    end
    true
end

function test_comparison()
    A = test_array()
    B = test_array()
    try
        e_A = map(a->monotonic_reinterpret(UInt32, a), A)
        e_B = map(b->monotonic_reinterpret(UInt32, b), B)
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
        e_A = map(a->monotonic_reinterpret(Float32, a), A)
        e_B = map(b->monotonic_reinterpret(Float32, b), B)
        for (x, e_x) in zip(A, e_A)
            for (y, e_y) in zip(B, e_B)
                if !isnan(e_x) & !isnan(e_y) & (x < y) != (e_x < e_y)
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
@benchmark monotonic_reinterpret(UInt32, f) setup = begin
    f = -rand(Float32) .- 0.5f0
end evals = 100 samples = 10
