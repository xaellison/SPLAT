function stage()
    z_floor = -2
    a = ℜ³(3, 1, z_floor)
    b = ℜ³(-3, 1, z_floor)
    c = ℜ³(0, -2, z_floor)
    d = ℜ³(0, -2, 2)
    t_a = ℜ³(0, 0, 0)
    t_b = ℜ³(1, 0, 0)
    t_c = ℜ³(0.5, 0.5, 0)
    t_d = ℜ³(0.5, 1, 0)
    n1 = normalize(cross(a - b, a - c))
    T1 = FTri(n1, a, b, c, n1, n1, n1, t_a, t_b, t_c)
    n2 = normalize(cross(d - b, d - c))
    T2 = FTri(n2, b, c, d, n2, n2, n2, t_b, t_c, t_d)
    n3 = normalize(cross(d - a, d - c))
    T3 = FTri(n3, a, d, c, n3, n3, n3, t_a, t_d, t_c)
    return [T1, T2, T3]
end

function ripples(t=0)
    # xy surface on unit square
    x0, y0, x1, y1 = 0, 0, 1, 1
    δx = 0.00125
    δy = 0.00125
    Ω(k) = k * tanh(k)
    f(k, x, t) = sin(k * x - Ω(k) * t)
    f(x, y) = (f(50, x, t)+f(60, x, t)+f(70, y, t)+f(80, y, t))*0.000125
    #sqrt((2+sin(15 * x - 2 * pi * t))^-1 + (2+sin(20 * y + 5 * x - 2 * pi * t))^-1) * 0.04
    V(x, y) = ℜ³(x, y, f(x, y))
    # https://mathworld.wolfram.com/NormalVector.html
    N(x, y) = ℜ³(
        ForwardDiff.derivative(x -> f(x, y), x),
        ForwardDiff.derivative(y -> f(x, y), y),
        -1,
    )

    x = collect(x0:δx:x1-δx)
    y = collect(y0:δy:y1-δy)
    y = reshape(y, 1, length(y))

    vertices = N.(x, y)
    # together, tri1(x,y) & tri2(x, y) complete a quad
    tri1(x, y) =
        let
            a = V(x, y)
            b = V(x + δx, y)
            c = V(x, y + δy)
            n = cross(a - b, a - c)
            n_a = N(x, y)
            n_b = N(x + δx, y)
            n_c = N(x, y + δy)
            t_a = ℜ³(a[1:2]..., 0)
            t_b = ℜ³(b[1:2]..., 0)
            t_c = ℜ³(c[1:2]..., 0)
            FTri(n, a, b, c, n_a, n_b, n_c, t_a, t_b, t_c)
        end
    tri2(x, y) =
        let
            a = V(x + δx, y + δy)
            b = V(x + δx, y)
            c = V(x, y + δy)
            n = cross(a - b, a - c)
            n_a = N(x + δx, y + δy)
            n_b = N(x + δx, y)
            n_c = N(x, y + δy)
            t_a = ℜ³(a[1:2]..., 0)
            t_b = ℜ³(b[1:2]..., 0)
            t_c = ℜ³(c[1:2]..., 0)
            FTri(n, a, b, c, n_a, n_b, n_c, t_a, t_b, t_c)
        end
    out = vcat(tri1.(x, y), tri2.(x, y))
    reshape(out, length(out))
end


function square()
    a = ℜ³(0, 0, 0)
    b = ℜ³(1, 0, 0)
    c = ℜ³(0, 1, 0)
    d = ℜ³(1, 1, 0)
    n = ℜ³(0, 0, 1)
    return [FTri(n, a, b, c, n, n, n, a, b, c), FTri(n, b, c, d, n, n, n, b, c, d)]
end


function checkered_tex(pixels_per_square, squares, N_λ)
    # output resolution = pixels_per_square * squares * 2 with N_λ channels
    x =
        repeat(
            vcat(repeat([1.0f0], pixels_per_square), repeat([0.0f0], pixels_per_square)),
            squares,
        ) |> CuArray
    y = reshape(x, 1, length(x)) |> CuArray
    f(x, y, λ) = (x + y) % 2
    Λ = CuArray(collect(1:N_λ))
    tex = f.(x, y, reshape(Λ, 1, 1, length(Λ)))
end
