include("../src/geo.jl")
include("../src/tracer.jl")
using Test
# diamond around origin

function test1()
    a = V3(1, 0, 0)
    b = V3(0, 1,0)
    c = -a
    d = -b
    na=a
    nc=c
    nb=nd=V3(0,0,1)
    t1 = STri(normalize(cross(a-b,b-d)), a, b, d, na, nb, nd)
    t2 = STri(normalize(cross(c-b,b-d)), c, b, d, nc, nb, nd)
    θ = pi / 6
    ϕ = pi / 3
    P(θ, ϕ) = V3(1,0,0)*sin(θ)*cos(ϕ)+V3(0,1,0)*sin(θ)*sin(ϕ) - V3(0,0,1)*cos(θ)
    D(θ, ϕ) = -P(θ, ϕ)

    # ray from point on unit sphere to origin. Dir varies but pos doesn't
    R = ADRay(P(θ, ϕ),
               zero(V3),
               zero(V3),
               zero(V3),
               D(θ, ϕ),
               zero(V3),
               ForwardDiff.derivative(θ->D(θ, ϕ), θ),
               ForwardDiff.derivative(ϕ->D(θ, ϕ), ϕ),
               true,
               -1,
               1,
               0.0f0,
               zero(V3),
               false)


    dx1 = ForwardDiff.derivative(x->p(R, t1, 0.0f0, x, 0.0f0), 0.0f0)
    dy1 = ForwardDiff.derivative(y->p(R, t1, 0.0f0, 0.0f0, y), 0.0f0)
    dx2 = ForwardDiff.derivative(x->p(R, t2, 0.0f0, x, 0.0f0), 0.0f0)
    dy2 = ForwardDiff.derivative(y->p(R, t2, 0.0f0, 0.0f0, y), 0.0f0)

    # test cross dx dy is parallel to tri normal
    @test dot(cross(dx1, dy1), V3(0, 0, 1)) ≈ norm(cross(dx1, dy1))
    @test dot(cross(dx2, dy2), V3(0, 0, 1)) ≈ norm(cross(dx2, dy2))
    @test dx1 ≈ dx2
    @test dy1 ≈ dy2

    d0 = refract(ray_dir(R, 0.0f0, 0.0f0, 0.0f0), optical_normal(t1, p(R, t1, 0.0f0, 0.0f0, 0.0f0)), 1.0f0, 2.0f0)
    dx1 = ForwardDiff.derivative(x->refract(ray_dir(R, 0.0f0, x, 0.0f0), optical_normal(t1, p(R, t1, 0.0f0, x, 0.0f0)), 1.0f0, 2.0f0), 0.0f0)
    dy1 = ForwardDiff.derivative(y->refract(ray_dir(R, 0.0f0, 0.0f0, y), optical_normal(t1, p(R, t1, 0.0f0, 0.0f0, y)), 1.0f0, 2.0f0), 0.0f0)
    dx2 = ForwardDiff.derivative(x->refract(ray_dir(R, 0.0f0, x, 0.0f0), optical_normal(t2, p(R, t2, 0.0f0, x, 0.0f0)), 1.0f0, 2.0f0), 0.0f0)
    dy2 = ForwardDiff.derivative(y->refract(ray_dir(R, 0.0f0, 0.0f0, y), optical_normal(t2, p(R, t2, 0.0f0, 0.0f0, y)), 1.0f0, 2.0f0), 0.0f0)
    # test dx and dy perpedicular to center of ray for both tris
    @test dot(cross(dx1, dy1), d0) ≈ norm(cross(dx1, dy1))
    @test dot(cross(dx2, dy2), d0) ≈ norm(cross(dx2, dy2))
    # test derivatives in refracted ray are same for same point on edge of two tris (and non-zero)
    @test dx1 ≈ dx2
    @test dy1 ≈ dy2
    @test norm(dx1) > 0
    @test norm(dy1) > 0
    @test norm(dx2) > 0
    @test norm(dy2) > 0

end

function test2()
    a = V3(1, 0, 0)
    b = V3(0, 1,0)
    c = -a
    d = -b
    na=a
    nc=c
    nb=nd=V3(0,0,1)
    t1 = STri(normalize(cross(a-b,b-d)), a, b, d, na, nb, nd)
    t2 = STri(normalize(cross(c-b,b-d)), c, b, d, nc, nb, nd)
    θ = pi / 6
    ϕ = pi / 3
    P(θ, ϕ) = V3(1,0,0)*sin(θ)*cos(ϕ)+V3(0,1,0)*sin(θ)*sin(ϕ) - V3(0,0,1)*cos(θ)
    D(θ, ϕ) = -P(θ, ϕ)
    # ray from point on unit sphere to origin. Dir and pos vary to always point at origin
    R = ADRay(P(θ, ϕ),
               zero(V3),
               ForwardDiff.derivative(θ->P(θ, ϕ), θ),
               ForwardDiff.derivative(ϕ->P(θ, ϕ), ϕ),
               D(θ, ϕ),
               zero(V3),
               ForwardDiff.derivative(θ->D(θ, ϕ), θ),
               ForwardDiff.derivative(ϕ->D(θ, ϕ), ϕ),
               true,
               -1,
               1,
               0.0f0,
               zero(V3),
               false)

    dx1 = ForwardDiff.derivative(x->p(R, t1, 0.0f0, x, 0.0f0), 0.0f0)
    dy1 = ForwardDiff.derivative(y->p(R, t1, 0.0f0, 0.0f0, y), 0.0f0)
    dx2 = ForwardDiff.derivative(x->p(R, t2, 0.0f0, x, 0.0f0), 0.0f0)
    dy2 = ForwardDiff.derivative(y->p(R, t2, 0.0f0, 0.0f0, y), 0.0f0)

    # test cross dx dy is parallel to tri normal
    @test dot(cross(dx1, dy1), V3(0, 0, 1)) ≈ norm(cross(dx1, dy1))
    @test dot(cross(dx2, dy2), V3(0, 0, 1)) ≈ norm(cross(dx2, dy2))
    @test dx1 ≈ dx2
    @test dy1 ≈ dy2

    d0 = refract(ray_dir(R, 0.0f0, 0.0f0, 0.0f0), optical_normal(t1, p(R, t1, 0.0f0, 0.0f0, 0.0f0)), 1.0f0, 2.0f0)
    dx1 = ForwardDiff.derivative(x->refract(ray_dir(R, 0.0f0, x, 0.0f0), optical_normal(t1, p(R, t1, 0.0f0, x, 0.0f0)), 1.0f0, 2.0f0), 0.0f0)
    dy1 = ForwardDiff.derivative(y->refract(ray_dir(R, 0.0f0, 0.0f0, y), optical_normal(t1, p(R, t1, 0.0f0, 0.0f0, y)), 1.0f0, 2.0f0), 0.0f0)
    dx2 = ForwardDiff.derivative(x->refract(ray_dir(R, 0.0f0, x, 0.0f0), optical_normal(t2, p(R, t2, 0.0f0, x, 0.0f0)), 1.0f0, 2.0f0), 0.0f0)
    dy2 = ForwardDiff.derivative(y->refract(ray_dir(R, 0.0f0, 0.0f0, y), optical_normal(t2, p(R, t2, 0.0f0, 0.0f0, y)), 1.0f0, 2.0f0), 0.0f0)
    # test dx and dy perpedicular to center of ray for both tris
    @test dot(cross(dx1, dy1), d0) ≈ norm(cross(dx1, dy1))
    @test dot(cross(dx2, dy2), d0) ≈ norm(cross(dx2, dy2))
    # test derivatives in refracted ray are same for same point on edge of two tris (and non-zero)
    @test dx1 ≈ dx2
    @test dy1 ≈ dy2
    @test norm(dx1) > 0
    @test norm(dy1) > 0
    @test norm(dx2) > 0
    @test norm(dy2) > 0
end

@testset "AD Optics" begin
test1()
test2()
end
