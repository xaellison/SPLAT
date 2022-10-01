
abstract type AbstractLight end

struct RectLight <: AbstractLight
    center::ℜ³
    dir::ℜ³
    dim1::ℜ³
    dim2::ℜ³
    res1::Int
    res2::Int
end


# Functions that generate rays: camera + light sources

function camera_ray(camera, height, width, x, y, λ, dv)




    dir(x, y, dv) = begin
        scale = height * _COS_45 / camera.FOV_half_sin
        _x, _y = (x - height / 2) / scale, (y - width / 2) / scale
        _z = sqrt(1 - _x^2 - _y^2)
        return _x * camera.right +
               _y * camera.up +
               _z * camera.dir +
               dv * 0.025f0 / max(height, width)
   end
    idx = (y - 1) * width + x
    return ADRay(
        camera.pos,
        zero(ℜ³),
        zero(ℜ³),
        zero(ℜ³),
        dir(x, y, dv),
        zero(ℜ³),
        ForwardDiff.derivative(x -> dir(x, y, dv), x),
        ForwardDiff.derivative(y -> dir(x, y, dv), y),
        false,
        1,
        idx,
        λ,
        0.0f0,
        0.0f0,
        RAY_STATUS_ACTIVE,
    )
end



function wrap_ray_gen(ray_generator, height, width)
    row_indices = CuArray(1:height)
    col_indices = reshape(CuArray(1:width), (1, width))
    dv = CUDA.rand(ℜ³, height, width)
    # TODO: generalize to λ as broadcastable input, not hardcoded
    rays = ray_generator.(row_indices, col_indices, 550.0f0, dv)
    rays = reshape(rays, length(rays))
    return rays
end

function rays_from_light(light::RectLight, upscale)

    function rect_light_ray(x, y, λ, dv)
        # returns a rectangular cross-section, unidirectional light source
        center, dir, δ1, δ2, height, width =
            light.center, light.dir, light.dim1, light.dim2, light.res1 ÷ upscale, light.res2 ÷ upscale
        origin(x, y) =
            center +
            δ1 * (x - height ÷ 2) / (height ÷ 2) +
            δ2 * (y - width ÷ 2) / (width ÷ 2) +
            cross(dv, δ1) ./ (height ÷ 2) +
            cross(dv, δ2) ./ (width ÷ 2)
        return ADRay(
            origin(x, y),
            zero(ℜ³),
            ForwardDiff.derivative(x -> origin(x, y), x),
            ForwardDiff.derivative(y -> origin(x, y), y),
            dir,
            zero(ℜ³),
            zero(ℜ³),
            zero(ℜ³),
            false,
            1,
            0, # lights are forward tracing, dest not known ahead of time
            λ,
            x,
            y,
            RAY_STATUS_ACTIVE,
        )
    end

    return wrap_ray_gen(rect_light_ray, light.res1 ÷ upscale, light.res2 ÷ upscale)
end

function rays_from_lights(lights::AbstractArray{T}, upscale) where {T<:AbstractLight}
    return foldl(vcat, rays_from_light(light, upscale) for light in lights)
end
