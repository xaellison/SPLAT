
abstract type AbstractLight end

struct RectLight <: AbstractLight
    center :: ℜ³
    dir :: ℜ³
    dim1 :: ℜ³
    dim2 :: ℜ³
    res1 :: Int
    res2 :: Int
end


# Functions that generate rays: camera + light sources

function camera_ray(camera, height, width, x, y, λ, dv)
    _x, _y = x - height / 2, y - width / 2
    scale = height * _COS_45 / camera.FOV_half_sin
    _x /= scale
    _y /= scale

    _z = sqrt(1 - _x^2 - _y^2)
    dir =
        _x * camera.right +
        _y * camera.up +
        _z * camera.dir +
        dv * 0.25f0 / max(height, width)
    dir = normalize(dir)
    idx = (y - 1) * height + x
    return ADRay(camera.pos, zero(ℜ³), dir, zero(ℜ³), false, 1, idx, λ, RAY_STATUS_ACTIVE)
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

function rays_from_light(light :: RectLight)

    function rect_light_ray(x, y, λ, dv)
        # returns a rectangular cross-section, unidirectional light source
        center, dir, δ1, δ2, height, width = light.center, light.dir, light.dim1, light.dim2, light.res1, light.res2
        origin =
            center +
            δ1 * (x - height ÷ 2) / (height ÷ 2) +
            δ2 * (y - width ÷ 2) / (width ÷ 2) +
            cross(dv, δ1) ./ (height ÷ 2) +
            cross(dv, δ2) ./ (width ÷ 2)
        return ADRay(
            origin,
            zero(ℜ³),
            dir,
            zero(ℜ³),
            false,
            1,
            0, # lights are forward tracing, dest not known ahead of time
            λ,
            RAY_STATUS_ACTIVE,
        )
    end

    return wrap_ray_gen(rect_light_ray, light.res1, light.res2)
end

function rays_from_lights(lights :: AbstractArray{T}) where {T <: AbstractLight}
    return foldl(vcat, rays_from_light(light) for light in lights)
end
