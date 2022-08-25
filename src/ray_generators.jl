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
    return ADRay(
        camera.pos,
        zero(V3),
        dir,
        zero(V3),
        false,
        1,
        idx,
        λ,
        RAY_STATUS_ACTIVE,
    )
end

function simple_light(center, dir, δ1, δ2, height, width, x, y, λ, dv)
    # returns a rectangular cross-section, unidirectional light source
    origin = center + δ1 * (x - height ÷ 2) / (height ÷ 2) + δ2 * (y - width ÷ 2) / (width ÷ 2) + dv ./ (width ÷ 2)
    return ADRay(origin,
                 zero(V3),
                 dir,
                 zero(V3),
                 false,
                 1,
                 0, # lights are forward tracing, dest not known ahead of time
                 λ,
                 RAY_STATUS_ACTIVE
                 )
end

function wrap_ray_gen(ray_generator; rays, row_indices, col_indices, dv, kwargs...)
    height = length(row_indices)
    width = length(col_indices)
    @assert length(rays) == width * height
    dv .= V3.(CUDA.rand(Float32, height, width), CUDA.rand(Float32, height, width), CUDA.rand(Float32, height, width))
    rays = reshape(rays, height, width)
    rays .= ray_generator.(row_indices, col_indices, 550.0, dv)
    rays = reshape(rays, height * width)
end
