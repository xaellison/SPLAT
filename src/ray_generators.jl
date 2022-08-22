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
