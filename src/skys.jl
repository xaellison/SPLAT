
function sky_color(dir, λ, phi) ::Float32
	d = normalize(dir)
	out = 0.1+0.7*d[3] ^ 2
	if isnan(out)
		return 0.0f0
	end
	return out
end


function sky_color_rezz(dir, λ, phi)
	t1 = atan(dir[1], sqrt(dir[2]^2+dir[3]^2))
	t2 = atan(dir[2], sqrt(dir[1]^2+dir[3]^2))
	t3 = atan(dir[3], sqrt(dir[1]^2+dir[2]^2))
	if t1 < 0
		t1 -= phi
	else
		t1 += phi
	end
	if t2 < 0
		t2 -= phi
	else
		t2 += phi
	end
	if t3 < 0
		t3 -= phi
	else
		t3 += phi
	end
	if t1 < 0
		t1 += pi
	end
	if t2 < 0
		t2 += pi
	end
	if t3 < 0
		t3 += pi
	end
	k = 2*pi/4
	if xor(abs(t1) % k < k/2, abs(t2) % k < k/2, abs(t3) % k < k/2)
		return 1.0
	end
	return 0.0
end
