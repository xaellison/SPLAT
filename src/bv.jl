using Clustering

"""
Shitty bounding volume method
1. Apply k-means on circumcenters of Triangles
2. For each cluster, find the smallest sphere enclosing the cluster's members'
    circumscribing spheres
3. Put each Triangle into each cluster where the circumscribing sphere of the
    triangle overlaps with the bounding sphere of the cluster.
"""
function cluster_fuck(path, N)

    mesh = mesh_to_FTri(load(path))
    # annoying matrix for clustering
    data = foldl(hcat, map(T->Array(circumcenter(T)), mesh))
    result = kmeans(data, N)
    cc_to_cluster = Dict(data[:, n] => v for (n, v) in enumerate(result.assignments))

    # initialize clusters so each Tri in one cluster
    clusters = Dict()
    for (n, T) in enumerate(mesh)
        c = result.assignments[n]
        if ! (c in keys(clusters))
            clusters[c] = Set()
        end
        push!(clusters[c], T)
    end
    ℜ³_centers = ℜ³.(result.centers[1, :], result.centers[2, :], result.centers[3, :])

    radii = [maximum(norm(circumcenter(T) - ℜ³_centers[c]) + circumscribing_sphere(T).radius for T in clusters[c]) for c in 1:N]
    spheres = Sphere.(ℜ³_centers, radii)
    # the clusters need to be updated to include Triangles which ma
    for T in mesh
        for (c, S) in enumerate(spheres)
            # it shouldn't be possible to have an empty cluster at this point but ya never know
            if ! (c in keys(clusters))
                clusters[c] = Set()
            end
            S_T = circumscribing_sphere(T)
            if norm(S_T.origin - S.origin) <= S_T.radius + S.radius
                push!(clusters[c], T)
            end
        end
    end
    @assert foldl(union, values(clusters)) == Set(mesh)
    @info "overcount factor = $(sum(map(length, values(clusters))) / length(mesh))"
    return spheres, clusters
end

@time cluster_fuck("objs\\artemis_smaller.obj", 50);
