using Clustering

"""
Shitty bounding volume method.
1. Apply k-means on circumcenters of Triangles  This gives you cluster locations, 
    and triangle memberships.
2. For each cluster, find the smallest sphere enclosing the cluster's members'
    circumscribing spheres
3. Put each Triangle into each cluster where the circumscribing sphere of the
    triangle overlaps with the bounding sphere of the cluster.
"""
function cluster_fuck(tris, N)
    @info "Clustering $(length(tris)) tris"
    # annoying matrix for clustering
    data = foldl(hcat, map(T->Array(circumcenter(T)), tris))
    result = kmeans(data, N)
    
    # initialize clusters so each Tri in one cluster
    clusters = Dict()
    for (n, T) in enumerate(tris)
        c = result.assignments[n]
        if ! (c in keys(clusters))
            clusters[c] = Set()
        end
        push!(clusters[c], T)
    end
    ℜ³_centers = ℜ³.(result.centers[1, :], result.centers[2, :], result.centers[3, :])
    @info Set(result.assignments)
    radii = [maximum(norm(circumcenter(T) - ℜ³_centers[c]) + circumscribing_sphere(T).radius for T in clusters[c]) for c in 1:N]
    spheres = Sphere.(ℜ³_centers, radii)
    sort!(spheres)
    # the clusters need to be updated to include Triangles which ma
    for T in tris
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
    @assert foldl(union, values(clusters)) == Set(tris)
    @info "overcount factor = $(sum(map(length, values(clusters))) / length(tris))"

    inverted_indices = Dict(T => n for (n, T) in enumerate(tris))
    clusters_by_index = Dict()

    for c in keys(clusters)
        # sort should help the gpu have better aligned accesses in certain instances (eg, two well separated models)
        clusters_by_index[c] = sort([inverted_indices[T] for T in clusters[c]])
    end

    return spheres, clusters_by_index
end

function load_and_cluster_fuck(path::String, N)
    tris = mesh_to_FTri(load(path))
    cluster_fuck(tris, N)
end

#@time load_and_cluster_fuck("objs\\artemis_smaller.obj", 3);
