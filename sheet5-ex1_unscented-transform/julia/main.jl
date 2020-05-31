include("../../common-julia-functions/common-functions.jl")

using DataFrames
using CSV
using Plots
using LinearAlgebra

function compute_sigma_points(mu, sigma, lambda, alpha, beta)

    n = length(mu)

    # Compute all sigma points, see slide 11
    sigma_points = mu # first point is mean
    square_root_of_matrix = sqrt((n+lambda)*sigma)
    for i = 1:n
        new_sigma_point_1 = mu + square_root_of_matrix[:,i];
        new_sigma_point_2 = mu - square_root_of_matrix[:,i];
        sigma_points = [sigma_points new_sigma_point_1 new_sigma_point_2];
    end

    # Compute weight vectors w_m and w_c, see slide 17
    w_m = lambda/(n+lambda)
    w_c = w_m[1] + (1-alpha^2+beta)
    for i = 1:2*n
        new_w_mc = 1/(2*(n+lambda))
        w_m = [w_m new_w_mc]
        w_c = [w_c new_w_mc]
    end

    return sigma_points, w_m, w_c
end

function transform(points)

    # Function 1 (linear)
    # Applies a translation to [x; y]
    # points[1,:] = points[1,:] .+ 1
    # points[2,:] = points[2,:] .+ 2

    # Function 2 (nonlinear)
    # Computes the polar coordinates corresponding to [x; y]
    # x = points[1,:]
    # y = points[2,:]
    # r = fill(NaN, 1, size(points)[2])
    # theta = fill(NaN, 1, size(points)[2])
    # for i = 1:size(points)[2]
    #     r[i] = sqrt(x[i]^2 + y[i]^2)
    #     theta[i] = atan(y[i], x[i])
    # end
    # points = [r
    #           theta]

    # Function 3 (nonlinear)
    points[1,:] = points[1,:].*cos.(points[1,:]).*sin.(points[1,:]);
    points[2,:] = points[2,:].*cos.(points[2,:]).*sin.(points[2,:]);

    return points
end

function recover_gaussian(sigma_points, w_m, w_c)

    n = size(sigma_points)[1]

    # Compute mu
    mu = sum(w_m.*sigma_points, dims=2)

    # Compute sigma
    sigma_points_minus_mean = [sigma_points[1,:]' .- mu[1]
                               sigma_points[2,:]' .- mu[2]]
    sigma = zeros(2,2)
    for i = 1:2*n+1
        sigma = sigma +
                w_c[i] .* (sigma_points_minus_mean[:,i] * transpose(sigma_points_minus_mean[:,i]));
    end

    return mu, sigma
end

function plot_distribution(mu, sigma, sigma_points; scale=1, color=:black, reuse=false, label="")

    # Plot mu
    if reuse # reuse as argument to scatter does not work with gr backend
        scatter!((mu[1], mu[2]), markersize=4, markercolor=color, label=label, aspect_ratio=:equal)
    else
        scatter((mu[1], mu[2]), markersize=4, markercolor=color, label=label, aspect_ratio=:equal)
    end

    # Plot error ellipse
    a, b, alpha = get_ellipse_parameters_from_covariance_matrix(sigma)
    x_ell, y_ell = get_ellipse_points(mu[1], mu[2], a, b, alpha, scale=scale)
    plot!((x_ell, y_ell), color=color, label="")

    # Plot sigma_points
    scatter!((sigma_points[1,:], sigma_points[2,:]), markersize=4, markershape = :cross,
             markercolor=color, label="")

    return nothing
end

function main()

    # Initial distribution
    sigma = [0.1 0
             0   0.1]
    mu = [1
          2]
    n = length(mu)

    # Compute lambda
    alpha = 0.9
    beta = 2
    kappa = 1
    lambda = alpha^2*(n+kappa)-n

    # Compute the sigma points corresponding to mu and sigma
    sigma_points, w_m, w_c = compute_sigma_points(mu, sigma, lambda, alpha, beta)

    # Plot original distribution with sampled sigma points
    plot_distribution(mu, sigma, sigma_points; scale=3, color=:red, label="original distribution")

    # Transform sigma points
    sigma_points_trans = transform(sigma_points)

    # Recover mu and sigma of the transformed distribution
    mu_trans, sigma_trans = recover_gaussian(sigma_points_trans, w_m, w_c)

    # Plot transformed sigma points with corresponding mu and sigma
    plot_distribution(mu_trans, sigma_trans, sigma_points_trans; scale=3, color=:blue, reuse=true,
                     label="transformed distribution")

    # Save plot
    title!("Unscented transformation")
    png("unscented_transform.png")

    return nothing
end

main()
