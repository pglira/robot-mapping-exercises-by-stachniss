include("../../common-julia-functions/common-functions.jl")

using DataFrames
using CSV
using Plots
using LinearAlgebra
using Printf

function compute_sigma_points(mu, sigma, scale)

    n = length(mu)
    lambda = scale - n

    # Compute all sigma points, see slide 11
    sigma_points = mu # first point is mean
    square_root_of_matrix = sqrt((n+lambda)*sigma)
    # It happens that the resulting matrix is complex, even tough the imaginary parts are all zero
    if square_root_of_matrix isa Array{Complex{Float64},2}
        square_root_of_matrix = extract_real_part_from_complex_matrix(square_root_of_matrix)
    end
    for i = 1:n
        new_sigma_point_1 = mu + square_root_of_matrix[:,i]
        new_sigma_point_2 = mu - square_root_of_matrix[:,i]
        sigma_points = [sigma_points new_sigma_point_1 new_sigma_point_2]
    end

    # Compute weight vectors w_m (we don't use w_c here)
    w_m = fill(NaN, 1, 2*n+1)
    w_m[1] = lambda/(n+lambda)
    for i = 2:2*n+1
        w_m[i] = 1/(2*(n+lambda))
    end

    return sigma_points, w_m
end

function extract_real_part_from_complex_matrix(complex_matrix)

    real_matrix = fill(NaN, size(complex_matrix)[1], size(complex_matrix)[2])

    for i = 1:size(complex_matrix)[1]
        for j = 1:size(complex_matrix)[2]
            real_matrix[i,j] = real(complex_matrix[i,j])
        end
    end

    return real_matrix
end

function recover_gaussian(sigma_points, w_m, isangle)

    no_sigma_points = size(sigma_points)[2]
    dim = size(sigma_points)[1] # dimension

    # Compute mu
    mu = fill(NaN, dim, 1)
    for i = 1:dim
        if ~isangle[i] # standard case
            mu[i] = sum(w_m.*sigma_points[i:i,:])
        else # consider averaging of angles
            x_mean = sum(w_m.*cos.(sigma_points[i:i,:]))
            y_mean = sum(w_m.*sin.(sigma_points[i:i,:]))
            mu[i] = normalize_angle(atan(y_mean, x_mean))
        end
    end
    # mu = sum(w_m.*sigma_points, dims=2) # does not consider averaging of angles

    # Compute sigma
    sigma_points_minus_mean = sigma_points - repeat(mu,1,no_sigma_points)
    for i = 1:dim
        if isangle[i]
            sigma_points_minus_mean[i,:] = normalize_angle.(sigma_points_minus_mean[i,:])
        end
    end
    sigma = zeros(dim,dim)
    for i = 1:no_sigma_points
        sigma = sigma +
                w_m[i] .* (sigma_points_minus_mean[:,i] * transpose(sigma_points_minus_mean[:,i]));
    end

    return mu, sigma
end

function get_first_estimate_of_landmark(x, Sig_x, z, Q, scale)
# See also "Unscented transformation" in Förstner & Wrobel, 2016, p.47

    no_prm = length(x)

    # Add z and Q to mu and sigma, resp.
    mu = [x
          z]

    sigma = [Sig_x           zeros(no_prm,2)
             zeros(2,no_prm) Q]

    sigma_points, w_m = compute_sigma_points(mu, sigma, scale)
    sigma_points[3,:] = normalize_angle.(sigma_points[3,:])

    # Transformation
    x_lm_sigma_points = sigma_points[1:1,:] + sigma_points[end-1:end-1,:] .*
                        cos.(sigma_points[3:3,:].+sigma_points[end:end,:])
    y_lm_sigma_points = sigma_points[2:2,:] + sigma_points[end-1:end-1,:] .*
                        sin.(sigma_points[3:3,:].+sigma_points[end:end,:])

    # Replace z sigma points with sigma points of lm coordinates
    sigma_points[end-1,:] = x_lm_sigma_points
    sigma_points[end,:] = y_lm_sigma_points

    no_lm = Int((length(mu)-3)/2)
    isangle = [false false true repeat([false false], 1, no_lm)]
    x, Sig_x = recover_gaussian(sigma_points, w_m, isangle)

    return x, Sig_x
end

function get_Sig_xz(sigma_points_x, sigma_points_z, z_exp, w_m)

    sigma_points_x_minus_mean = sigma_points_x - repeat(sigma_points_x[:,1],1,no_sigma_points)
    sigma_points_x_minus_mean[3,:] = normalize_angle.(sigma_points_x_minus_mean[3,:])
    # sigma_points_z_minus_mean = sigma_points_z - repeat(sigma_points_z[:,1],1,no_sigma_points) # wrong!
    sigma_points_z_minus_mean = sigma_points_z - repeat(z_exp,1,no_sigma_points)
    sigma_points_z_minus_mean[2,:] = normalize_angle.(sigma_points_z_minus_mean[2,:])
    no_rows = size(sigma_points_x_minus_mean)[1]
    no_cols = size(sigma_points_z_minus_mean)[1]
    Sig_xz = zeros(no_rows, no_cols)
    for i = 1:no_sigma_points
        Sig_xz = Sig_xz +
                 w_m[i] .*
                 (sigma_points_x_minus_mean[:,i] * transpose(sigma_points_z_minus_mean[:,i]));
    end

    return Sig_xz
end

function prediction_step(x, Sig_x, odo, scale)

    no_prm = length(x)

    # Todo Can this be avoided?
    r1 = odo.r1[1]
    t = odo.t[1]
    r2 = odo.r2[1]

    # Get sigma points
    sigma_points, w_m = compute_sigma_points(x, Sig_x, scale)

    # Update state vector x by transforming sigma points
    # 3 first rows correspond to x, y, theta of the pose of the robot
    sigma_points[1,:] = sigma_points[1,:] .+ t*cos.(sigma_points[3,:].+r1)
    sigma_points[2,:] = sigma_points[2,:] .+ t*sin.(sigma_points[3,:].+r1)
    sigma_points[3,:] = normalize_angle.(sigma_points[3,:].+r1.+r2)

    # Recover sigma from sigma points
    isangle = falses(no_prm,1)
    isangle[3] = true
    x, Sig_x = recover_gaussian(sigma_points, w_m, isangle)
    x[3] = normalize_angle(x[3])

    # Define motion noise
    R = zeros(3, 3)
    motion_noise = 0.1
    R[1,1] = motion_noise
    R[2,2] = motion_noise
    R[3,3] = motion_noise/10

    # Add motion noise to covariance matrix Sig_x
    Sig_x[1:3,1:3] = Sig_x[1:3,1:3]+R

    return x, Sig_x
end

function correction_step(x, Sig_x, lm, sen, scale)

    # For each sensor observation
    for i = 1:nrow(sen)

        # Observation vector z
        z = [sen.range[i]
             normalize_angle(sen.bearing[i])]

        # Sensor noise Q
        Q = [0.01 0
             0    0.01]

        # If new landmarks are observed, add them to x, Sig_x and save their indices in lm
        if lm.x_idx[sen.id[i]] == 0 # sen.id[i] corresponds to row in lm

            x, Sig_x = get_first_estimate_of_landmark(x, Sig_x, z, Q, scale)

            # Add it to lm
            lm.x_idx[sen.id[i]] = length(x)-1
            lm.y_idx[sen.id[i]] = length(x)

        end

        # Create sigma points for state vector
        sigma_points_x, w_m = compute_sigma_points(x, Sig_x, scale)
        sigma_points_x[3,:] = normalize_angle.(sigma_points_x[3,:])
        no_sigma_points = size(sigma_points_x)[2]

        # Create sigma points for the observations on the basis of the current state vector x
        sigma_points_z = fill(NaN, length(z), no_sigma_points)

        x_idx = filter(row -> row.id == sen.id[i], lm).x_idx[1]
        y_idx = filter(row -> row.id == sen.id[i], lm).y_idx[1]

        dx = sigma_points_x[x_idx,:] .- sigma_points_x[1,:]
        dy = sigma_points_x[y_idx,:] .- sigma_points_x[2,:]

        sigma_points_z[1,:] = sqrt.(dx.^2+dy.^2)
        sigma_points_z[2,:] = normalize_angle.(atan.(dy, dx) .- sigma_points_x[3,:])

        isangle = [false true]
        z_exp, Sig_z_exp = recover_gaussian(sigma_points_z, w_m, isangle)

        S_t = Sig_z_exp + no_sigma_points*Q
        # S_t = Sig_z_exp + Q # wrong!

        Sig_xz = get_Sig_xz(sigma_points_x, sigma_points_z, z_exp, w_m)

        # Kalman gain K
        K = Sig_xz/S_t

        dz = z-z_exp
        dz[2] = normalize_angle(dz[2])

        x = x+K*dz
        x[3] = normalize_angle(x[3])

        Sig_x = Sig_x - K*S_t*K'

    end

    return x, Sig_x, lm
end

function update_lm_positions_from_state_vector(lm, x)

    for i = 1:nrow(lm)
        if lm.x_idx[i] != 0
            lm.x[i] = x[lm.x_idx[i]]
            lm.y[i] = x[lm.y_idx[i]]
        else
            lm.x[i] = NaN
            lm.y[i] = NaN
        end
    end

    return lm
end

function get_C_lm(lm, Sig_x)
    C_lm = []
    for i = 1:nrow(lm)
        x_idx = lm.x_idx[i]
        y_idx = lm.y_idx[i]
        if x_idx != 0
            push!(C_lm, Sig_x[x_idx:y_idx,x_idx:y_idx])
        else
            push!(C_lm, [])
        end
    end
    return C_lm
end

function main()
# Todo Short explanation

    lm = read_landmarks("world.dat")
    no_lm = nrow(lm)
    lm[:x_idx] = 0 # index of x landmark coordinate in status vector x
    lm[:y_idx] = 0 # index of y landmark coordinate in status vector x

    odo, sen = read_sensor_observations("sensor_data.dat")
    # Parameter for unscented transformation
    scale = 3

    # Initialize belief for state parameters with robot pose x, y, theta
    x = zeros(3,1)

    # Initialize covariance matrix of state parameters
    Sig_x = 0.001*Matrix(I,3,3)

    anim = Animation()

    x0_path = Array{Float64}(undef,0,2)

    for (i, timestamp) in enumerate(odo.timestamp)

        x, Sig_x = prediction_step(x, Sig_x, odo[odo.timestamp .== timestamp, :], scale)

        x, Sig_x, lm = correction_step(x, Sig_x, lm, sen[sen.timestamp .== timestamp, :], scale)

        lm = update_lm_positions_from_state_vector(lm, x)

        x0_path = [x0_path; x[1:2]']

        plot_state!(x[1:3], Sig_x[1:3, 1:3], x0_path, lm, get_C_lm(lm, Sig_x),
                    sen[sen.timestamp .== timestamp, :], timestamp, (-5, 15), (-2.5, 12.5))

        frame(anim)

        println("Robot pose at time=$timestamp: $(x[1:3])")

    end

    gif(anim, "state.gif", fps=15)

    return nothing
end

main()
