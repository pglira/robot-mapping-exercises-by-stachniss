include("../../common-julia-functions/common-functions.jl")

using DataFrames
using CSV
using Plots
using LinearAlgebra

function prediction_step(x, Sig_x, odo)

    no_prm = length(x)

    # Todo Can this be avoided?
    r1 = odo.r1[1]
    t = odo.t[1]
    r2 = odo.r2[1]

    # Update state vector x
    x[1] = x[1] + t*cos(x[3]+r1)
    x[2] = x[2] + t*sin(x[3]+r1)
    x[3] = normalize_angle(x[3] + r1 + r2)

    # Jakobian G
    G = zeros(no_prm, no_prm)
    G[1,1] = 1
    G[1,2] = 0
    G[1,3] = -t*sin(x[3]+r1)
    G[2,1] = 0
    G[2,2] = 1
    G[2,3] = t*cos(x[3]+r1)
    G[3,1] = 0
    G[3,2] = 0
    G[3,3] = 1
    for i = 1:size(G)[1]
        G[i,i] = 1 # landmarks coordinates are unchanged my motion update
    end

    R = zeros(no_prm, no_prm)
    motion_noise = 0.1
    R[1,1] = motion_noise
    R[2,2] = motion_noise
    R[3,3] = motion_noise/10

    # Update covariance matrix Sig_x
    Sig_x = G*Sig_x*G'+R

    return x, Sig_x
end

function correction_step(x, Sig_x, lm_obs, sen)

    no_prm = length(x)
    no_obs = nrow(sen)*2

    # Set initial values of landmarks which are observed for the first time
    for i = 1:nrow(sen)
        if ~lm_obs[sen.id[i]] # if landmark was not observed in the past
            x[2*sen.id[i]+2] = x[1] + sen.range[i] * cos(sen.bearing[i]+x[3])
            x[2*sen.id[i]+3] = x[2] + sen.range[i] * sin(sen.bearing[i]+x[3])
            lm_obs[sen.id[i]] = true
        end
    end

    # Observation vector z as sequence of observed range, bearing
    z = fill(NaN, no_obs, 1)
    for i = 1:nrow(sen)
        z[2*i-1] = sen.range[i]
        z[2*i] = sen.bearing[i]
    end

    # Expected observation vector z_exp derived from current state vector x
    z_exp = fill(NaN, no_obs, 1)
    for i = 1:nrow(sen)
        dx = x[2*sen.id[i]+2] - x[1];
        dy = x[2*sen.id[i]+3] - x[2];
        z_exp[2*i-1] = sqrt(dx^2+dy^2)
        z_exp[2*i] = normalize_angle(atan(dy, dx) - x[3])
    end

    # Jakobian H
    H = zeros(no_obs, no_prm)
    for i = 1:nrow(sen)

        dx = x[2*sen.id[i]+2] - x[1];
        dy = x[2*sen.id[i]+3] - x[2];

        # Derivation of range and bearing w.r.t. x[1]
        H[2*i-1, 1] = 1/2*(dx^2+dy^2)^(-1/2)*(-2*dx)
        H[2*i, 1] = dy/(dx^2+dy^2)

        # Derivation of range and bearing w.r.t. x[2]
        H[2*i-1, 2] = 1/2*(dx^2+dy^2)^(-1/2)*(-2*dy)
        H[2*i, 2] = -dx/(dx^2+dy^2)

        # Derivation of range and bearing w.r.t. x[3]
        H[2*i-1, 3] = 0
        H[2*i, 3] = -1

        # Derivation of range and bearing w.r.t landmark x coordinate, i.e. x[4], x[6], ...
        H[2*i-1, 2*sen.id[i]+2] = 1/2*(dx^2+dy^2)^(-1/2)*(2*dx)
        H[2*i, 2*sen.id[i]+2] = -dy/(dx^2+dy^2)

        # Derivation of range and bearing w.r.t landmark y coordinate, i.e. x[5], x[7], ...
        H[2*i-1, 2*sen.id[i]+3] = 1/2*(dx^2+dy^2)^(-1/2)*(2*dy)
        H[2*i, 2*sen.id[i]+3] = dx/(dx^2+dy^2)

    end

    # Difference dz
    dz = z-z_exp
    for i = 2:2:length(z)
        dz[i] = normalize_angle(dz[i])
    end

    # Sensor noise Q
    Q = 0.01*Matrix(I, no_obs, no_obs)

    # Kalman gain K
    K = Sig_x*H'/(H*Sig_x*H'+Q)

    x = x+K*dz
    x[3] = normalize_angle(x[3])

    Sig_x = (Matrix(I, no_prm, no_prm)-K*H)*Sig_x

    return x, Sig_x, lm_obs
end

function update_lm_positions_from_state_vector(lm, x)

    for i = 1:nrow(lm)
        lm.x[i] = x[2*i+2]
        lm.y[i] = x[2*i+3]
    end

    return lm
end

function main()
# Todo Short explanation

    lm = read_landmarks("world.dat")
    no_lm = nrow(lm)
    lm_obs = falses(no_lm,1) # true if lm was already observed by sensor

    odo, sen = read_sensor_observations("sensor_data.dat")

    # Initialize belief for state parameters:
    # - robot pose x, y, theta (elements 1:3)
    # - landmark positions x, y (elements 3:3+no_landmarks*2)
    x = zeros(3+no_lm*2,1)
    no_prm = length(x)

    # Initialize covariance matrix of state parameters
    Sig_x = zeros(no_prm, no_prm)
    for i = 4:size(Sig_x)[1]
        Sig_x[i,i] = 1000 # used instead of infinity
    end

    anim = Animation()

    for timestamp in odo.timestamp

        x, Sig_x = prediction_step(x, Sig_x, odo[odo.timestamp .== timestamp, :])

        x, Sig_x, lm_obs = correction_step(x, Sig_x, lm_obs, sen[sen.timestamp .== timestamp, :])

        lm = update_lm_positions_from_state_vector(lm, x)

        plot_state!(anim, x, Sig_x[1:2, 1:2], lm, sen[sen.timestamp .== timestamp, :], timestamp,
                    (-5, 15), (-2.5, 12.5))

        println("Robot pose at time=$timestamp: $(x[1:3])")

    end

    gif(anim, "state.gif", fps=15)

    return nothing
end

main()
