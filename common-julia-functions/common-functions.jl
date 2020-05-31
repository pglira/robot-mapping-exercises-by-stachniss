using LinearAlgebra

function read_landmarks(path_to_file)

    lm = DataFrame(CSV.File("world.dat", header=false, types=[Int, Float64, Float64]))
    rename!(lm, [:id, :x, :y])

    return lm

end

function read_sensor_observations(path_to_file)
# Splits the sensor observations into an odometry and a sensor data frame. For each observation a
# timestamp is added.

    # Initialize data frames for odometry and sensor messages
    odo = DataFrame(timestamp=Int[], r1=Float64[], t=Float64[], r2=Float64[])
    sen = DataFrame(timestamp=Int[], id=Int[], range=Float64[], bearing=Float64[])

    # Read all sensor observations
    obs = DataFrame(CSV.File(path_to_file, header=false))
    rename!(obs, [:source, :obs1, :obs2, :obs3])

    timestamp = 0;
    for i = 1:nrow(obs)
        if obs[i,:source] == "ODOMETRY"
            timestamp = timestamp + 1
            push!(odo, (timestamp, obs[i,:obs1], obs[i,:obs2], obs[i,:obs3]))
        else
            push!(sen, (timestamp, obs[i,:obs1], obs[i,:obs2], obs[i,:obs3]))
        end
    end

    return odo, sen
end

function normalize_angle(angle)
# Normalize angle to be between -pi and pi

    while angle > pi
        angle = angle - 2*pi;
    end

    while angle < -pi
        angle = angle + 2*pi;
    end

    return angle
end

function get_ellipse_parameters_from_covariance_matrix(C)
# From "Ausgleichsrechnung II", p.54: "Helmert'sche Fehlerellipse"

    sxx = C[1,1]
    syy = C[2,2]
    sxy = C[1,2]

    w = sqrt((sxx^2-syy^2)^2+4*sxy^2)
    a = real(sqrt(Complex(1/2*(sxx^2+syy^2+w))))
    b = real(sqrt(Complex(1/2*(sxx^2+syy^2-w))))
    alpha = 1/2*atan(2*sxy/(sxx^2-sxy^2))

    return a, b, alpha
end

function get_ellipse_points(x0, y0, a, b, alpha; scale=1)

    no_pts = 100;

    beta = collect(0:2*pi/no_pts:2*pi)
    x = a*cos.(beta)*scale
    y = a*sin.(beta)*scale

    # Rotate
    x = cos.(alpha)*x - sin.(alpha)*y
    y = sin.(alpha)*x + cos.(alpha)*y

    # Translate
    x = x .+ x0
    y = y .+ y0

    return x, y
end

function plot_state!(anim, x0, C_x0, x0_path, lm, C_lm, sen, timestamp, x_lims, y_lims)

    gr()

    # Landmarks
    scatter((lm.x, lm.y), markershape=:star, markersize=5, markercolor=:yellow, label="landmarks", 
            aspect_ratio=:equal)

    # Landmarks error ellipses
    if ~isempty(C_lm)
        for i = 1:length(C_lm)
            a, b, alpha = get_ellipse_parameters_from_covariance_matrix(C_lm[i])
            x_ell, y_ell = get_ellipse_points(lm.x[i], lm.y[i], a, b, alpha)
            plot!((x_ell, y_ell), color=:red, label="")
        end
    end

    # Real sensor observations to landmarks
    for i = 1:nrow(sen)
        range = sen[i, :range]
        bearing = sen[i, :bearing]
        dx = range*cos(x0[3]+bearing)
        dy = range*sin(x0[3]+bearing)
        plot!([x0[1]; x0[1]+dx], [x0[2]; x0[2]+dy], color=:gray, label="")
    end

    # Fake sensor observations to landmarks given the robot pose
    for i = 1:nrow(sen)
        lm_id = sen[i,:id]
        lm_x = lm[lm_id,:x]
        lm_y = lm[lm_id,:y]
        plot!([x0[1]; lm_x], [x0[2]; lm_y], color=:black, label="")
    end

    # Full path of robot pose
    if ~isempty(x0_path)
        plot!(x0_path[:,1], x0_path[:,2], color=:gray, label="path")
    end

    # Robot pose
    scatter!((x0[1], x0[2]), markersize=4, markercolor=:green, label="robot")

    # Robot error ellipse
    if ~isempty(C_x0)
        a, b, alpha = get_ellipse_parameters_from_covariance_matrix(C_x0)
        x_ell, y_ell = get_ellipse_points(x0[1], x0[2], a, b, alpha)
        plot!((x_ell, y_ell), color=:red, label="")
    end

    # Robot orientation
    ori = [cos(x0[3]) sin(x0[3])]./norm([cos(x0[3]) sin(x0[3])]) # normalize length
    quiver!([x0[1]], [x0[2]], quiver=([ori[1]], [ori[2]]), color=:green)

    title!("Robot pose, landmarks and sensor observations (t=$timestamp)")
    xlims!(x_lims)
    ylims!(y_lims)
    frame(anim)

end
