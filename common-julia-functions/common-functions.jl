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

function plot_state!(anim, x, lm, sen, timestamp)

    gr()

    x_margin = 5
    y_margin = 5

    # Landmarks
    scatter((lm.x, lm.y), markershape=:star, markersize=5, markercolor=:yellow, label="landmarks")

    # Real sensor observations to landmarks
    for i = 1:nrow(sen)
        range = sen[i, :range]
        bearing = sen[i, :bearing]
        dx = range*cos(x[3]+bearing)
        dy = range*sin(x[3]+bearing)
        plot!([x[1]; x[1]+dx], [x[2]; x[2]+dy], color=:gray, label="")
    end

    # Fake sensor observations to landmarks given the robot pose
    for i = 1:nrow(sen)
        lm_id = sen[i,:id]
        lm_x = lm[lm_id,:x]
        lm_y = lm[lm_id,:y]
        plot!([x[1]; lm_x], [x[2]; lm_y], color=:black, label="")
    end

    # Robot pose
    scatter!((x[1], x[2]), markersize=5, markercolor=:red, label="robot")

    title!("Robot pose, landmarks and sensor observations (t=$timestamp)")
    xlims!((minimum(lm.x)-x_margin,maximum(lm.x)+x_margin))
    ylims!((minimum(lm.y)-y_margin,maximum(lm.y)+y_margin))
    frame(anim)

end
