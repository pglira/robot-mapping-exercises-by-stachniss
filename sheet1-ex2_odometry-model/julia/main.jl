include("../../common-julia-functions/common-functions.jl")

using DataFrames
using CSV
using Plots

function motion_command(x, odo)

    # Todo Can this be avoided?
    r1 = odo.r1[1]
    t = odo.t[1]
    r2 = odo.r2[1]

    x[1] = x[1] + t*cos(x[3]+r1)
    x[2] = x[2] + t*sin(x[3]+r1)
    x[3] = normalize_angle(x[3] + r1 + r2)

    return x
end

function main()
# We start with the robot pose x=0, y=0, theta=0. For each timestep the odometry motion (lines
# with "ODOMETRY" in "sensor_data.dat" containing "rot1", "translation", and "rot2") is added to
# the current state of the robot.
# The landmarks ("world.dat") and the observations to the landmarks (lines with "SENSOR" in
# "sensor_data.dat" containing "id", "range", and "bearing") are not considered at all! The
# landmarks and its observations are visualized for educational purposes only.

    lm = read_landmarks("world.dat")
    odo, sen = read_sensor_observations("sensor_data.dat")

    # Initialize belief for robot pose x, y, theta
    x = zeros(3,1)

    anim = Animation()

    x0_path = Array{Float64}(undef,0,2)

    for timestamp in odo.timestamp

        x = motion_command(x, odo[odo.timestamp .== timestamp, :])

        x0_path = [x0_path; x[1:2]']

        plot_state!(x[1:3], [], x0_path, lm, [], sen[sen.timestamp .== timestamp, :], timestamp,
                    (-5, 15), (-2.5, 12.5))

        frame(anim)

        println("Robot pose at time=$timestamp: $x")

    end

    gif(anim, "state.gif", fps=15)

    return nothing
end

main()
