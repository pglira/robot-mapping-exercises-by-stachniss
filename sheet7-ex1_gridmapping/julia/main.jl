# Coordinate frames:
# - world frame: global cartesian coordinate frame
# - map frame  : world frame reduced by local origin in x and y; z unchanged
# - grid frame : same local origin as map frame; consists of cells, i.e. x and y must be integer 
#                values; z unchanged; origin corresponds to lower left corner of lower left cell

import JSON
using Logging
using PyPlot
using Printf

function get_options()

    opt = Dict("datafile" => "csail.json", # path to data
               "prob_prior" => 0.5, # initial grid occupancy probability
               "prob_occ" => 0.9, # probability of occupied cells
               "prob_free" => 0.35, # probability of free cells
               "grid_cellsize" => 0.1, # cell size of grid
               "grid_border" => 30, # border of grid with respect to limits of robot pose
               "laser_max_dist" => 30, # maximum distance of points used for mapping
               "img_dir" => "img", # dir for storing of images used to generate gif file
               "giffile" => "grid.gif") # path to output gif file

    return opt
end

function get_robot_limits(data)

    x_min = Inf
    x_max = -Inf
    y_min = Inf
    y_max = -Inf

    for i = 1:length(data)
        if (data[i]["pose"][1] < x_min) x_min = data[i]["pose"][1] end
        if (data[i]["pose"][1] > x_max) x_max = data[i]["pose"][1] end
        if (data[i]["pose"][2] < y_min) y_min = data[i]["pose"][2] end
        if (data[i]["pose"][2] > y_max) y_max = data[i]["pose"][2] end
    end

    return x_min, x_max, y_min, y_max
end

function define_grid(data, grid_border, grid_cellsize)

    robot_x_min, robot_x_max, robot_y_min, robot_y_max = get_robot_limits(data)

    grid_limits = [floor((robot_x_min-grid_border)/grid_cellsize)*grid_cellsize
                    ceil((robot_x_max+grid_border)/grid_cellsize)*grid_cellsize
                   floor((robot_y_min-grid_border)/grid_cellsize)*grid_cellsize
                    ceil((robot_y_max+grid_border)/grid_cellsize)*grid_cellsize]

    grid_origin = [grid_limits[1], grid_limits[3]];

    return grid_limits, grid_origin
end

function read_trajectory(data)

    no_poses = length(data)
    trj = Array{Float64}(undef, 3, no_poses)
    for i = 1:no_poses
        trj[1,i] = data[i]["pose"][1]
        trj[2,i] = data[i]["pose"][2]
        trj[3,i] = data[i]["pose"][3]
    end

    return trj
end

function prob_to_log_odds(p)
# Convert proability values p to the corresponding log odds l
    l = log(p/(1-p))
end

function initialize_grid(grid_limits, grid_cellsize, prob_prior)

    grid_dx = grid_limits[2]-grid_limits[1]
    grid_dy = grid_limits[4]-grid_limits[3]
    grid_rows = round(Int, grid_dy/grid_cellsize)
    grid_cols = round(Int, grid_dx/grid_cellsize)
    log_odds_prior = prob_to_log_odds(prob_prior)
    grid = log_odds_prior .* ones(grid_rows, grid_cols)
    @info "Grid initialized with ...\n" *
          "limits: $grid_limits\n" *
          "rows x cols: $grid_rows x $grid_cols"

    return grid
end

function pose_to_hmatrix(v)
# Computes the homogeneous transformation matrix A from the pose vector v
    c = cos(v[3])
    s = sin(v[3])
    A = [c -s v[1]
	       s  c v[2]
	       0  0 1]

    return A
end

function world_to_map(pts_worldframe, map_origin)

    size(pts_worldframe)[1] == 3 || 
        error("The point array must be of size (3,n) but is $(size(pts_worldframe))")
    no_points = size(pts_worldframe)[2]
    pts_mapframe = fill(NaN, 3, no_points)
    for i = 1:no_points
        pts_mapframe[:,i] = [pts_worldframe[1,i]-map_origin[1]
                             pts_worldframe[2,i]-map_origin[2]
                             pts_worldframe[3,i]]
    end

    return pts_mapframe
end

function map_to_grid(pts_mapframe, cellsize)

    size(pts_mapframe)[1] == 3 || 
        error("The point array must be of size (3,n) but is $(size(pts_mapframe))")
    no_points = size(pts_mapframe)[2]
    pts_grid = fill(NaN, 3, no_points)
    for i = 1:no_points
        pts_grid[:,i] = [floor(pts_mapframe[1,i]/cellsize)+1 # col index
                         floor(pts_mapframe[2,i]/cellsize)+1 # row index
                         pts_mapframe[3,i]]
    end

    return pts_grid
end

function laser_polar_to_cartesian(data; max_dist=Inf)
# Note: I skipped the consideration of the laser offset, as it is always very, very close to zero
#       (10^-16 ... 10^-18)

    ranges = convert(Array{Float64}, data["ranges"])
    num_beams = length(ranges)
    max_dist = min(max_dist, data["maximum_range"])
    idx_valid = ranges .< max_dist

    angles = range(data["start_angle"], data["start_angle"] + num_beams*data["angular_resolution"],
                   length=num_beams)[idx_valid]

    x = ranges[idx_valid].*cos.(angles)
    y = ranges[idx_valid].*sin.(angles)

    # Create 3-by-n array
    X = [reshape(x, 1, length(x))
         reshape(y, 1, length(y))
         ones(1, sum(idx_valid))]

    return X
end

function bresenheim(x1::Int, y1::Int, x2::Int, y2::Int)
# From https://stackoverflow.com/a/40274836

    # Calculate distances
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep == true
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    end

    # Swap start and end points if necessary and store swap state
    swapped = false
    if x1 > x2
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = true
    end

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = round(Int, dx/2.0)

    if y1 < y2
        ystep = 1
    else
        ystep = -1
    end

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in x1:(x2+1)
        if is_steep == true
            coord = (y, x)
        else
            coord = (x, y)
        end
        push!(points,coord)
        error -= abs(dy)

        if error < 0
            y += ystep
            error += dx
        end
    end

    # Reverse the list if the coordinates were swapped
    if swapped == true
        points = points[end:-1:1]
    end

    return points
end

function inverse_sensor_model(grid, data, grid_cellsize, grid_origin, prob_prior, prob_occ, 
                              prob_free, max_dist)
        
    pose_worldframe = convert(Array{Float64}, reshape(data["pose"], 3, 1)) # 3-by-1 array

    # Initialize grid update
    grid_update = zeros(size(grid))

    # Robot pose as homogeneous tranformation matrix
    H_pose_worldframe = pose_to_hmatrix(pose_worldframe)

    pose_mapframe = world_to_map(pose_worldframe, grid_origin)
    pose_gridframe = map_to_grid(pose_mapframe, grid_cellsize)

    @debug "Robot pose in ...\n" * 
           "world frame: $pose_worldframe\n" * 
           "map frame: $pose_mapframe\n" *
           "grid frame: $pose_gridframe"

    X_laser_localframe = laser_polar_to_cartesian(data; max_dist=max_dist)

    # Transform the local laser points into the grid frame
    X_laser_worldframe = H_pose_worldframe*X_laser_localframe;
    X_laser_mapframe = world_to_map(X_laser_worldframe, grid_origin)
    X_laser_gridframe = map_to_grid(X_laser_mapframe, grid_cellsize)

    # Log odds values
    log_odds_prior = prob_to_log_odds(prob_prior)
    log_odds_occ = prob_to_log_odds(prob_occ)
    log_odds_free = prob_to_log_odds(prob_free)

    # Set log_odds_free for all cells along the laser rays
    for i = 1:size(X_laser_gridframe)[2]
        cells_along_ray = bresenheim(Int(pose_gridframe[1]), Int(pose_gridframe[2]),
                                     Int(X_laser_gridframe[1,i]), Int(X_laser_gridframe[2,i]))
        for j = 1:length(cells_along_ray)
            col = cells_along_ray[j][1]
            row = cells_along_ray[j][2]
            grid_update[row, col] = log_odds_free
        end
    end

    # Set log_odds_occ for endpoints of the laser rays, i.e. the laser points
    for i = 1:size(X_laser_gridframe)[2]
        col = Int(X_laser_gridframe[1,i])
        row = Int(X_laser_gridframe[2,i])
        grid_update[row, col] = log_odds_occ
    end
    
    grid_update = grid_update - log_odds_prior*ones(size(grid))

    return grid_update, pose_gridframe, X_laser_gridframe
end

function plot_grid(grid, trj, pts, img_dir, i)

    cla() # otherwise very slow
    matshow(grid.*-1, fignum=0, cmap=ColorMap("gray"), origin="lower", vmin=-1, vmax=1)
    plot(trj[1,1:i], trj[2,1:i], "-g", linewidth=1) # trajectory
    plot(trj[1,i], trj[2,i], "og", markersize=2) # current pose
    plot(pts[1,:], pts[2,:], ".r", markersize=0.1)

    if i == 1
        if !isdir(img_dir)
            mkdir(img_dir)
        end
        sleep(0.5) # if omitted, first image might have a different resolution (not maximized)
    end

    xticks([])
    yticks([])
    img_path = joinpath(img_dir, @sprintf "grid%04d" i)
    savefig(img_path, bbox_inches="tight", dpi=150, quality=95)

    return nothing
end

function create_gif(img_dir, giffile)

    run(`ffmpeg -i $img_dir/grid0001.png -vf palettegen=16 palette.png`)
    run(`ffmpeg -pattern_type glob -i $img_dir/*.png -i palette.png -filter_complex 
        "fps=20,scale=600:-1:flags=lanczos[x];[x][1:v]paletteuse" $giffile`)

    return nothing
end

function main()

    opt = get_options()

    # Set up logger
    logger = ConsoleLogger(stdout, Logging.Debug)
    global_logger(logger)

    # Read data
    data = JSON.parsefile(opt["datafile"]; dicttype=Dict, inttype=Int64, use_mmap=true)

    # Initialize grid
    grid_limits, grid_origin = define_grid(data, opt["grid_border"], opt["grid_cellsize"])
    grid = initialize_grid(grid_limits, opt["grid_cellsize"], opt["prob_prior"])

    # Read robot trajectory and transform to grid frame (only for plotting)
    trj_worldframe = read_trajectory(data)
    trj_mapframe = world_to_map(trj_worldframe, grid_origin)
    trj_gridframe = map_to_grid(trj_mapframe, opt["grid_cellsize"])

    for i = 1:length(data)

        @info "Processing data chunk $i of $(length(data))"

        grid_update, pose_gridframe, X_laser_gridframe = inverse_sensor_model(grid, data[i], 
            opt["grid_cellsize"], grid_origin, opt["prob_prior"], opt["prob_occ"], 
            opt["prob_free"], opt["laser_max_dist"])

        grid = grid + grid_update

        plot_grid(grid, trj_gridframe, X_laser_gridframe, opt["img_dir"], i)

    end

    close("all")
    create_gif(opt["img_dir"], opt["giffile"])

    return nothing
end

main()
