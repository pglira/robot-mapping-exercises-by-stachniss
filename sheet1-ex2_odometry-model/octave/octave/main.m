% This script runs the main loop and calls all the required
% functions in the correct order.
%
% You can disable the plotting or change the number of steps the filter
% runs for to ease the debugging. You should however not change the order
% or calls of any of the other lines, as it might break the framework.
%
% If you are unsure about the input and return values of functions you
% should read their documentation which tells you the expected dimensions.

% Make tools available
addpath('tools');

% Read world data, i.e. landmarks.
landmarks = read_world('../data/world.dat');
% Read sensor readings, i.e. odometry and range-bearing sensor
data = read_data('../data/sensor_data.dat');

% Initialize belief
% x: 3x1 vector representing the robot pose [x; y; theta]
x = zeros(3, 1);

% Iterate over odometry commands and update the robot pose
% according to the motion model
% for t = 1:size(data.timestep, 2)
for t = 1:10

    % Update the pose of the robot based on the motion model
    x = motion_command(x, data.timestep(t).odometry);
    
    data.timestep(t).odometry

    disp(sprintf('Robot pose at t=%d: %.3f, %.3f %.3f', t, x(1), x(2), x(3)));
    
    %Generate visualization plots of the current state
    plot_state(x, landmarks, t, data.timestep(t).sensor);

endfor

% Display the final state estimate
disp("Final robot pose:")
disp("x = "), disp(x)