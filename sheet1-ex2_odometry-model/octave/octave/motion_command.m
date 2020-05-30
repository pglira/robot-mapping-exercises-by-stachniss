function [x] = motion_command(x, u)
% Updates the robot pose according to the motion model
% x: 3x1 vector representing the robot pose [x; y; theta]
% u: struct containing odometry reading (r1, t, r2).
% Use u.r1, u.t, and u.r2 to access the rotation and translation values

% Update x according to the motion represented by u
x(1) = x(1) + u.t*cos(x(3) + u.r1);
x(2) = x(2) + u.t*sin(x(3) + u.r1);
x(3) = normalize_angle(x(3) + u.r1 + u.r2);

end
