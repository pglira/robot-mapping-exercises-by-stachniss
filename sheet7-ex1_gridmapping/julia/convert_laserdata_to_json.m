% Import laser data to struct
laser = read_robotlaser('csail.log')

% Convert struct to json
pkg load io % install with pkg install -forge io
json = object2json(laser);

% Write json string to file
fid = fopen('csail.json', 'wt');
fprintf(fid, '%s', json);
fclose(fid);
