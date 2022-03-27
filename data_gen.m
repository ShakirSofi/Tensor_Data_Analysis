function [X, y]= data_gen(size)
% Generate 100 points uniformly distributed in  unit dist and the annulus.
rng(1); % For reproducibility
r = sqrt(rand(size,1)); % Radius
t = 2*pi*rand(size,1);  % Angle
data1 = [r.*cos(t), r.*sin(t), r.*exp(t)]; % Points

r2 = sqrt(5*rand(size,1)+1); % Radius
t2 = 5*pi*rand(size,1);      % Angle
data2 = [r2.*cos(t2), r2.*sin(t2), r.*exp(t) ]; % points

X = [data1;data2];
y = ones(size*2,1);
y(1:size) = 0;
end

