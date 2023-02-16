clear;

%% settings
global MU_0;
global EPS_0;
global XI;
global W;
MU_0 = 1.25663706212e-6;                         %  permeability in H/m
EPS_0 = 8.8541878128e-12;                        %  permittivity in F/m
C = sqrt(1 / (MU_0 * EPS_0));                    %  speed of light in m/s
N = 51;                                          %  number of elements
XI = [-0.8611363115940526, -0.3399810435848563, ...
        0.3399810435848563, 0.8611363115940526]; %  integration points
W = [0.3478548451374538, 0.6521451548625461, ...
        0.6521451548625461, 0.3478548451374538]; %  integration weights
V_in = 1;                                        %  voltage of the source
P_out = 10 / 1000;                               %  output power in W

frequencies = [10., 30., 90.];                   %  frequencies in GHz

%% simulation 1
output = zeros((N + 1) * length(frequencies), 7);
rel_idx = 1;
tic;
for idx = 1:length(frequencies)
    f = frequencies(idx) * 1e9;
    lambda = C / f;
    L = lambda / 2;
    r = L / N / 10;
    [I, x] = solver(N, f, L, r, V_in);
    Z_in = V_in / I(length(I)/2);
    V_out = sqrt(2 * P_out * real(Z_in));
    I = V_out / V_in * I;
    output(rel_idx:rel_idx+N, 1) = repelem(N, N + 1)';
    output(rel_idx:rel_idx+N, 2) = repelem(f, N + 1)';
    output(rel_idx:rel_idx+N, 3) = repelem(L, N + 1)';
    output(rel_idx:rel_idx+N, 4) = repelem(V_out, N + 1)';
    output(rel_idx:rel_idx+N, 5:7) = [x', real(I)', imag(I)'];
    rel_idx = rel_idx + N + 1;
end
disp('Run finalized');
toc;

%% simulation 2
% output = zeros((N + 1) * length(frequencies), 7);
% rel_idx = 1;
% tic;
% for idx = 1:length(frequencies)
%     f = frequencies(idx) * 1e9;
%     lambda = C / f;
%     L = lambda / 2;
%     r = L / N / 10;
%     [I, x] = solver(N, f, L, r, V_in);
%     P = 1 / 2 * real(V_in * (conj(I(length(I)/2))));
%     scaler = P_out / P;
%     I = sqrt(scaler) * I;
%     output(rel_idx:rel_idx+N, 1) = repelem(N, N + 1)';
%     output(rel_idx:rel_idx+N, 2) = repelem(f, N + 1)';
%     output(rel_idx:rel_idx+N, 3) = repelem(L, N + 1)';
%     output(rel_idx:rel_idx+N, 4) = repelem(V_out, N + 1)';
%     output(rel_idx:rel_idx+N, 5:7) = [x', real(I)', imag(I)'];
%     rel_idx = rel_idx + N + 1;
% end
% disp('Run finalized');
% toc;

%% save simulation
save('half-wavelength-dipole-10mW.mat', 'output');