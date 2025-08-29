% %%% example 5.8; Table 3
% compare NLS with convex relaxation
% The error shown in the output is (err-abs/||delta A||_Omega)

clc; clear all;

%% Input parameters
% n1,n2,n3 - dimensions for the cubic tensor
% den - the density of observed entries (den = |Omega|/(n1n2n3))
% sigma - the scale of randomly generated noise
% num_experiments - number of total random instances in the experiment
n1 = 11; 
n2 = 12; 
n3 = 10; %%% dimensions
den = 0.30;  %%%% density
sigma = 1e-2; %%%% noise
num_experiments = 10;

%% Initialize arrays to store results
nls_errors = zeros(num_experiments, 1);
sdp_errors = zeros(num_experiments, 1);
nls_times = zeros(num_experiments, 1);
sdp_times = zeros(num_experiments, 1);

for exp_idx = 1:num_experiments
    fprintf('Running experiment %d/%d...\n', exp_idx, num_experiments);
    
    % ==================== SYNTHETIC TENSOR GENERATION ====================
    % Generate random factors for ground truth tensor
    ax = randn(n1, 1);
    ax = ax / norm(ax);
    
    by = randn(n2, 1); 
    by = by / norm(by);
    
    cz = randn(n3, 1);
    
    % Create ground truth tensor T0 = a ⊗ b ⊗ c
    Ndim = n1 * n2;
    XY = ax * by';
    T0 = [];
    for k = 1:n3
        T0(:, :, k) = cz(k) * XY;
    end
    
    % Add Gaussian noise
    noise = sigma * rand(n1, n2, n3);
    TN = T0 .* noise;
    T = T0 + TN;
    
    % ==================== SAMPLING PATTERN GENERATION ====================
    % Generate random observation pattern
    i = 1:n1;
    j = 1:n2;
    k = 1:n3;
    [I, J, K] = meshgrid(i, j, k);
    mesh = [I(:), J(:), K(:)];
    
    OM = ceil(sprand(length(mesh), 3, den));
    [row, col] = find(OM);
    Omrow = mesh(row, 1:3);
    Omsort = sortrows(Omrow, [3, 2]);
    Omega = unique(Omsort, 'rows', 'stable');
    
    % ==================== NONLINEAR LEAST SQUARES (NLS) ====================
    x0 = randn(n1 + n2 + n3, 1); % Random initial point
    
    % Levenberg-Marquardt optimization
    options = optimoptions('lsqnonlin', 'Display', 'off', 'Algorithm', 'levenberg-marquardt');
    
    tic;
    [x_opt, resnorm] = lsqnonlin(@(x) objectiveFunction(x, T, Omega, n1, n2, n3), x0, [], [], options);
    nls_time = toc;
    
    % Extract factors from NLS solution
    a_nls = x_opt(1:n1);
    b_nls = x_opt(n1+1:n1+n2);
    c_nls = x_opt(n1+n2+1:end);
    
    % Compute NLS reconstruction error
    diff_nls1 = 0;
    diff_nls2 = 0;
    for i = 1:size(Omega, 1)
        rowid = Omega(i, :);
        diff_nls1 = diff_nls1 + (T(rowid(1), rowid(2), rowid(3)) - ...
                        a_nls(rowid(1)) * b_nls(rowid(2)) * c_nls(rowid(3)))^2;
        diff_nls2 = diff_nls2 + (TN(rowid(1), rowid(2), rowid(3)))^2;
    end
    err_nls = sqrt(diff_nls1) / sqrt(diff_nls2);
    
    % ==================== CONVEX RELAXATION (SDP)  ====================
    tic;
    [t1, err_sdp] = convex_relaxation(den, n1, n2, n3, sigma);
    sdp_time = toc;
    
    % Store results
    nls_errors(exp_idx) = err_nls;
    sdp_errors(exp_idx) = err_sdp;
    nls_times(exp_idx) = nls_time;
    sdp_times(exp_idx) = sdp_time;
    
end

% ==================== STATISTICAL ANALYSIS ====================
fprintf('\n===== STATISTICAL RESULTS (%d EXPERIMENTS) =====\n', num_experiments);
fprintf('Parameters: n1=%d, n2=%d, n3=%d, den=%.2f, sigma=%.0e\n', n1, n2, n3, den, sigma);

% NLS statistics
nls_min_error = min(nls_errors);
nls_max_error = max(nls_errors);


% SDP statistics
sdp_min_error = min(sdp_errors);
sdp_max_error = max(sdp_errors);

% Display results
fprintf('\nNONLINEAR LEAST SQUARES (NLS):\n');
fprintf('Minimum Error: %.6f\n', nls_min_error);
fprintf('Maximum Error: %.6f\n', nls_max_error);


fprintf('\nCONVEX RELAXATION (SDP):\n');
fprintf('Minimum Error: %.6f\n', sdp_min_error);
fprintf('Maximum Error: %.6f\n', sdp_max_error);



% Save results 
results.nls_errors = nls_errors;
results.sdp_errors = sdp_errors;
results.nls_times = nls_times;
results.sdp_times = sdp_times;
results.parameters.n1 = n1;
results.parameters.n2 = n2;
results.parameters.n3 = n3;
results.parameters.den = den;
results.parameters.sigma = sigma;


% ==================== OBJECTIVE FUNCTION ====================
function F = objectiveFunction(x, T, Omega, n1, n2, n3)
    a = x(1:n1);
    b = x(n1+1:n1+n2);
    c = x(n1+n2+1:end);

    num_samples = size(Omega, 1);
    F = zeros(num_samples, 1);

    for idx = 1:num_samples
        i = Omega(idx, 1);
        j = Omega(idx, 2);
        k = Omega(idx, 3);
        F(idx) = T(i, j, k) - a(i) * b(j) * c(k); 
    end
end