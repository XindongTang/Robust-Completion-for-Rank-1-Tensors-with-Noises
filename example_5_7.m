clc; clear all;
%%%% example 5.7,
% Table 2: The performance of the convex relaxation(4.4)
% The error shown in the output is (err-abs/||delta A||_Omega)

%% Input parameters
% n1,n2,n3 - dimensions for the cubic tensor
% den - the density of observed entries (den = |Omega|/(n1n2n3))
% sigma - the scale of randomly generated noise
% num_experiments - number of total random instances in the experiment
den = 0.23;
n1 = 10;
n2 = 10;
n3 = 10;
sigma = 1e-2;
num_experiments = 20;

%% Calculate additional outputs
dimy = n1 * (n1 + 1) * n2 * (n2 + 1) / 4;
lenG_y = n1 * n2;

%% Initialize arrays to store results
time_results = zeros(num_experiments, 1);
err_rat_results = zeros(num_experiments, 1);

%% Run experiments
for exp_idx = 1:num_experiments
    fprintf('Running experiment %d/%d...\n', exp_idx, num_experiments);
    
    % Call the convex_relaxation function
    [t1, err_rat] = convex_relaxation(den, n1, n2, n3, sigma);
    
    % Store results
    time_results(exp_idx) = t1;
    err_rat_results(exp_idx) = err_rat;
end

% Calculate statistics
avg_time = mean(time_results);
max_err_rat = max(err_rat_results);
min_err_rat = min(err_rat_results);
std_time = std(time_results);
std_err_rat = std(err_rat_results);

% Display comprehensive results
fprintf('\n===== Statistical Analysis (20 Experiments) =====\n');
fprintf('Parameters: den=%.2f, n1=%d, n2=%d, n3=%d, sigma=%.0e\n', den, n1, n2, n3, sigma);
fprintf('dimy = %d, lenG[y] = %d\n', dimy, lenG_y);

% Display all individual results
fprintf('\n===== Individual Experiment Results =====\n');
for exp_idx = 1:num_experiments
    fprintf('Case %2d: Time = %7.4f s, Err_Rat = %9.6f\n', ...
            exp_idx, time_results(exp_idx), err_rat_results(exp_idx));
end

fprintf('\nTime Statistics:\n');
fprintf('Average Time: %.4f seconds\n', avg_time);



fprintf('\nError Ratio Statistics:\n');
fprintf('Minimum Error Ratio: %.6f\n', min_err_rat);
fprintf('Maximum Error Ratio: %.6f\n', max_err_rat);
