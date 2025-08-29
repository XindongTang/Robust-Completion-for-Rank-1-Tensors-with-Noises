clc; clear all;

%% Input parameters
% n - dimension for the tensor (n1=n2=n3=n)
% den - the density of observed entries (den = |Omega|/(n1n2n3))
% sigma - the scale of randomly generated noise
% num_experiments - number of total random instances in the experiment
n = 11;
den = 0.12;
sigma = 1e-4;
num_experiments = 20;

%% Run experiments
nonzero_counts = zeros(num_experiments, 1); % Store number of non-zero elements for each experiment

for exp_idx = 1:num_experiments
    % Call function to get top 6 largest singular values
    value_sig = rank_percent(n, den, sigma);
    
    % Count number of non-zero elements (considering tolerance 1e-6)
    nonzero_count = sum(abs(value_sig) > 1e-6);
    nonzero_counts(exp_idx) = nonzero_count;
end

%% Statistical analysis
fprintf('\n===== Statistical Results =====\n');
fprintf('Total experiments: %d\n', num_experiments);

% Calculate percentages for different cases
one_nonzero = sum(nonzero_counts == 1) / num_experiments * 100;
two_nonzero = sum(nonzero_counts == 2) / num_experiments * 100;
three_or_more = sum(nonzero_counts >= 3) / num_experiments * 100;

fprintf('Percentage of rank-1 G[y^*]: %.1f%%\n', one_nonzero);
fprintf('Percentage of rank-2 G[y^*]: %.1f%%\n', two_nonzero);
fprintf('Percentage of rank>=3 G[y^*]: %.1f%%\n', three_or_more);



