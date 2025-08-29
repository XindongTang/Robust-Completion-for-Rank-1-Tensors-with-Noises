clear all; clc;

%%% Example 5.2 
% Problem dimensions
n1 = 10; n2 = 10; n3 = 10;
Ndim = n1 * n2; 

%% ==================== SYNTHETIC TENSOR GENERATION ====================
% Generate a rank-1 ground truth tensor: T0(i,j,k) = sin(i)*cos(j)*sin(k)
T0 = zeros(n1, n2, n3);
for i1 = 1:n1
    for i2 = 1:n2
        for i3 = 1:n3
            T0(i1, i2, i3) = sin(i1) * cos(i2) * sin(i3);
        end
    end
end

% Add Gaussian noise to the tensor
noise = 1e-4 * rand(n1, n2, n3); % Small noise level
TN = T0 .* noise; 
T = T0 + TN; % Noisy tensor

%% ==================== OBSERVATION PATTERN GENERATION ====================
% Create observation pattern: sample 1/3 of the entries
% Pattern: select entries where (i+j+k) mod 3 == 0
Omega = zeros(n1 * n2 * n3, 3); 
idx = 1; 

for i = 1:n1
    for j = 1:n2
        for k = 1:n3
            if mod(i + j + k, 3) == 0
                Omega(idx, :) = [i, j, k]; 
                idx = idx + 1; 
            end
        end
    end
end

% Remove unused rows from Omega
Omega(idx:end, :) = [];

% Group observations by slice (third dimension)
subset = {};
for i = 1:n3
    om = Omega(:, 3);
    K3 = find((om == i));
    subset{i} = Omega(K3, 1:3);
end

%% ==================== SDP FORMULATION ====================
% Generate constraint matrix for primal SDP formulation
Cone.s = Ndim; % Cone dimension for semidefinite programming

% Initialize zero matrices
ZM1 = zeros(n1, n1);
ZM2 = zeros(n2, n2);
AAt = []; % Constraint matrix

% Construct the constraint matrix AAt using Kronecker products
for i1 = 1:n1
    for j1 = i1:n1
        for i2 = 1:n2
            for j2 = i2:n2
                % Create symmetric matrices for Kronecker product
                ZMAij = ZM1; ZMAij(i1, j1) = 1; ZMAij(j1, i1) = 1;
                ZMBij = ZM2; ZMBij(i2, j2) = 1; ZMBij(j2, i2) = 1;
                
                % Form Kronecker product and vectorize
                aat = vec(sparse_kron(ZMAij, ZMBij)); 
                AAt = [AAt, aat];
            end
        end
    end
end
AAt = sparse(AAt); % Convert to sparse for efficiency

% Construct the right-hand side vector bb for SDP constraints
bbb = zeros(Ndim, Ndim); 
for k3 = 1:n3
    Om3 = subset{k3}; % Observations in current slice
    if size(Om3, 1) > 1
        for i = 1:size(Om3, 1)
            r1 = Om3(i, :);
            for j = i+1:size(Om3, 1)
                r2 = Om3(j, :);
                is = r1(1); js = r1(2);
                it = r2(1); jt = r2(2);
                
                % Create basis vectors
                ais = zeros(n1, 1); ait = zeros(n1, 1);
                bjs = zeros(n2, 1); bjt = zeros(n2, 1);
                ais(is) = 1; ait(it) = 1;
                bjs(js) = 1; bjt(jt) = 1;
                
                % Form constraint based on observed values
                bij = T(r2(1), r2(2), k3) * sparse_kron(ais, bjs) - ...
                      T(r1(1), r1(2), k3) * sparse_kron(ait, bjt);
                bbb = bbb + bij * bij'; % Accumulate quadratic terms
            end
        end
    end
end

% Process constraint matrix and right-hand side for SDP formulation
bb = (vec(bbb)' * AAt)';
val_b1 = bb(1);
vecidenN = sparse(vec(eye(Ndim)));
bb = -(bb(2:length(bb)) - val_b1 * (vecidenN' * AAt(:, 2:size(AAt, 2)))'); 
cc = AAt(:, 1); 

AAt = -(AAt(:, 2:size(AAt, 2)) - AAt(:, 1) * (vecidenN' * AAt(:, 2:size(AAt, 2)))); 

%% ================SOLVE SDP BY SDPNAL+====================
% Convert to SeDuMi format and solve the SDP
[blk, At, C, b] = read_sedumi(AAt', bb, cc, Cone);
OPTIONS.tol = 10^(-6); % Solver tolerance

% Solve the semidefinite program
tic;
[obj, X1, s1, y1, S1, Z1, ybar, v, info, runhist] = sdpnalplus(blk, At, C, b, [], [], [], [], [], OPTIONS);
t1 = toc;


MoMab = S1{1};
svds(MoMab)'; % Display singular values 

%% ==================== FACTOR RECOVERY ====================
% Recover factors anew and bnew from MoMab
[lmv, lmd] = eigs(MoMab, 1, 'largestabs'); 
anew = lmv(1:n2:Ndim, 1); 
bnew = lmv(1:n2, 1); 

% Normalize factors
anew = anew / norm(anew);
bnew = bnew / norm(bnew);

% Calculate cnew and absolute absolute error
[cnew, err_ab] = error_abs(anew, bnew, T, Omega);


% Compute relative error
diff1 = 0; 
diff2 = 0; 

for i = 1:size(Omega, 1)
    rowid = Omega(i, :);
    diff1 = diff1 + (T(rowid(1), rowid(2), rowid(3)) - ...
                    anew(rowid(1)) * bnew(rowid(2)) * cnew(rowid(3)))^2;
    % Total squared magnitude of observed values
    diff2 = diff2 + (T(rowid(1), rowid(2), rowid(3)))^2;
end

err_rel = sqrt(diff1) / sqrt(diff2); 

%% ==================== ERROR EVALUATION (NON-SINGULAR ONLY) ====================
if ~isempty(anew)
    % Calculate cnew and absolute error only for non-singular solutions
    [cnew, err_ab] = error_abs(anew, bnew, T, Omega);

    % Compute relative error
    diff1 = 0; 
    diff2 = 0; 

    for i = 1:size(Omega, 1)
        rowid = Omega(i, :);
        
        diff1 = diff1 + (T(rowid(1), rowid(2), rowid(3)) - ...
                        anew(rowid(1)) * bnew(rowid(2)) * cnew(rowid(3)))^2;
        
        diff2 = diff2 + (T(rowid(1), rowid(2), rowid(3)))^2;
    end

    err_rel = sqrt(diff1) / sqrt(diff2); % Relative error

    %% ==================== FINAL RESULTS DISPLAY ====================
    fprintf('\n===== FINAL TENSOR COMPLETION RESULTS =====\n');
    fprintf('Problem Dimensions: n1=%d, n2=%d, n3=%d\n', n1, n2, n3);
    fprintf('Observation Count: %d entries\n', size(Omega, 1));
    fprintf('Computation Time: %.4f seconds\n', t1);

    fprintf('Absolute Error: %.6f\n', err_ab);
    fprintf('Relative Error: %.6f\n', err_rel);

    % Display the selected non-singular solution and third factor
    fprintf('\nThe (nonsingular) rank-1 tensor completion is:\n');
    fprintf('Factor a:\n'); disp(anew(:, :, 1)');
    fprintf('Factor b:\n'); disp(bnew(:, :, 1)');
    fprintf('Factor c:\n'); disp(cnew');
else
    fprintf('\nNo non-singular solutions found! Only displaying singular solutions.\n');
end