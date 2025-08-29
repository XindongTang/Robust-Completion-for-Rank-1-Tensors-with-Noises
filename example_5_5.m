clear all; clc;

%%%% Example 5.5 - Tensor Completion with p* < f*


% Problem dimensions
n1 = 3; n2 = 3; n3 = 7;
Ndim = n1 * n2; 

%% ==================== SYNTHETIC TENSOR GENERATION ====================
% Create a base tensor with all zeros
ax = [0; 0; 0];
by = [0; 0; 0];
cz = [0; 0; 0; 0; 0; 0; 0];
XY = ax * by';
T0 = [];

% Initialize base tensor
for k = 1:n3
    T0(:, :, k) = cz(k) * XY;
end

% Add specific entries to create the test tensor
TN = T0;
TN(1, 1, 1) = 1; TN(2, 2, 1) = 1; TN(3, 3, 1) = 1;      % Diagonal entries in slice 1
TN(1, 2, 2) = 0; TN(2, 3, 3) = 0; TN(3, 1, 4) = 0;      % Zero entries
TN(1, 3, 5) = 0; TN(2, 1, 6) = 0; TN(3, 2, 7) = 0;      % More zero entries
TN(1, 1, 2) = sqrt(3); TN(1, 1, 3) = sqrt(3); TN(1, 1, 4) = sqrt(3); % Special values
TN(1, 1, 5) = 1; TN(1, 1, 6) = 1; TN(1, 1, 7) = 1;      % Unit values

T = T0 + TN; % Final tensor

%% ==================== OBSERVATION SET ====================
% Define which entries are observed
Omega = [1 1 1;
         2 2 1;
         3 3 1;
         1 2 2;
         2 3 3;
         3 1 4;
         1 3 5;
         2 1 6;
         3 2 7;
         1 1 2;
         1 1 3;
         1 1 4;
         1 1 5;
         1 1 6;
         1 1 7];

% Group observations by slice (third dimension)
subset = {};
for i = 1:n3
    om = Omega(:, 3);
    K3 = find((om == i));
    subset{i} = Omega(K3, 1:3);
end

%% ==================== SDP FORMULATION ====================
% Define cone constraints for SDP
Cone.l = 2;    % Linear cone dimension
Cone.s = Ndim; % Semidefinite cone dimension

% Initialize zero matrices
ZM1 = sparse(zeros(n1));
ZM2 = sparse(zeros(n2));
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
                aat = vec(kron(ZMAij, ZMBij)); 
                AAt = [AAt, aat];
            end
        end
    end
end

% Construct the right-hand side vector bb for SDP constraints
bbb = sparse(Ndim, Ndim); 
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
                bij = T(r2(1), r2(2), k3) * kron(ais, bjs) - ...
                      T(r1(1), r1(2), k3) * kron(ait, bjt);
                bbb = bbb + bij * bij'; % Accumulate quadratic terms
            end
        end
    end
end

% Process constraint matrix and right-hand side
bb = (vec(bbb)' * AAt)';
bb_full = bb;

% Create trace constraints for SDP formulation
trace_vec = zeros(size(AAt, 2), 1);
for l = 1:size(AAt, 2)
    if nnz(AAt(:, l)) == 1 % nnz(AAt(:,l))=1 iff y_l is on the diagonal of Gy
        trace_vec(l) = 1;
    end
end

% Augment constraint matrix with trace constraints
AAt_full = [trace_vec'; -trace_vec'; AAt];
cc_full = [-1; 1; zeros(size(AAt, 1), 1)];

%% ========== SOLVE SDP BY SDPNAL+ ====================
% Solve using SeDuMi solver
[x_sdm, y_sdm, info_sdm] = sedumi(AAt_full', bb_full, cc_full, Cone);
MoMab_full = mat(cc_full(3:end) - AAt_full(3:end, :) * y_sdm);

% Solve using SDPNAL+ solver (more robust for larger problems)
[blk, At, C, b] = read_sedumi(AAt_full', bb_full, cc_full, Cone);
OPTIONS.tol = 10^(-6);
[obj, X1, s1, y1, S1, Z1, ybar, v, info, runhist] = sdpnalplus(blk, At, C, b, [], [], [], [], [], OPTIONS);


MoMab = S1{2};
svds(MoMab)'; % Display singular values

%% ==================== MOMENT OPTIMIZATION ====================
% Use moment optimization to recover the factors
mpol('x', n1 + n2);
a = [x(1:n1)];    % First factor
b = [x(n1+1:n1+n2)]; % Second factor

% Build objective function for moment optimization
f = 0;
for k3 = 1:n3
    Om3 = subset{k3};
    for i = 1:size(Om3, 1)-1
        rowid = Om3(i, :);
        for j = i+1:size(Om3, 1)
            colid = Om3(j, :);
            % Form 2x2 determinant constraints
            det22 = (T(rowid(1), rowid(2), k3) * a(colid(1)) * b(colid(2)) - ...
                    T(colid(1), colid(2), k3) * a(rowid(1)) * b(rowid(2)))^2;
            f = f + det22;
        end
    end
end

% Solve the moment optimization problem
PO = msdp(min(f), mom(vec(a*b')' * vec(a*b')) == 1, 3);
[status, fsos] = msol(PO);
fsos; % Optimal value from moment relaxation

if status == 1
    fprintf('\n===== THE COMPUTED G[y*] IS SEPARABLE =====\n');
    fprintf('Number of non-singular solutions found: %d\n', size(anew, 3));
else
    fprintf('\n===== THE COMPUTED G[y*] IS NOT SEPARABLE =====\n');
    fprintf('\n===== TURN TO SPECTRAL DECOMPOSITION FOR FINDING TENSOR COMPLETIONS =====\n\n');
end

%% ==================== FACTOR RECOVERY FROM SDP ====================
% Extract leading eigenvectors from SDP solution
[lmv, lmd] = eigs(MoMab, 4, 'largestabs');

% Recover factors from the 4 leading eigenvectors
lmv1 = lmv(:, 1); lmv2 = lmv(:, 2); lmv3 = lmv(:, 3); lmv4 = lmv(:, 4);

% Extract and normalize factors from first eigenvector
anew1 = lmv1(1:n2:Ndim, 1);
bnew1 = lmv1(1:n2, 1);
anew1 = anew1 / norm(anew1);
bnew1 = bnew1 / norm(bnew1);

% Extract and normalize factors from second eigenvector
anew2 = lmv2(1:n2:Ndim, 1);
bnew2 = lmv2(1:n2, 1);
anew2 = anew2 / norm(anew2);
bnew2 = bnew2 / norm(bnew2);

% Extract and normalize factors from third eigenvector
anew3 = lmv3(1:n2:Ndim, 1);
bnew3 = lmv3(1:n2, 1);
anew3 = anew3 / norm(anew3);
bnew3 = bnew3 / norm(bnew3);

% Extract and normalize factors from fourth eigenvector
anew4 = lmv4(1:n2:Ndim, 1);
bnew4 = lmv4(1:n2, 1);
anew4 = anew4 / norm(anew4);
bnew4 = bnew4 / norm(bnew4);

%% ==================== ERROR EVALUATION ====================
% Use the fourth eigenvector for error evaluation (best solution)
[cnew, err_ab] = error_abs(anew4, bnew4, T, Omega);

% Compute relative  error
diff1 = 0; 
diff2 = 0; 

for i = 1:size(Omega, 1)
    rowid = Omega(i, :);
    diff1 = diff1 + (T(rowid(1), rowid(2), rowid(3)) - ...
                    anew4(rowid(1)) * bnew4(rowid(2)) * cnew(rowid(3)))^2;
    diff2 = diff2 + (T(rowid(1), rowid(2), rowid(3)))^2;
end

err_rel = sqrt(diff1) / sqrt(diff2); % Relative reconstruction error

%% ==================== RESULTS DISPLAY ====================
fprintf('\n===== TENSOR COMPLETION RESULTS =====\n');
fprintf('Problem Dimensions: n1=%d, n2=%d, n3=%d\n', n1, n2, n3);
fprintf('Observation Count: %d entries\n', size(Omega, 1));

fprintf('Moment Relaxation Optimal Value (f*): %.6f\n', fsos);
fprintf('Absolute Error: %.6f\n', err_ab);
fprintf('Relative Error: %.6f\n', err_rel);
fprintf('The computed G[y*] is not separable\n');
% Display the selected solution

fprintf('\nSelected Solution (from 4th eigenvector):\n');
fprintf('Factor a:\n'); disp(anew4');
fprintf('Factor b:\n'); disp(bnew4');
fprintf('Factor c:\n'); disp(cnew');

