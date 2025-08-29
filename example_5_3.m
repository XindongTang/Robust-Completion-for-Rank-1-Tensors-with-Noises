clear all; clc;

%%% Example 5.4 - Separable 

% Problem dimensions
n1 = 3; n2 = 3; n3 = 4;
Ndim = n1 * n2; 

% Input tensor T with observed entries (separable structure)
T(:, :, 1) = [1.01    2.01    3.02;
              2.03    4.01    6.02;
              3.03    6.04    9.07];

T(:, :, 2) = [0.76    1.51    2.23;
              1.52    3.01    4.52;
              2.22    4.53    6.78];

T(:, :, 3) = [0.24    0.51    0.77;
              0.54    1.05    1.52;
              0.76    1.53    2.26];

T(:, :, 4) = [1.28    2.53    3.76;
              2.54    5.07    7.58;
              3.73    7.51   11.26];

% Observation set Omega: specifies which entries are known
% Format: [row_index, column_index, slice_index]
Omega = [3     1     1;
         3     2     1;
         1     3     1;
         2     1     2;
         1     1     2;
         3     1     3;
         3     2     3;
         1     2     3;
         2     3     3;
         3     3     3;
         1     3     4];

% Extract (i,j) pairs for non-singularity check
ij_pairs = Omega(:, 1:2);

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

%% ========== SOLVE SDP BY SDPNAL+===================
% Convert to SeDuMi format and solve the SDP
[blk, At, C, b] = read_sedumi(AAt', bb, cc, Cone);
OPTIONS.tol = 10^(-6); % Solver tolerance

% Solve the semidefinite program
tic;
[obj, X1, s1, y1, S1, Z1, ybar, v, info, runhist] = sdpnalplus(blk, At, C, b, [], [], [], [], [], OPTIONS);
t1 = toc;


MoMab = S1{1};
svds(MoMab)'; % Display singular values

%% ==================== MOMENT OPTIMIZATION ====================
% Use moment optimization to recover the factors
mpol('z', n1 + n2); 
x = [z(1:n1)];    % First factor
y = [z(n1+1:n1+n2)]; % Second factor

% Build moment constraints matching the SDP solution
K_eq = [];
for i = 1:n1
    for k = i:n1  % k from i (symmetric)
        for j = 1:n2
            for l = j:n2  % l from j (symmetric)
                rowid = (i-1)*n2 + j;
                colid = (k-1)*n2 + l;
                % Moment constraints matching the SDP solution
                eq = [mom(x(i) * y(j) * x(k) * y(l)) - MoMab(rowid, colid) == 0];
                K_eq = [K_eq, eq];
            end
        end
    end
end

% Additional constraints
con_eq = [x'*x - 1 == 0, y'*y - 1 == 0]; % Norm constraints
con_ineq = [ones(n1, 1)'*x >= 0, ones(n2, 1)'*y >= 0]; % Non-negativity

% Objective function for moment optimization (random quadratic)
bracx = mmon(z, 0, 3);
G = randn(length(bracx));
R = bracx' * (G'*G) * bracx;
k_order = 3;

% Solve the moment problem
Problem = msdp(min(mom(R)), K_eq, con_eq, con_ineq, k_order);
[status, obj] = msol(Problem);

%% ==================== EXTRACT ALL SOLUTIONS ====================
% Extract all candidate solutions
double_a = double(x);
double_b = double(y);

% Filter out singular solutions (where a_i * b_j ≈ 0 for some observed (i,j))
anew = [];
bnew = [];
singular_a = [];
singular_b = [];

for i = 1:3
    a = double_a(:, :, i);
    b = double_b(:, :, i);
    is_nonsingular = true;  
    
    % Check if any observed (i,j) pair gives near-zero product
    for j = 1:size(ij_pairs, 1)
        row = ij_pairs(j, :);
        if abs(a(row(1))' * b(row(2))) < 1e-6
            is_nonsingular = false;
            break;  
        end
    end
    
    % Separate singular and non-singular solutions
    if is_nonsingular
        anew = cat(3, anew, a);
        bnew = cat(3, bnew, b);
    else
        singular_a = cat(3, singular_a, a);
        singular_b = cat(3, singular_b, b);
    end
end

if status == 1
    fprintf('\n===== THE COMPUTED G[y*] IS SEPARABLE =====\n');
    fprintf('Number of non-singular solutions found: %d\n', size(anew, 3));
else
    fprintf('\n===== THE COMPUTED G[y*] IS NOT SEPARABLE =====\n');
    fprintf('\n===== TURN TO SPECTRAL DECOMPOSITION FOR FINDING TENSOR COMPLETIONS =====\n');
end

% Display all solutions
fprintf('\n===== ALL SOLUTIONS =====\n');
fprintf('Number of non-singular solutions found: %d\n', size(anew, 3));
fprintf('Number of singular solutions found: %d\n', size(singular_a, 3));

% Display non-singular solutions
if ~isempty(anew)
    fprintf('\n===== NON-SINGULAR SOLUTIONS =====\n');
    for i = 1:size(anew, 3)
        fprintf('\nNon-Singular Solution %d:\n', i);
        fprintf('Factor a:\n'); disp(anew(:, :, i)');
        fprintf('Factor b:\n'); disp(bnew(:, :, i)');
    end
end

% Display singular solutions (without computing c)
if ~isempty(singular_a)
    fprintf('\n===== SINGULAR SOLUTIONS =====\n');
    for i = 1:size(singular_a, 3)
        fprintf('\nSingular Solution %d:\n', i);
        fprintf('Factor a:\n'); disp(singular_a(:, :, i)');
        fprintf('Factor b:\n'); disp(singular_b(:, :, i)');
    end
end

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