clear all; clc;

%%% Example 5.4 - Exact Completion with Rank=2, Separable Case
% This example demonstrates exact tensor completion for a rank-2 separable tensor

% Problem dimensions
n1 = 3; n2 = 3; n3 = 3;
Ndim = n1 * n2; % Dimension of the flattened matrix

% Define the incomplete tensor T with only specific entries known
% Format: T(row, column, slice) = value
T = zeros(n1, n2, n3); % Initialize empty tensor

% Set known entries in tensor T
T(1,3,1) = 4;
T(1,3,3) = 4;
T(2,1,3) = 1;
T(1,1,2) = 4;
T(2,3,2) = 16;
T(1,2,2) = 4;
T(3,1,2) = 2;
T(2,1,1) = 1;
T(3,2,2) = 2;
T(3,3,3) = 2;
T(3,3,1) = 2;
T(2,2,1) = 1;
T(2,2,3) = 1;

% Observation set Omega: specifies which entries are known
% Format: [row, column, slice]
Omega = [1 3 1;
         1 3 3;
         2 1 3;
         1 1 2;
         2 3 2;
         1 2 2;
         3 1 2;
         2 1 1;
         3 2 2;
         3 3 3;
         3 3 1;
         2 2 1;
         2 2 3];

% Group observations by slice (third dimension)
subset = {};
for i = 1:n3
    om = Omega(:,3);
    K3 = find((om == i));
    subset{i} = Omega(K3, 1:3);
end

%% ==================== SDP FORMULATION ====================
Cone.s = Ndim; % Cone dimension for semidefinite programming

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
                ZMAij = ZM1; ZMAij(i1,j1) = 1; ZMAij(j1,i1) = 1;
                ZMBij = ZM2; ZMBij(i2,j2) = 1; ZMBij(j2,i2) = 1;
                
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
        for i = 1:size(Om3,1)
            r1 = Om3(i, :);
            for j = i+1:size(Om3, 1)
                r2 = Om3(j, :);
                is = r1(1); js = r1(2);
                it = r2(1); jt = r2(2);
                
                % Create basis vectors
                ais = zeros(n1,1); ait = zeros(n1,1);
                bjs = zeros(n2,1); bjt = zeros(n2,1);
                ais(is) = 1; ait(it) = 1;
                bjs(js) = 1; bjt(jt) = 1;
                
                % Form constraint based on observed values
                bij = T(r2(1),r2(2),k3) * kron(ais, bjs) - ...
                      T(r1(1),r1(2),k3) * kron(ait, bjt);
                bbb = bbb + bij * bij'; % Accumulate quadratic terms
            end
        end
    end
end

% Process constraint matrix and right-hand side for SDP formulation
bb = (vec(bbb)' * AAt)';
val_b1 = bb(1);
vecidenN = sparse(vec(eye(Ndim)));
bb = -(bb(2:length(bb)) - val_b1 * (vecidenN' * AAt(:,2:size(AAt,2)))'); 
cc = AAt(:,1); 

AAt = -(AAt(:,2:size(AAt,2)) - AAt(:,1) * (vecidenN' * AAt(:,2:size(AAt,2)))); 

%% ================SOLVE SDP BY SDPNAL+====================
% Convert to SeDuMi format and solve the SDP
[blk, At, C, b] = read_sedumi(AAt', bb, cc, Cone);
OPTIONS.tol = 10^(-6); % Solver tolerance

% Solve the semidefinite program bysdpnal+
tic;
[obj, X1, s1, y1, S1, Z1, ybar, v, info, runhist] = sdpnalplus(blk, At, C, b, [], [], [], [], [], OPTIONS);
t1 = toc;


MoMab = S1{1};
svds(MoMab)'; % Display singular values

%% ==================== MOMENT OPTIMIZATION ====================
% Use moment optimization to recover the factors
mpol('z', n1+n2); 
x = [z(1:n1)];   
y = [z(n1+1:n1+n2)]; 

% Build moment constraints
K_eq = [];
for i = 1:n1
    for k = i:n1  
        for j = 1:n2
            for l = j:n2  
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
con_ineq = [ones(n1,1)'*x >= 0, ones(n2,1)'*y >= 0]; % Non-negativity

% Objective function for moment optimization
bracx = mmon(z, 0, 3);
G = randn(length(bracx));
R = bracx' * (G'*G) * bracx;
k_order = 3;

% Solve the moment problem
Problem = msdp(min(mom(R)), K_eq, con_eq, con_ineq, k_order);
[status, obj] = msol(Problem);

%% ==================== EXTRACT AND EVALUATE RESULTS ====================
% Extract the recovered factors
double_a = double(x);
double_b = double(y);

anew = double_a(:,:,:);
bnew = double_b(:,:,:);


% Evaluate reconstruction quality
[cnew(:,:,1), err_ab1] = error_abs(anew(:,:,1), bnew(:,:,1), T, Omega);
[cnew(:,:,2), err_ab2] = error_abs(anew(:,:,2), bnew(:,:,2), T, Omega);

if status == 1
    fprintf('\n===== THE COMPUTED G[y*] IS SEPARABLE =====\n');
    fprintf('Number of non-singular solutions found: %d\n', size(anew, 3));
else
    fprintf('\n===== THE COMPUTED G[y*] IS NOT SEPARABLE =====\n');
end

% Display results
fprintf('\n===== FINAL TENSOR COMPLETION RESULTS =====\n');
fprintf('Solution 1 - Factor a:\n'); disp(anew(:,:,1));
fprintf('Solution 1 - Factor b:\n'); disp(bnew(:,:,1));
fprintf('Solution 1 - Factor c:\n'); disp(cnew(:,:,1));


fprintf('\nSolution 2 - Factor a:\n'); disp(anew(:,:,2));
fprintf('Solution 2 - Factor b:\n'); disp(bnew(:,:,2));
fprintf('Solution 2 - Factor c:\n'); disp(cnew(:,:,2));
