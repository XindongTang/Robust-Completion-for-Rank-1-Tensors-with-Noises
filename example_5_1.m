clear all;clc; 
%%%%% example 5.1
%%%% rank G[y]=1

%Problem dimensions
n1 = 3; n2 = 4;n3 = 3;
Ndim = n1*n2; %dimension of the matrix

% tensor T
T(:,:,1) = [20.06  -10.01   40.35   40.11;
           40.02  -20.07   80.02   80.20;
           20.15 -10.03   40.34   40.20];

T(:,:,2) = [20.14  -10.09   40.39   40.00;
            40.08  -20.12   80.72   80.57;
            20.04 -10.03   40.07   40.34];

T(:,:,3) = [30.03  -15.09   60.41   60.07;
            60.02  -30.15  121.13  120.42;
            30.18  -15.00   60.52   60.14];
%%%% suppose the observation set is 
Omega =[1     1     1; 2    1     1;
        3     1     1; 1    2     1;
        3     2     1; 1    3     1;
        3     4     1; 2    1     2;
        3     1     2; 1    1     2;
        1     2     2; 3    2     2;
        2     2     2; 1    4     2;
        2     4     2; 1    1     3;
        3     1     3; 1    3     3;
        2     3     3; 3    3     3];

% Group observations by slice (third dimension)
% Omega_k in papper
subset={};
for i=1:n3
    om=Omega(:,3);
    K3=find((om==i));
    subset{i}=Omega(K3, 1:3);
end


%%%%generate defining matrix for primal SDP %%%%%%%%%%%%%%%%%%%%%%

Cone.s = Ndim; % Cone dimension for SDP
 
ZM1 = zeros( n1,n1 ) ;
ZM2 = zeros( n2 ,n2 );
AAt = [];   % Constraint matrix for SDP

% Construct the constraint matrix AAt using Kronecker products
for i1 =1:n1
for j1 = i1:n1
     for i2 = 1: n2
     for j2 = i2: n2
     ZMAij = ZM1; ZMAij(i1,j1) = 1; ZMAij(j1,i1) = 1;
     ZMBij = ZM2; ZMBij(i2,j2) = 1; ZMBij(j2,i2) = 1;
     % Construct the constraint matrix AAt using Kronecker products
     aat = vec( sparse_kron(ZMAij,ZMBij)  ); 
     AAt = [AAt, aat];
     end
     end
end
end
AAt = sparse(AAt); % Convert to sparse for efficiency


% Construct the right-hand side vector bb for SDP constraints
bbb = zeros(Ndim,Ndim); 
for k3 = 1 : n3
    Om3 = subset{k3}; 
    if size(Om3, 1) > 1
    for i = 1: size(Om3,1)
        r1 = Om3(i, :);
      for  j = i+1 : size(Om3, 1)
            r2 = Om3(j, :);
            is = r1(1); js = r1(2);
            it = r2(1); jt = r2(2);
            % Create basis vectors
            ais = zeros(n1,1); ait = zeros(n1,1);
            bjs = zeros(n2,1); bjt = zeros(n2,1);
            ais(is) = 1; ait(it) = 1;
            bjs(js) = 1; bjt(jt) = 1;
            % Form the constraint based on observed values
            bij = T(r2(1),r2(2),k3)*sparse_kron(ais, bjs)-T(r1(1),r1(2),k3)*sparse_kron(ait, bjt);
            bbb = bbb + bij*bij'; % Accumulate quadratic terms          
        end
    end
    end
end

% Process the constraint matrix and right-hand side for SDP formulation
bb = ( vec(bbb)'*AAt )';

val_b1 = bb(1);
vecidenN = sparse( vec( eye(Ndim) ) );
bb = -( bb(2:length(bb))- val_b1*( vecidenN'*AAt(:,2:size(AAt,2)) )' ); 
cc = AAt(:,1); 

AAt = -(AAt(:,2:size(AAt,2)) - AAt(:,1)*( vecidenN'*AAt(:,2:size(AAt,2)) ) ); 


%% ================SOLVE SDP BY SDPNAL+====================
% Convert to SeDuMi format and solve the SDP
[blk,At,C,b] = read_sedumi(AAt',bb,cc,Cone);
OPTIONS.tol = 10^(-6);

% Solve the SDP using sdpnalplus solver
tic;
[obj,X1,s1,y1,S1,Z1,ybar,v,info,runhist]=sdpnalplus(blk,At,C,b,[],[],[],[],[],OPTIONS);
t1 = toc,
 

MoMab =   S1{1}  ;
svds(MoMab)';% Display singular values 
 
%% find anew,bnew
% Recover factors anew and bnew from the MoMab
[lmv,lmd] = eigs(MoMab,1,'largestabs');
anew=lmv(1:n2:Ndim, 1);
bnew=lmv(1:n2,1);
% Normalize factors
anew = anew/norm(anew);
bnew = bnew/norm(bnew);

% Calculate  cnew and absolute error
[cnew,err_ab]=error_abs(anew,bnew,T,Omega);


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

