function K = sparse_kron(A, B)
    % Get sizes of A and B
    [mA, nA] = size(A);
    [mB, nB] = size(B);
    
    
    K = sparse(mA * mB, nA * nB);
    
    % Find the non-zero elements in A and B
    [rowA, colA, valA] = find(A);
    [rowB, colB, valB] = find(B);
    
    % Compute the Kronecker product
    for i = 1:length(valA)
        for j = 1:length(valB)
            rowK = (rowA(i) - 1) * mB + (rowB(j) - 1) + 1;
            colK = (colA(i) - 1) * nB + (colB(j) - 1) + 1;
            K(rowK, colK) = valA(i) * valB(j);
        end
    end
end
