%%% compute absolute error

function [cnew,error_abs]=error_abs(anew,bnew,T,Omega)
linear_least_squares_AT=zeros(size(Omega));
linear_least_squares_bt=zeros(size(Omega,1),1);
for i=1:size(Omega,1)
    row=Omega(i,:);
    linear_least_squares_AT(i,row(3))=anew(row(1))*bnew(row(2));
    linear_least_squares_bt(i)=T(row(1),row(2),row(3));
end
cnew= lsqr(linear_least_squares_AT,linear_least_squares_bt);
error_abs=0;
for i=1:size(Omega,1)
    rowid=Omega(i,:);
    error_abs=error_abs+(T(rowid(1),rowid(2),rowid(3))-...
        anew(rowid(1))*bnew(rowid(2))*cnew(rowid(3)))^2;
end
error_abs=sqrt(error_abs);

