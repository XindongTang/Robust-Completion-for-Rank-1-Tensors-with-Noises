%%%% example 5.7, Table 2, input den,n1,n2,n3,sigma
%%%% n1,n2,n3: dimension
%%% den:density
%%% sigma: the scale of noise, for example : 1e-4;

%%%%% err_rat: ratio error
function [t1,err_rat] = convex_relaxation(den,n1,n2,n3,sigma)

%ax=randi([1 n1], n1, 1);
ax = randn(n1,1);
ax = ax/norm(ax);

by = randn(n2,1); 
by = by/norm(by);
  
cz = randn(n3,1);

Ndim = n1*n2;

XY=ax*by';
T0=[];
for k=1:n3
    T0(:,:,k)=cz(k)*XY;
end


sigma_tensor=sigma*rand(n1,n2,n3);

TN=T0.*sigma_tensor;
T=T0+TN;

i=1:n1;
j=1:n2;
k=1:n3;
[I,J,K]=meshgrid(i,j,k);
mesh=[I(:),J(:),K(:)];

OM=ceil(sprand(length(mesh),3,den));
[row,col]=find(OM);
Omrow=mesh(row,1:3);
Omsort=sortrows(Omrow,[3,2]);
Omega=unique(Omsort,'rows','stable');
subset={};
for i=1:n3
    om=Omega(:,3);
    K3=find((om==i));
    subset{i}=Omega(K3, 1:3);
end


%%%%generate defining matrix for primal SDP %%%%%%%%%%%%%%%%%%%%%%

Cone.s = Ndim;
 
ZM1 = sparse( zeros( n1 ) );
ZM2 = sparse( zeros( n2 ) );
AAt = [];  
for i1 =1:n1
for j1 = i1:n1
     for i2 = 1: n2
     for j2 = i2: n2
     ZMAij = ZM1; ZMAij(i1,j1) = 1; ZMAij(j1,i1) = 1;
     ZMBij = ZM2; ZMBij(i2,j2) = 1; ZMBij(j2,i2) = 1;
     aat = vec( kron(ZMAij,ZMBij)  ); 
     AAt = [AAt, aat];
     end
     end
end
end
 

bbb = sparse(Ndim,Ndim); 
for k3 = 1 : n3
    Om3 = subset{k3}; 
    if size(Om3, 1) > 1
    for i = 1: size(Om3,1)
        r1 = Om3(i, :);
      for  j = i+1 : size(Om3, 1)
            r2 = Om3(j, :);
            is = r1(1); js = r1(2);
            it = r2(1); jt = r2(2);
            ais = zeros(n1,1); ait = zeros(n1,1);
            bjs = zeros(n2,1); bjt = zeros(n2,1);
            ais(is) = 1; ait(it) = 1;
            bjs(js) = 1; bjt(jt) = 1;
            bij = T(r2(1),r2(2),k3)*kron(ais, bjs)-T(r1(1),r1(2),k3)*kron(ait, bjt);
            bbb = bbb + bij*bij';           
        end
    end
    end
end

bb = ( vec(bbb)'*AAt )';

val_b1 = bb(1);
vecidenN = sparse( vec( eye(Ndim) ) );
bb = -( bb(2:length(bb))- val_b1*( vecidenN'*AAt(:,2:size(AAt,2)) )' ); 
cc = AAt(:,1); 

AAt = -(AAt(:,2:size(AAt,2)) - AAt(:,1)*( vecidenN'*AAt(:,2:size(AAt,2)) ) ); 
 
[blk,At,C,b] = read_sedumi(AAt',bb,cc,Cone);
OPTIONS.tol = 10^(-6);

tic;
[obj,X1,s1,y1,S1,Z1,ybar,v,info,runhist]=sdpnalplus(blk,At,C,b,[],[],[],[],[],OPTIONS);
t1 = toc;
 

MoMab =   S1{1}  ;
svds(MoMab)';

%% find anew,bnew

[lmv,lmd] = eigs(MoMab,1,'largestabs');
anew=lmv(1:n2:Ndim, 1);
bnew=lmv(1:n2,1);
anew = anew/norm(anew); 
bnew = bnew/norm(bnew);



%%%% solve cnew by linear least squares
linear_least_squares_AT=zeros(size(Omega));
linear_least_squares_bt=zeros(size(Omega,1),1);
for i=1:size(Omega,1)
    row=Omega(i,:);
    linear_least_squares_AT(i,row(3))=anew(row(1))*bnew(row(2));
    linear_least_squares_bt(i)=T(row(1),row(2),row(3));
end
cnew= lsqr(linear_least_squares_AT,linear_least_squares_bt);

%%%% error
diff1 = 0;
diff2 = 0;
for i=1:size(Omega,1)
    rowid=Omega(i,:);
    diff1= diff1+(T(rowid(1),rowid(2),rowid(3))-...
        anew(rowid(1))*bnew(rowid(2))*cnew(rowid(3)))^2;
    diff2=diff2+(TN(rowid(1),rowid(2),rowid(3)))^2;
end
err_rat=sqrt(diff1)/sqrt(diff2);
