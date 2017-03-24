% Input: number of features F
% data matrix X, with n rows (samples), d columns (features)
% Output: average mu, with d rows, 1 column
% principal component matrix Z, with d rows, F columns
function [mu Z] = pcalearn(F,X)
[n,d]=size(X);
mu=zeros(d,1);
for i=1:d
    mu(i)=mean(X(:,i));
end
for t=1:n
    for i=1:d
        X(t,i)=X(t,i)-mu(i);
    end
end
[U,D,V]=svd(X);
E=D(1:F,1:F);
W=V(:,1:F);
Z=sqrt(n)*W*inv(E);