% Input: number of features F
% data matrix X, with n rows (samples), d columns (features)
% average mu, with d rows, 1 column
% principal component matrix Z, with d rows, F columns
% Output: projected data matrix P, with n rows, F columns
function P = pcaproj(X,mu,Z)
[n,d]=size(X);
for t=1:n
    for i=1:d
        X(t,i)=X(t,i)-mu(i);
    end
end
P=X*Z;