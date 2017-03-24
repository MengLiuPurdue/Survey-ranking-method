function [coeff,mu]=feature_reduction(X,F)
% [mu,coeff]=pcalearn(F,X);
[coeff,~,~,~,~,mu] = pca(X);
coeff=coeff(:,1:F);