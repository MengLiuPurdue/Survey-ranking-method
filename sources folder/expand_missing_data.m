function X_new=expand_missing_data(X,feature_num,data_num)
[coeff,score,~,~,~,mu]=pca(X,'algorithm','als');
for i=1:feature_num
    if isnan(mu(i))
        mu(i)=0;
    end
end
X_new=real(score*coeff'+repmat(mu,data_num,1));