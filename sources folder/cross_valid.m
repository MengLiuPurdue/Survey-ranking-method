function [M,V,M1,V1]=cross_valid(X,y,qid,k,S,method,para,F)
[m,n]=size(X);
id=linspace(1,m,m);
NDCG_test=zeros(k,1);
NDCG_train=zeros(k,1);
for i=1:k
    fprintf('cross validation: iter %d\n',i);
    s=S/k;
    train_id=[id(1:(i-1)*s),id(i*s+1:m)];
%     [mu,coeff]=pcalearn(F,X(train_id,:));
    [coeff,mu]=feature_reduction(X(train_id,:),F);
    P_train=pcaproj(X(train_id,:),mu,coeff);
    valid_id=id((i-1)*s+1:i*s);
    P_valid=pcaproj(X(valid_id,:),mu,coeff);
    writeData(P_train,y(train_id),qid(train_id),'new.set2.train.txt');
    writeData(P_valid,y(valid_id),qid(valid_id),'new.set2.valid.txt');
    if strcmp(method,'MART')
        cmd=['java -jar RankLib-2.7.jar -train new.set2.train.txt -test new.set2.valid.txt -ranker 0 -metric2t NDCG@10 -norm zscore -tree ',num2str(para(1)),' -leaf ',num2str(para(2)),' -shrinkage ',num2str(para(3))];
    elseif strcmp(method,'RankNet')
        cmd=['java -jar RankLib-2.7.jar -train new.set2.train.txt -test new.set2.valid.txt -ranker 1 -metric2t NDCG@10 -norm zscore -layer ',num2str(para(1)),' -node ',num2str(para(2)),' -lr ',num2str(para(3))];
    elseif strcmp(method,'RankBoost')
        cmd=['java -jar RankLib-2.7.jar -train new.set2.train.txt -test new.set2.valid.txt -ranker 2 -metric2t NDCG@10 -norm zscore -tc ',num2str(para(1))];
    elseif strcmp(method,'AdaRank')
        cmd=['java -jar RankLib-2.7.jar -train new.set2.train.txt -test new.set2.valid.txt -ranker 3 -metric2t NDCG@10 -norm zscore -round ',num2str(para(1)),' -tolerance ',num2str(para(2))];
    elseif strcmp(method,'LambdaMART')
        cmd=['java -jar RankLib-2.7.jar -train new.set2.train.txt -test new.set2.valid.txt -ranker 6 -metric2t NDCG@10 -norm zscore -tree ',num2str(para(1)),' -leaf ',num2str(para(2)),' -shrinkage ',num2str(para(3))];
    elseif strcmp(method,'ListNet')
        cmd=['java -jar RankLib-2.7.jar -train new.set2.train.txt -test new.set2.valid.txt -ranker 7 -metric2t NDCG@10 -norm zscore -lr ',num2str(para(1))];
    elseif strcmp(method,'Random Forests')
        cmd=['java -jar RankLib-2.7.jar -train new.set2.train.txt -test new.set2.valid.txt -ranker 8 -metric2t NDCG@10 -norm zscore -bag ',num2str(para(1)),' -tree ',num2str(para(2)),' -leaf ',num2str(para(3)),' -shrinkage ',num2str(para(4))];
    end
    [status,results]=system(cmd);
    [NDCG_test(i),NDCG_train(i)]=get_NDCG(results);
end
M=mean(NDCG_test);
V=var(NDCG_test);
M1=mean(NDCG_train);
V1=var(NDCG_train);