%This file contains all single parameter tuning experiments.

%%
%read and preprocess training data
clear all
feature_num=200;
data_num=2000;
[X,y,qid]=readData('set2.train.txt',feature_num,data_num);
new_X=[];
new_feature_num=0;
for i=1:feature_num
    count=0;
    for j=1:data_num
        if isnan(X(j,i))
            count=count+1;
            if count>0.5*data_num
                break;
            end
        end
    end
    if count<=0.5*data_num
        new_X=[new_X,X(:,i)];
        new_feature_num=new_feature_num+1;
    end
end
fprintf('read training data success\n');
X_train=expand_missing_data(new_X,new_feature_num,data_num);
X_train=zscore(X_train);
fprintf('expand mising data success\n');
F=20;

%%
%MART number of trees
i=1;
for Ntrees=linspace(500,2000,16)
    Ntrees
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,10,data_num,'MART',[Ntrees,10,0.1],F);
    i=i+1;
end
figure
[AX,H1,H2]=plotyy(linspace(500,2000,16),1-M,linspace(500,2000,16),1-M1);
xlabel('number of trees','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('MART (number of trees)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(linspace(500,2000,16),sqrt(V),linspace(500,2000,16),sqrt(V1));
xlabel('number of trees','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('MART (number of trees)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

%%
%MART number of leaves for each tree
M=[];
V=[];
M1=[];
V1=[];
i=1;
for Nleaves=linspace(5,20,16)
    Nleaves
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,5,data_num,'MART',[800,Nleaves,0.1],F);
    i=i+1;
end

figure
[AX,H1,H2]=plotyy(linspace(5,20,16),1-M,linspace(5,20,16),1-M1);
xlabel('number of leaves for each tree','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('MART (number of leaves for each tree)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(linspace(5,20,16),sqrt(V),linspace(5,20,16),sqrt(V1));
xlabel('number of leaves for each tree','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('MART (number of leaves for each tree)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

%%
%MART learning rate
M=[];
V=[];
M1=[];
V1=[];
i=1;
for Shrinkage=linspace(0.01,0.1,10)
    Shrinkage
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,10,data_num,'MART',[1000,10,Shrinkage],F);
    i=i+1;
end

figure
[AX,H1,H2]=plotyy(linspace(0.01,0.1,10),1-M,linspace(0.01,0.1,10),1-M1);
xlabel('learning rate','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('MART (learning rate)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(linspace(0.01,0.1,10),sqrt(V),linspace(0.01,0.1,10),sqrt(V1));
xlabel('learning rate','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('MART (learning rate)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

%%
%RankNet learning rate
M=[];
V=[];
M1=[];
V1=[];
i=1;
for lr=0.00005*2.^linspace(-5,5,11)
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,10,data_num,'RankNet',[1,10,lr],F);
    i=i+1;
end

figure
[AX,H1,H2]=plotyy(log(0.00005*2.^linspace(-5,5,11)),1-M,log(0.00005*2.^linspace(-5,5,11)),1-M1);
xlabel('log(learning rate)','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('RankNet (learning rate)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(log(0.00005*2.^linspace(-5,5,11)),sqrt(V),log(0.00005*2.^linspace(-5,5,11)),sqrt(V1));
xlabel('log(learning rate)','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('RankNet (learning rate)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);


%%
%RankNet number of layers
M=[];
V=[];
M1=[];
V1=[];
i=1;
for layers=linspace(1,10,10)
    layers
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,10,data_num,'RankNet',[layers,10,0.000025],F);
    i=i+1;
end
figure
[AX,H1,H2]=plotyy(linspace(1,10,10),1-M,linspace(1,10,10),1-M1);
xlabel('layers','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('RankNet (number of hidden layers)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(linspace(1,10,10),sqrt(V),linspace(1,10,10),sqrt(V1));
xlabel('layers','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('RankNet (number of hidden layers)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

%%
%RankNet number of nodes per layer
M=[];
V=[];
M1=[];
V1=[];
i=1;
for nodes=linspace(5,25,11)
    nodes
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,10,data_num,'RankNet',[1,nodes,0.000025],F);
    i=i+1;
end

figure
[AX,H1,H2]=plotyy(linspace(5,25,11),1-M,linspace(5,25,11),1-M1);
xlabel('number of nodes per layer','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);

set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('RankNet (number of nodes per layer)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(linspace(5,25,11),sqrt(V),linspace(5,25,11),sqrt(V1));
xlabel('number of nodes per layer','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('RankNet (number of nodes per layer)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

%%
%LambdaMART number of trees
M=[];
V=[];
M1=[];
V1=[];
i=1;
for Ntrees=linspace(500,2000,16)
    Ntrees
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,10,data_num,'LambdaMART',[Ntrees,10,0.1],F);
    i=i+1;
end

figure
[AX,H1,H2]=plotyy(linspace(500,2000,16),1-M,linspace(500,2000,16),1-M1);
xlabel('number of trees','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('LambdaMART (number of trees)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(linspace(500,2000,16),sqrt(V),linspace(500,2000,16),sqrt(V1));
xlabel('number of trees','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('LambdaMART (number of trees)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

%%
%LambdaMART number of leaves
M=[];
V=[];
M1=[];
V1=[];
i=1;
for Nleaves=linspace(5,20,16)
    Nleaves
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,10,data_num,'LambdaMART',[1000,Nleaves,0.1],F);
    i=i+1;
end
figure
[AX,H1,H2]=plotyy(linspace(5,20,16),1-M,linspace(5,20,16),1-M1);
xlabel('number of leaves','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('LambdaMART (number of leaves per tree)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(linspace(5,20,16),sqrt(V),linspace(5,20,16),sqrt(V1));
xlabel('number of leaves','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('LambdaMART (number of leaves per tree)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

%%
%LambdaMART learning rate
M=[];
V=[];
M1=[];
V1=[];
i=1;
for Shrinkage=linspace(0.01,0.1,10)
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,10,data_num,'LambdaMART',[1000,10,Shrinkage],F);
    i=i+1;
end
figure
[AX,H1,H2]=plotyy(linspace(0.01,0.1,10),1-M,linspace(0.01,0.1,10),1-M1);
xlabel('learning rate','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('LambdaMART (learning rate)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(linspace(0.01,0.1,10),sqrt(V),linspace(0.01,0.1,10),sqrt(V1));
xlabel('learning rate','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('LambdaMART (learning rate)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);


%%
%RankBoost threshold
M=[];
V=[];
M1=[];
V1=[];
i=1;
for threshold=linspace(5,15,11)
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,10,data_num,'RankBoost',[threshold],F);
    i=i+1;
end

figure
[AX,H1,H2]=plotyy(linspace(5,15,11),1-M,linspace(5,15,11),1-M1);
xlabel('threshold','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('RankBoost (threshold)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(linspace(5,15,11),sqrt(V),linspace(5,15,11),sqrt(V1));
xlabel('threshold','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('RankBoost (threshold)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

%%
%ListNet learning rate
M=[];
V=[];
M1=[];
V1=[];
i=1;
for lr=0.00001*2.^linspace(-5,5,11)
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,10,data_num,'ListNet',[lr],F);
    i=i+1;
end

figure
[AX,H1,H2]=plotyy(log(0.00001*2.^linspace(-5,5,11)),1-M,log(0.00001*2.^linspace(-5,5,11)),1-M1);
xlabel('learning rate','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('ListNet (learning rate)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(log(0.00001*2.^linspace(-5,5,11)),sqrt(V),log(0.00001*2.^linspace(-5,5,11)),sqrt(V1));
xlabel('learning rate','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('ListNet (learning rate)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);


%%
%Random Forests number of bags
M=[];
V=[];
M1=[];
V1=[];
i=1;
for bags=linspace(100,500,11)
    bags
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,10,data_num,'Random Forests',[bags,1,100,0.1],F);
    i=i+1;
end
figure
[AX,H1,H2]=plotyy(linspace(100,500,11),1-M,linspace(100,500,11),1-M1);
xlabel('number of bags','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('Random Forests (number of bags)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(linspace(100,500,11),sqrt(V),linspace(100,500,11),sqrt(V1));
xlabel('number of bags','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('Random Forests (number of bags)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

%%
%Random Forests number of trees in each bag
M=[];
V=[];
M1=[];
V1=[];
i=1;
for Ntrees=linspace(1,10,10)
    Ntrees
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,10,data_num,'Random Forests',[300,Ntrees,100,0.1],F);
    i=i+1;
end
figure
[AX,H1,H2]=plotyy(linspace(1,10,10),1-M,linspace(1,10,10),1-M1);
xlabel('number of trees in each bag','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('Random Forests (number of trees in each bag)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(linspace(1,10,10),sqrt(V),linspace(1,10,10),sqrt(V1));
xlabel('number of trees in each bag','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('Random Forests (number of trees in each bag)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

%%
%Random Forests learning rate
M=[];
V=[];
M1=[];
V1=[];
i=1;
for shrinkage=0.1*2.^linspace(-5,5,11)
    shrinkage
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,10,data_num,'Random Forests',[300,1,100,shrinkage],F);
    i=i+1;
end
figure
[AX,H1,H2]=plotyy(log(0.1*2.^linspace(-5,5,11)),1-M,log(0.1*2.^linspace(-5,5,11)),1-M1);
xlabel('log(learning rate)','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('Random Forests (learning rate)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(log(0.1*2.^linspace(-5,5,11)),sqrt(V),log(0.1*2.^linspace(-5,5,11)),sqrt(V1));
xlabel('log(learning rate)','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('Random Forests (learning rate)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

%%
%Random Forests number of leaves for each tree
M=[];
V=[];
M1=[];
V1=[];
i=1;
for leaf=linspace(40,300,14)
    leaf
    [M(i),V(i),M1(i),V1(i)]=cross_valid(X_train,y,qid,10,data_num,'Random Forests',[300,1,leaf,0.1],F);
    i=i+1;
end

figure
[AX,H1,H2]=plotyy(linspace(40,300,14),1-M,linspace(40,300,14),1-M1);
xlabel('number of leaves','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','mean of (1-NDCG@10)','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','mean of (1-NDCG@10)','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('Random Forests (number of leaves for each tree)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);

figure
[AX,H1,H2]=plotyy(linspace(40,300,14),sqrt(V),linspace(40,300,14),sqrt(V1));
xlabel('number of leaves','FontSize',15);
HH1=get(AX(1),'Ylabel');
set(HH1,'String','std dev of NDCG@10','FontSize',15);
HH2=get(AX(2),'Ylabel');
set(HH2,'String','std dev of NDCG@10','FontSize',15);
set(AX(1),'FontSize',12);
set(AX(2),'FontSize',12);
title('Random Forests (number of leaves for each tree)','FontSize',15);
legend([H1,H2],{'validation','training'},'FontSize',15);