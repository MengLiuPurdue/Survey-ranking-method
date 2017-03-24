%This file contains all grid search experiments.

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
%grid search of RankNet
M=[];
V=[];
iter=1;
lr=[0.000025,0.0016];
layers=[3,6,10];
nodes=[13,17,25];
for i=1:2
    for j=1:3
        for k=1:3
            iter
            [M(iter),V(iter),~,~]=cross_valid(X_train,y,qid,10,data_num,'RankNet',[layers(j),nodes(k),lr(i)],F);
            iter=iter+1;
        end
    end
end
name=cell(iter-1,1);
iter=1;
for i=1:2
    for j=1:3
        for k=1:3
            temp=[num2str(layers(j)),', ',num2str(nodes(k)),', ',num2str(lr(i))];
            name(iter)=cellstr(temp);
            iter=iter+1;
        end
    end
end
figure
for i=1:(iter-1)
    u=1-M(i);
    s=sqrt(V(i));
    x=linspace(u-2*s,u+2*s,100);
    y=normpdf(x,u,s);
    plot(x,y)
    hold on
end
legend(name,'FontSize',10)
set(gca,'FontSize',15);

%%
%grid search of MART
M=[];
V=[];
iter=1;
lr=[0.06,0.09];
trees=[500,700,1800];
leaves=[13,18];
for i=1:2
    for j=1:3
        for k=1:2
            iter
            [M(iter),V(iter),~,~]=cross_valid(X_train,y,qid,10,data_num,'MART',[trees(j),leaves(k),lr(i)],F);
            iter=iter+1;
        end
    end
end
name=cell(iter-1,1);
iter=1;
for i=1:2
    for j=1:3
        for k=1:2
            temp=[num2str(trees(j)),', ',num2str(leaves(k)),', ',num2str(lr(i))];
            name(iter)=cellstr(temp);
            iter=iter+1;
        end
    end
end
figure
for i=1:(iter-1)
    u=1-M(i);
    s=sqrt(V(i));
    x=linspace(u-2*s,u+2*s,100);
    y=normpdf(x,u,s);
    plot(x,y)
    hold on
end
legend(name,'FontSize',10)
set(gca,'FontSize',15);
%%
%grid search of LambdaMART
M=[];
V=[];
iter=1;
tree=1900;
leaves=[5,14,20];
lr=[0.03,0.06,0.09];
for i=1:3
    for j=1:3
        iter
        [M(iter),V(iter),~,~]=cross_valid(X_train,y,qid,10,data_num,'LambdaMART',[tree,leaves(i),lr(j)],F);
        iter=iter+1;
    end
end
name=cell(iter-1,1);
iter=1;
for i=1:3
    for j=1:3
        temp=[num2str(tree),', ',num2str(leaves(i)),', ',num2str(lr(j))];
        name(iter)=cellstr(temp);
        iter=iter+1;
    end
end
figure
for i=1:(iter-1)
    u=1-M(i);
    s=sqrt(V(i));
    x=linspace(u-2*s,u+2*s,100);
    y=normpdf(x,u,s);
    plot(x,y)
    hold on
end
legend(name,'FontSize',10)
set(gca,'FontSize',15);

%%
%grid search of Random Forests
M=[];
V=[];
iter=1;
bags=[220,420];
trees=[7,10];
leaves=[60,240];
lr=[0.025,1.6];
for i=1:2
    for j=1:2
        for k=1:2
            for m=1:2
                iter
                [M(iter),V(iter),~,~]=cross_valid(X_train,y,qid,10,data_num,'Random Forests',[bags(i),trees(j),leaves(k),lr(m)],F);
                iter=iter+1;
            end
        end
    end
end
name=cell(iter-1,1);
iter=1;
for i=1:2
    for j=1:2
        for k=1:2
            for m=1:2
                temp=[num2str(bags(i)),', ',num2str(trees(j)),', ',num2str(leaves(k)),', ',num2str(lr(m))];
                name(iter)=cellstr(temp);
                iter=iter+1;
            end
        end
    end
end
figure
for i=1:(iter-1)
    u=1-M(i);
    s=sqrt(V(i));
    x=linspace(u-2*s,u+2*s,100);
    y=normpdf(x,u,s);
    plot(x,y)
    hold on
end
legend(name,'FontSize',10)
set(gca,'FontSize',15);