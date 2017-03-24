clear all
j=1;
data_num=500;
rptr=fopen('set2.test.txt','r');
wptr=fopen('test.txt','w');
for i=1:20000
    tline=fgetl(rptr);
    fprintf(wptr,'%s\n',tline);
end
fclose(wptr);
fclose(rptr);
rptr=fopen('set2.train.txt','r');
wptr=fopen('train.txt','w');
for i=1:100
    tline=fgetl(rptr);
    fprintf(wptr,'%s\n',tline);
end
fclose(wptr);
id=[0,1,2,6,7,8];
para={'-tree 700 -leaf 18 -shrinkage 0.06','-lr 0.0016 -layer 3 -node 25','-tc 13','-tree 1900 -leaf 14 -shrinkage 0.03','-lr 0.00001','-bag 220 -tree 7 -leaf 60 -shrinkage 1.6'};
for k=1:20
    k
    for i=1:6
        t1=clock;
        cmd=['java -jar RankLib-2.7.jar -train train.txt -test test.txt -ranker ',num2str(id(i)), ' -metric2t NDCG@10 -norm zscore -feature feature.txt ',char(para(i))];
        cmd
        [status,results]=system(cmd)
        t2=clock;
        rumtime(i,j)=etime(t2,t1);
        [NDCG_test(i,j),NDCG_train(i,j)]=get_NDCG(results);
    end
    j=j+1;
    wptr=fopen('train.txt','a+');
    for i=1:data_num
        tline=fgetl(rptr);
        fprintf(wptr,'%s\n',tline);
    end
    fclose(wptr);
end
fclose(rptr);
x=linspace(100,9960,20);
figure
for i=1:6
    plot(x,1-NDCG_test(i,:))
    hold on
end
legend('MART','RankNet','RankBoost','LambdaMART','ListNet','Random Forests');
xlabel('training size','FontSize',15);
ylabel('1-NDCG','FontSize',15);
set(gca,'FontSize',15);
figure
for i=1:6
    plot(x,rumtime(i,:))
    hold on
end
legend('MART','RankNet','RankBoost','LambdaMART','ListNet','Random Forests');
xlabel('training size','FontSize',15);
ylabel('runtime/sec','FontSize',15);
set(gca,'FontSize',15);