clear all
data_num=50;
for i=1:20
    rptr=fopen('set2.train.txt','r');
    wptr=fopen(['bag',num2str(i),'.txt'],'w');
    for j=1:20
        if j~=i
            for k=1:data_num
                tline=fgetl(rptr);
                fprintf(wptr,'%s\n',tline);
            end
        else
            for k=1:data_num
                tline=fgetl(rptr);
            end
        end
    end
    fclose(rptr);
    fclose(wptr);
end
rptr=fopen('set2.test.txt','r');
wptr=fopen('test.txt','w');
for i=1:2000
    tline=fgetl(rptr);
    fprintf(wptr,'%s\n',tline);
end
fclose(wptr);
fclose(rptr);
id=[0,1,2,6,7,8];
para={'-tree 700 -leaf 18 -shrinkage 0.06','-lr 0.0016 -layer 3 -node 25','-tc 13','-tree 1900 -leaf 14 -shrinkage 0.03','-lr 0.00001','-bag 220 -tree 7 -leaf 60 -shrinkage 1.6'};
for i=1:6 
    for j=1:20
        cmd=['java -jar RankLib-2.7.jar -train bag',num2str(j),'.txt -test test.txt -ranker ',num2str(id(i)), ' -metric2t NDCG@10 -norm zscore -feature feature.txt ',char(para(i)),' -save model.txt'];
        [status,results]=system(cmd)
        [NDCG_test(i,j),~]=get_NDCG(results);
        cmd='java -jar RankLib-2.7.jar -load model.txt -test test.txt -norm zscore -metric2T MAP -feature feature.txt';
        [status,results]=system(cmd)
        MAP_test(i,j)=get_MAP(results);
    end
end

figure
g={'MART','RankNet','RankBoost','LambdaMART','ListNet','Random Forests'};
boxplot(MAP_test',g);
set(gca,'FontSize',12);
ylabel('MAP','FontSize',15);
figure
g={'MART','RankNet','RankBoost','LambdaMART','ListNet','Random Forests'};
boxplot(NDCG_test',g);
set(gca,'FontSize',12);
ylabel('NDCG@10','FontSize',15);