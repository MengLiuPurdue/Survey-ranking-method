function [NDCG_test,NDCG_train]=get_NDCG(results)
[m,n]=size(results);
digits=['0','1','2','3','4','5','6','7','8','9'];
i=n;
while ismember(results(i),digits)==0
    i=i-1;
end
end_pos=i;
while results(i)~=':'
    i=i-1;
end
start_pos=i+1;
NDCG_test=str2num(results(start_pos:end_pos));
while results(i)~='@'
    i=i-1;
end
while ismember(results(i),digits)==0
    i=i-1;
end
end_pos=i;
while results(i)~=':'
    i=i-1;
end
start_pos=i+1;
NDCG_train=str2num(results(start_pos:end_pos));