%wrtite preprocessed data
function writeData(X,y,qid,filename)
wptr=fopen(filename,'w');
[m,n]=size(X);
for i=1:m
    fprintf(wptr,'%d qid:%d',y(i),qid(i));
    for j=1:n
        fprintf(wptr,' %d:%f',j,X(i,j));
    end
    fprintf(wptr,'\n');
end
fclose(wptr);