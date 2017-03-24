%This file is used to read the raw data.
function [X,y,qid]=readData(filename,feature_num,data_num)
X=zeros(data_num,feature_num)+NaN;
y=zeros(data_num,1);
qid=zeros(data_num,1);
rptr=fopen(filename);
for i=1:data_num
    for k=1:5
        tline=fgetl(rptr);
    end
    temp=char(regexp(tline,' ', 'split'));
    y(i)=str2num(temp(1));
    s=char(regexp(temp(2,:),':','split'));
    qid(i)=str2num(s(2,:));
    j=3;
    s=char(regexp(temp(j,:),':','split'));
    count=str2num(s(1,:));
    value=str2num(s(2,:));
    while count<=feature_num
        X(i,count)=value;
        j=j+1;
        s=char(regexp(temp(j,:),':','split'));
        count=str2num(s(1,:));
        value=str2num(s(2,:));
    end
end
fclose(rptr);