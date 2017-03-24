function MAP_test=get_MAP(results)
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
MAP_test=str2num(results(start_pos:end_pos));