% file = 'four_scores.csv';
% data=csvread(file);
% plot(1:27655,data(:,1));hold on;
% plot(1:27655,data(:,2));hold on;
% plot(1:27655,data(:,3));hold on;
% plot(1:27655,data(:,4));hold on;

file = 'obs_data_16d_255.csv';
data=csvread(file);
figure(1)
plot(1:27914,data(:,1));hold on;

data1 = data(1:end-1,1);
data2 = data(2:end,1);
change_idx = find(abs(data2-data1)>=10);
% change_idx = reshape(change_idx,[2,length(change_idx)/2]);
% figure(2)
% for i = 1:2:size(change_idx,1)
%     index = change_idx(i)+1:change_idx(i+1);
%     segment = data(index,1);
%     plot(index,segment);hold on;
% end
%% extract segments from the first dimension of object data
segment_idx = [];
i = 1;
Start = [];
End  = [];
while i <= 27914%= 1:27914
    if data(i,1) ~= 255
        Start = [Start,i];
        while data(i,1) ~= 255
            i = i+1;
        end
        End = [End,i-1];
        %segment_idx = [segment_idx,i];
    end
    i = i+1;
end
figure(2)
for i = 1:size(Start,2)
    index = Start(i):End(i);
    segment = data(index,1);
    plot(index,segment);hold on;
end
