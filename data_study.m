clear all;clc;
% file = 'obs_data_16d_255.csv';
file = 'obs_data_24d.csv';%
data=csvread(file);
data = data(1150:27149,:); % get rid of parking lot
figure(1)
plot(1:size(data,1),data(:,3));hold on;

% data1 = data(1:end-1,1);
% data2 = data(2:end,1);
% change_idx = find(abs(data2-data1)>=10);
% change_idx = reshape(change_idx,[2,length(change_idx)/2]);
% figure(2)
% for i = 1:2:size(change_idx,1)
%     index = change_idx(i)+1:change_idx(i+1);
%     segment = data(index,1);
%     plot(index,segment);hold on;
% end
%% extract segments from the first dimension of object data
% segment_idx = [];
% i = 1;
% Start = [];
% End  = [];
% while i <= 27914%= 1:27914
%     if data(i,1) ~= 255
%         Start = [Start,i];
%         while data(i,1) ~= 255
%             i = i+1;
%         end
%         End = [End,i-1];
%         %segment_idx = [segment_idx,i];
%     end
%     i = i+1;
% end
% figure(2)
% for i = 1:size(Start,2)
%     index = Start(i):End(i);
%     segment = data(index,1);
%     plot(index,segment);hold on;
% end
%% extract object segments
% figure(2)
% obs_id = unique(data(:,1));
% for i = 2:length(obs_id)
%     seg_idx = find(data(:,1) == obs_id(i));
%     segment = data(seg_idx,3);
%     plot(seg_idx,segment);hold on;
% end
segments_all_channels = {};
ctr_all_segments = 0; % count the number of total segments
for i_channel = 0:3
    obj_idx = find(data(:,i_channel*6+1) ~= -1);
    segments = {};
    prev_obs_ID = data(obj_idx(1),i_channel*6+1);
    prev_frame_ID = obj_idx(1);
    seg_idx = [prev_frame_ID];
    j = 0;
    for i = 2:length(obj_idx)
        if data(obj_idx(i),i_channel*6+1) == prev_obs_ID %&& obj_idx(i) == prev_frame_ID +1
            seg_idx = [seg_idx, obj_idx(i)];
        else
            j = j+1; % id of the new segment
            segments{j} = seg_idx; % a new segment
            seg_idx = [obj_idx(i)];
        end
        prev_obs_ID = data(obj_idx(i),i_channel*6+1);
    end
    long_segments = {}; % buffer to save long segments
    j = 0; % id of long segments
%     figure(2)
    for i = 1:length(segments)
        seg_id = segments{i};
        if length(seg_id) > 20
            j = j+1;
            long_segments{j} = [seg_id' data(seg_id,i_channel*6+1:i_channel*6+6)]; % [frame_ID in whole trajectory, obs_ID, obs_age, x_obs, y_obs, v_obs, a_obs]
            ctr_all_segments = ctr_all_segments + 1;
            csvwrite(['drive_segments/', sprintf('seg%04d.csv',ctr_all_segments)], long_segments{j})
%             plot(seg_id,data(seg_id,3));hold on;
        end
    end
    segments_all_channels{i_channel+1} = long_segments;
end
% min_len = 1e7;
% max_len = 0;
% sum = 0;
% num_total_segs = 0;
% for i_channel = 0:3
%     segs = segments_all_channels{i_channel+1};
%     num_segs = size(segs,2);
%     num_total_segs = num_total_segs + num_segs; 
%     for i = 1:length(segs)
%         len_seg = size(segs{i},1);
%         sum = sum + len_seg;
%         if len_seg < min_len
%             min_len = len_seg;
%         end
%         if len_seg > max_len
%             max_len = len_seg;
%         end
%     end
% end

