function [h_best,new_weight,bin_best,I]=realboost_features(subfeatures,N_pos,weight)
%for one image.

row=size(subfeatures,1); %number of filters 100
col=size(subfeatures,2); %number of images 37194
bin_num=50;%how many bins do we want?
Z_rec=zeros(1,row);
h_rec=zeros(row,bin_num);
bin_lab_rec=zeros(row,col);
bin_rec=zeros(row,bin_num+1);
indicator=[repelem(1,N_pos),repelem(-1,(col-N_pos))];

for i=1:row
     bin_size = (max(subfeatures(i,:))-min(subfeatures(i,:)))/bin_num; %bin_size
     bin_interval=(min(subfeatures(i,:)): bin_size: max(subfeatures(i,:)));%Note: edges     
     bin_rec(i,:)=bin_interval;
     %know the bin labels of each image features in each bins.
     bin_lab=discretize(subfeatures(i,:),bin_interval);
     bin_lab_rec(i,:)=bin_lab;
     indicator_w=weight.*indicator;

     h_bin_rec=zeros(1,bin_num);
     z_bin_rec=zeros(1,bin_num);
     
     for j=1:bin_num
         w_temp=indicator_w(bin_lab==j);
         w_sum_pos=sum(w_temp(w_temp>0));
         w_sum_neg=-sum(w_temp(w_temp<0));
         h_bin=0.5*log((w_sum_pos+(exp(-16)))/(w_sum_neg+(exp(-16))));%1/2log(p/q)
         z_bin=sqrt(w_sum_pos*w_sum_neg);
         h_bin_rec(1,j)=h_bin; %record h for each bin
         z_bin_rec(1,j)=z_bin; 
     end
     
     Z=2*sum(z_bin_rec); %calculate the Z so that we can choose best filter
     Z_rec(1,i)=Z; 
     h_rec(i,:)=h_bin_rec;
end
[~,I]=min(Z_rec);
h_best=h_rec(I,:);
bin_best=bin_rec(I,:);

%update the weights
t_features=subfeatures(I,:);
t_index=discretize(t_features,bin_best);
w_change=exp(-h_best(t_index).*indicator);
new_w_temp=weight.*w_change;
new_weight=new_w_temp/sum(new_w_temp);

% w_change=exp(-repelem(h_best,tag_num_best).*sorted_indicator_rec(I,:));
% [~,I2]=sort(sorted_index);
% w_change2=w_change(I2);
% new_w_temp=weight.*w_change2;
% new_weight=new_w_temp/sum(new_w_temp);

end