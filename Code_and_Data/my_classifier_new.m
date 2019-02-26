function [alpha,new_weight,weakclassifer_num,polarity,threshold,weak_err,least_err]=my_classifier_new(features,N_pos,weight)
row=size(features,1); %number of filters
col=size(features,2); %number of images
bin_num=50;
theta_minerr_f=zeros(1,row);%record the least weighted error of each filters
theta_best_f=zeros(1,row);%record the threhold of best weak classfier
s_f=zeros(1,row);

for i=1:row
     [sorted_features,sorted_index]=sort(features(i,:)); %sort from low to high
     bin_size = (max(features(i,:))-min(features(i,:)))/bin_num;
     bin_interval=(min(features(i,:)): bin_size: max(features(i,:)));
     theta_temp=([0,bin_interval]+[bin_interval,0])/2;
     theta=theta_temp(2:(length(theta_temp)-1));
     myerr=zeros(1,length(theta));s_rec=zeros(1,length(theta));
     T_pos=sum(weight(1:N_pos));
     T_neg=sum(weight(N_pos+1:col));
     for j=1:length(theta)
         %suppose large than theta be positive-images, if err rate large
         %than 0.5 we can suppose inverse.
         S_pos=sum(weight(sorted_index(sorted_index(sorted_features<=theta(j))<=N_pos)));
         S_neg=sum(weight(sorted_index(sorted_index(sorted_features<=theta(j))>N_pos)));
         [err,s_indicator]=min([S_pos+(T_neg-S_neg),S_neg+(T_pos-S_pos)]);
         if s_indicator==1
             s=1;
         else
             s=-1;
         end
         myerr(1,j)=err;
         s_rec(1,j)=s;
     end
     [N,I]=min(myerr);
     s_f(1,i)=s_rec(I);
     theta_best_f(1,i)=theta(I);%know the threhold
     theta_minerr_f(1,i)=N;
end
[least_err,I2]=min(theta_minerr_f);
weakclassifer_num=I2;
threshold=theta_best_f(1,I2);
polarity=s_f(1,I2);

%get top 1000 weak_err
theta_minerr_temp=sort(theta_minerr_f);
weak_err=theta_minerr_temp(1:1000);


%calculate alpha using least_err 
alpha=0.5*log((1-least_err)/least_err);

%update the weights
y_indicator=[repelem(1,N_pos),repelem(-1,(col-N_pos))];
cl_rp=features(I2,:);
pred_indicator=repelem(-1,col);
pred_indicator(polarity*cl_rp>polarity*threshold)=1;
w_indicator=y_indicator.*pred_indicator;
weight_change=exp(-alpha*w_indicator);
new_weight=weight.*weight_change/sum(weight.*weight_change);
end
