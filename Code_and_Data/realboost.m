function realboost()

% flags
flag_load_original_features=0;
flag_boosting = 1;
flag_data_subset=0;

if flag_load_original_features
load('features.mat','features');
load('adaboost.mat','classifer_rec');
features_real=features(classifer_rec,:);
save('features_real.mat', '-v7.3', 'features_real');
else
load('features_real.mat', 'features_real');
end

% constants
if flag_data_subset
    N_pos = 100;
    N_neg = 100;
else
    %N_pos = 11838;
    %N_neg = 45356;
    
    N_pos = 11838;
    N_neg = 25356;
end

N=N_pos+N_neg;

%% performing realboost
weight=repelem(1/N,N);
T=100;

if flag_boosting
    for i=1:T
    [h_best,new_weight,bin_best,index]=realboost_features(features_real,N_pos,weight); 
    
    h_best_rec2(i,:)=h_best;
    bin_best_rec2(i,:)=bin_best;
    I_rec2(i)=index;
    
    weight=new_weight;
    end   
end

map=features_real(I_rec2,:);
map_result=zeros(T,N);

for k=1:T
    loc=discretize(map(k,:),bin_best_rec2(k,:));
    h_temp=h_best_rec2(k,:); 
    map_result(k,:)=h_temp(loc);
end

F=map_result; %a matrix represent final F value of all images.
% F=result;
% result(result>0)=1;
% result(result<0)=-1;
% correct_answer(repelem(1,N_pos),repelem(-1,N-N_pos));
% right=result.*correct_answer;


%% draw negative positive histograms for realboost
for j=[10,50,100]
    F_temp=F(1:j,:);
    result=sum(F_temp,1);
    F_pos=result(1:N_pos);
    F_neg=result(N_pos+1:length(result));
    figure;
    histogram(F_pos);
    hold on 
    histogram(F_neg);
    title(sprintf('Histogram T=%d for realboost',j));
    legend('Pos','Neg');
end

%% plot roc curve for real boost
correct_answer=[repelem(1,N_pos),repelem(-1,N-N_pos)];
%T=10
judge_mat_temp=F(1:10,:);
F_value_10=sum(judge_mat_temp,1);
[X_10,Y_10]=perfcurve(correct_answer,F_value_10,1);
%T=50
judge_mat_temp=F(1:50,:);
F_value_50=sum(judge_mat_temp,1);
[X_50,Y_50]=perfcurve(correct_answer,F_value_50,1);
%T=100
judge_mat_temp=F(1:100,:);
F_value_100=sum(judge_mat_temp,1);
[X_100,Y_100]=perfcurve(correct_answer,F_value_100,1);

figure;
plot(X_10,Y_10,'red');
hold on
plot(X_50,Y_50,'blue');
plot(X_100,Y_100,'green');
legend('T=10','T=50','T=100')
title('The ROC Curve for realboost')
xlabel('False positive rate')
ylabel('True positive rate')
hold off

end






