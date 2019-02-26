function adaboost()

% flags
flag_data_subset = 0;
flag_extract_features = 0;
flag_parpool = 0;
flag_boosting = 0;

flag_detect=1;
flag_extract_rectangle=0;%used to scan the images and save it to file,when perform hard mining, change the save mat name

flag_hard__mining=0;%used to get hard_negative images, default to 0
extract_neg_mining=0;%used to get hard_negative images,scan the images, default to 0

flag_perform_hard_mining=1; %read hard_mining images into the negative images.
flag_load_hard_mining=0; % want to load features from adaboosthard.mat
flag_draw_graphs=1; %want to draw result pictures,set to 1

% parpool
if flag_parpool
    delete(gcp('nocreate'));
    parpool(8);
end

% unit tests
test_sum_rect();
test_filters();

% constants
if flag_perform_hard_mining
    N_pos=11838;
    N_neg1=15356; %set the negative number small for speed.
    N_neg=N_neg1+4161+4202+2992;
    
else
    
    if flag_data_subset
    N_pos = 100;
    N_neg = 100;
    else
    %N_pos = 11838;
    %N_neg = 45356;
    
    N_pos = 11838;
    N_neg = 25356;
    end
end
N = N_pos + N_neg;
w = 16;
h = 16;

% (1) haar filter

%% load images
if flag_extract_features
    tic;
    
    if flag_perform_hard_mining
    
    I = zeros(N, h, w);
    for i=1:N_pos
    I(i,:,:) = rgb2gray(imread(sprintf('newface16/face16_%06d.bmp',i), 'bmp'));
    end
    for i=1:N_neg1
    I(N_pos+i,:,:) = rgb2gray(imread(sprintf('nonface16/nonface16_%06d.bmp',i), 'bmp'));
    end
    disp('going load hard negative images')
    for i=1:4161
    I(N_pos+N_neg1+i,:,:)=imread(sprintf('nonface16/hard_negative_mining/image1_%d.png',i));
    end
    for i=1:4202
    I(N_pos+N_neg1+4161+i,:,:)=imread(sprintf('nonface16/hard_negative_mining/image2_%d.png',i));
    end
    for i=1:2992
    I(N_pos+N_neg1+4161+4202+i,:,:)=imread(sprintf('nonface16/hard_negative_mining/image3_%d.png',i));
    end
    
    else
    I = zeros(N, h, w);
    for i=1:N_pos
        I(i,:,:) = rgb2gray(imread(sprintf('newface16/face16_%06d.bmp',i), 'bmp'));
    end
    for i=1:N_neg
        I(N_pos+i,:,:) = rgb2gray(imread(sprintf('nonface16/nonface16_%06d.bmp',i), 'bmp'));
    end
    
    end
    fprintf('Loading images took %.2f secs.\n', toc);
    disp(size(I,1))
end

%% construct filters
A = filters_A();
B = filters_B();
C = filters_C();
D = filters_D();
if flag_data_subset
    filters = [A(1:250,:); B(1:250,:); C(1:250,:); D(1:250,:)];
else
    filters = [A; B; C; D];
end

%% extract features
if flag_extract_features
    tic;
    I = normalize(I);
    II = integral(I);
    features = compute_features(II, filters);
    clear I;
    clear II;
    if flag_perform_hard_mining
        disp('start to save features_hard')
        save('features_hard.mat', '-v7.3', 'features');
        disp('already finished the save process')
    else
        save('features.mat', '-v7.3', 'features');
    end
    fprintf('Extracting %d features from %d images took %.2f secs.\n', size(filters, 1), N, toc);
else
if flag_perform_hard_mining
    disp('start load the hard features')
    load('features_hard.mat','features');
    disp('finish load the hard features')
else
    load('features.mat','features');
end
end
disp('I finished extract_features')

%% perform boosting
% implement this
if(flag_boosting == 1)
   fprintf('Running AdaBoost with %d features from %d images.\n', size(filters, 1), N);
   tic;
   T=100;
   %initial weights
   weight=repelem(1/N,N);
   alpha_rec=zeros(1,T);
   classifer_rec=zeros(1,T);
   polarity_rec=zeros(1,T);
   threshold_rec=zeros(1,T);
   count=0;
   for i=1:T
       if ismember(i,[1,10,50,100])
       [my_alpha,updated_weight,classifer,polarity,threshold,weak_err]=my_classifier_new(features,N_pos,weight);
       count=count+1;
       weak_err_rec(count,:)=weak_err;
       else
       [my_alpha,updated_weight,classifer,polarity,threshold]=my_classifier_new(features,N_pos,weight);
       end
       alpha_rec(1,i)=my_alpha;
       classifer_rec(1,i)=classifer;
       polarity_rec(1,i)=polarity;
       threshold_rec(1,i)=threshold;
       weight=updated_weight;
       sprintf('finish %d round out of %d round',[i,T]')
   end
   if flag_perform_hard_mining
    save('adaboosthard.mat','-v7.3','alpha_rec','classifer_rec','polarity_rec','threshold_rec','weak_err_rec','filters');
   else
    save('adaboost.mat','-v7.3','alpha_rec','classifer_rec','polarity_rec','threshold_rec','weak_err_rec','filters');
   end
    fprintf('Running AdaBoost %d with features from %d images took %.2f secs.\n', size(filters, 1), N, toc);
else
    T=100;
    if flag_perform_hard_mining
    load('adaboosthard.mat','alpha_rec','classifer_rec','polarity_rec','threshold_rec','weak_err_rec','filters');
    disp('finish load adaboosthard.mat')
    else
    load('adaboost.mat','alpha_rec','classifer_rec','polarity_rec','threshold_rec','weak_err_rec','filters');
    end
end

%% (1) top-20 haar filters
if flag_draw_graphs
%implement this
a=filters(classifer_rec(1:20),:);
figure;
for i=1:20
    if polarity_rec(1,i)==1
    color=['w','k'];
    else
    color=['k','w'];
    end
    subplot(4,5,i)
    if size(a{i,1},1)==1 && size(a{i,2},1)==1
        rectangle('Position',a{i,1},'FaceColor',color(1));
        axis([0,16,0,16])
        hold on
        rectangle('Position',a{i,2},'FaceColor',color(2));
    elseif size(a{i,1},1)==1 && size(a{i,2},1)==2
        b=a{i,2};
        rectangle('Position',a{i,1},'FaceColor',color(1));
        axis([0,16,0,16])
        hold on 
        rectangle('Position',b(1,:),'FaceColor',color(2));
        rectangle('Position',b(2,:),'FaceColor',color(2));
    elseif size(a{i,1},1)==2 && size(a{i,2},1)==1
        c=a{i,1};
        rectangle('Position',c(1,:),'FaceColor',color(1));
        axis([0,16,0,16])
        hold on
        rectangle('Position',c(2,:),'FaceColor',color(1));
        rectangle('Position',a{i,2},'FaceColor',color(2));
    else
        d=a{i,1};
        e=a{i,2};
        rectangle('Position',d(1,:),'FaceColor',color(1));
        axis([0,16,0,16])
        hold on
        rectangle('Position',d(2,:),'FaceColor',color(1));
        rectangle('Position',e(1,:),'FaceColor',color(2))
        rectangle('Position',e(2,:),'FaceColor',color(2))
    end
     hold off
end
suptitle('The top-20 Haar filters')
%% (2) plot training error
%implement this

%  T=5;
%  N_pos=100;
%  N_neg=100;
%  N=200;
map=features(classifer_rec,:);
map_temp=bsxfun(@times,map,polarity_rec');
result_indicator=bsxfun(@ge,map_temp,(threshold_rec.*polarity_rec)');

result_mat=-ones(size(result_indicator));
result_mat(result_indicator)=1;

judge_mat=bsxfun(@times,result_mat,alpha_rec');
correct_answer=[repelem(1,N_pos),repelem(-1,N_neg)];
err_rec=zeros(1,T);

for i =1:T
    judge_mat_temp=judge_mat(1:i,:);
    result_vec=sign(sum(judge_mat_temp,1));
    err_vec=result_vec.*correct_answer;
    err=length(err_vec(err_vec==-1))/length(err_vec);
    err_rec(1,i)=err;
end
figure;

plot(1:1:T,err_rec);
title('Training error')
xlabel('The number of weak classifiers')
ylabel('The training error of strong classifier')

%% (3) training errors of top-1000 weak classifiers
% implement this
figure;
plot(1:1:1000,weak_err_rec(1,:));
hold on
for i =1:size(weak_err_rec,1)-1
plot(1:1:1000,weak_err_rec(i+1,:));
end
hold off
legend('T=1','T=10','T=50','T=100','location','northeastoutside')
title('Weak error')
%% (4) negative positive histograms
% implement this
for j=[10,50,100]
    judge_mat_temp=judge_mat(1:j,:);
    F_value=sum(judge_mat_temp,1);
    F_pos=F_value(1:N_pos);
    F_neg=F_value(N_pos+1:length(F_value));
    figure;
    histogram(F_pos);
    hold on 
    histogram(F_neg);
    title(sprintf('Histogram T=%d',j));
    legend('Pos','Neg');
end

%% (5) plot ROC curves
% implement this
%T=10
judge_mat_temp=judge_mat(1:10,:);
F_value_10=sum(judge_mat_temp,1);
[X_10,Y_10]=perfcurve(correct_answer,F_value_10,1);
%T=50
judge_mat_temp=judge_mat(1:50,:);
F_value_50=sum(judge_mat_temp,1);
[X_50,Y_50]=perfcurve(correct_answer,F_value_50,1);
%T=100
judge_mat_temp=judge_mat(1:100,:);
F_value_100=sum(judge_mat_temp,1);
[X_100,Y_100]=perfcurve(correct_answer,F_value_100,1);

figure;
plot(X_10,Y_10,'red');
hold on
plot(X_50,Y_50,'blue');
plot(X_100,Y_100,'green');
legend('T=10','T=50','T=100')
title('The ROC Curve for adaboost')
xlabel('False positive rate')
ylabel('True positive rate')
hold off
end
%% (6) detect faces
% implement this
%load the testing image
%use permute to change dimension


if flag_detect   
    myfilter=filters(classifer_rec,:);
    tic
    disp('start to run flag_detect');
       for i=1:3
          I_test(:,:,i)=rgb2gray(imread(sprintf('Testing_Images/Face_%d.jpg',i)));
       end
    fprintf('Loading images took %.2f sec.\n',toc);
    if flag_extract_rectangle
    %scale the images
    scale_min=0.1;
    scale_max=0.9;
    step=0.1;
    for p=1:3
       myrectangle=[];myscore=[];
       for q=scale_min:step:scale_max
           temp_picture=imresize(I_test(:,:,p),q);
           imageset=face_detector(temp_picture);
           
           disp(size(imageset,1));
           
           imageset=normalize(imageset);
           imageset2=integral(imageset);
           myfeatures=compute_features(imageset2,myfilter);
           num_pic=size(myfeatures,2);
           
           
          %apply to strong classifier
          map_temp2=bsxfun(@times,myfeatures,polarity_rec');
          result_indicator2=bsxfun(@ge,map_temp2,(threshold_rec.*polarity_rec)');

          result_mat2=-ones(size(result_indicator2));
          result_mat2(result_indicator2)=1;
          judge_mat2=bsxfun(@times,result_mat2,alpha_rec');
          
          result2=sum(judge_mat2,1);
          position=find(result2>1.2);
          
          
          score=result2(position);
          
          colnum=size(temp_picture,2)-15;%use to know the position of rectangle.
          myrow=ceil(position/colnum);
          mycol=rem(position,colnum);
          px_row=repelem(16,size(position,2));
          myrec=vertcat(mycol,myrow,px_row,px_row)'/q;
          score=score';
          
          myrectangle=vertcat(myrectangle,myrec);
          myscore=vertcat(myscore,score);
       end
    [selectedBbox,selectedScore] = selectStrongestBbox(myrectangle,myscore,'RatioType','Min','OverlapThreshold',0);
    save(sprintf('Testing_images/hard_rectangle%d.mat',p), '-v7.3', 'selectedBbox','selectedScore');
    end
    
    else
    rec1=load('Testing_Images/hard_rectangle1.mat');  
    rec2=load('Testing_Images/hard_rectangle2.mat');
    rec3=load('Testing_Images/hard_rectangle3.mat');
    
    rectangle1=rec1.selectedBbox;
    rectangle2=rec2.selectedBbox;
    rectangle3=rec3.selectedBbox;
   
    
    score1=rec1.selectedScore;
    score2=rec2.selectedScore;
    score3=rec3.selectedScore;
    
    %draw bounding box
    [score1_sort,I_1]=sort(score1);
    rectangle1_sort=rectangle1(I_1,:);
    figure;
    imshow(I_test(:,:,1));
    hold on
    for i=1:length(score1_sort(score1_sort<=1.25))
        rectangle('Position',rectangle1_sort(i,:),'EdgeColor','r');
    end
    for i=(1+length(score1_sort(score1_sort<=1.25))):length(score1_sort(score1_sort<=1.6))
        rectangle('Position',rectangle1_sort(i,:),'EdgeColor','y');
    end
    for i=(1+length(score1_sort(score1_sort<=1.6))):size(rectangle1,1)
        rectangle('Position',rectangle1_sort(i,:),'EdgeColor','g');
    end
    hold off
    
    
    [score2_sort,I_2]=sort(score2);
    rectangle2_sort=rectangle2(I_2,:);
    figure;
    imshow(I_test(:,:,2));
    hold on
    for i=1:length(score2_sort(score2_sort<=1.25))
        rectangle('Position',rectangle2_sort(i,:),'EdgeColor','r');
    end
    for i=(1+length(score2_sort(score2_sort<=1.25))):length(score2_sort(score2_sort<=1.6))
        rectangle('Position',rectangle2_sort(i,:),'EdgeColor','y');
    end
    for i=(1+length(score2_sort(score2_sort<=1.6))):size(rectangle2,1)
        rectangle('Position',rectangle2_sort(i,:),'EdgeColor','g');
    end
    hold off
    
    
    
    [score3_sort,I_3]=sort(score3);
    rectangle3_sort=rectangle3(I_3,:);
    figure;
    imshow(I_test(:,:,3));
    hold on
    for i=1:length(score3_sort(score3_sort<=1.25))
        rectangle('Position',rectangle3_sort(i,:),'EdgeColor','r');
    end
    for i=(1+length(score3_sort(score3_sort<=1.25))):length(score3_sort(score3_sort<=1.6))
        rectangle('Position',rectangle3_sort(i,:),'EdgeColor','y');
    end
    for i=(1+length(score3_sort(score3_sort<=1.6))):size(rectangle3,1)
        rectangle('Position',rectangle3_sort(i,:),'EdgeColor','g');
    end
    hold off
   
    
    end
end

%% Hard negative mining
if flag_hard__mining
myfilter=filters(classifer_rec,:);
tic
disp('start to run hard_mining');
for i=1:3
  I_test(:,:,i)=rgb2gray(imread(sprintf('Testing_Images/Non_Face_%d.jpg',i)));
end
fprintf('Loading images took %.2f sec.\n',toc);

if extract_neg_mining
%scale the images
    scale_min=0.1;
    scale_max=0.9;
    step=0.1;
    for p=1:3
       myrectangle=[];myscore=[];
       for q=scale_min:step:scale_max
           temp_picture=imresize(I_test(:,:,p),q);
           imageset=face_detector(temp_picture);
           
           disp(size(imageset,1));
           
           imageset=normalize(imageset);
           imageset2=integral(imageset);
           myfeatures=compute_features(imageset2,myfilter);
           num_pic=size(myfeatures,2);
           
           
          %apply to strong classifier
          map_temp2=bsxfun(@times,myfeatures,polarity_rec');
          result_indicator2=bsxfun(@ge,map_temp2,(threshold_rec.*polarity_rec)');

          result_mat2=-ones(size(result_indicator2));
          result_mat2(result_indicator2)=1;
          judge_mat2=bsxfun(@times,result_mat2,alpha_rec');
          
          result2=sum(judge_mat2,1);
          position=find(result2>0);
          
          
          score=result2(position);
          
          colnum=size(temp_picture,2)-15;%use to know the position of rectangle.
          myrow=ceil(position/colnum);
          mycol=rem(position,colnum);
          px_row=repelem(16,size(position,2));
          myrec=vertcat(mycol,myrow,px_row,px_row)'/q;
          score=score';
          
          myrectangle=vertcat(myrectangle,myrec);
          myscore=vertcat(myscore,score);
       end
       save(sprintf('Testing_images/Non_face_rectangle%d.mat',p), '-v7.3', 'myrectangle','myscore');
    end
    
else
    rec1=load('Testing_Images/Non_face_rectangle1.mat');  
    rec2=load('Testing_Images/Non_face_rectangle2.mat');
    rec3=load('Testing_Images/Non_face_rectangle3.mat');
    
    rectangle1=rec1.myrectangle;
    rectangle2=rec2.myrectangle;
    rectangle3=rec3.myrectangle;
   
    
    %cut the hard mining and save in file.
    for i=1:size(rectangle1,1)
        test=imcrop(I_test(:,:,1),rectangle1(i,:));
        test=imresize(test,[16,16]);
        imwrite(test,sprintf('nonface16/hard_negative_mining/image1_%d.png',i));
    end
    
    
    for i=1:size(rectangle2,1)
        test=imcrop(I_test(:,:,2),rectangle2(i,:));
        test=imresize(test,[16,16]);
        imwrite(test,sprintf('nonface16/hard_negative_mining/image2_%d.png',i));
    end

    
   for i=1:size(rectangle3,1)
        test=imcrop(I_test(:,:,3),rectangle3(i,:));
        test=imresize(test,[16,16]);
        imwrite(test,sprintf('nonface16/hard_negative_mining/image3_%d.png',i));
   end
end
end

disp('Done.');

end





























%% filters

function features = compute_features(II, filters)
features = zeros(size(filters, 1), size(II, 1));
for j = 1:size(filters, 1)
    [rects1, rects2] = filters{j,:};
    features(j,:) = apply_filter(II, rects1, rects2);
end
end

function I = normalize(I)
[N,~,~] = size(I);
for i = 1:N
    image = I(i,:,:);
    sigma = std(image(:));
    I(i,:,:) = I(i,:,:) / sigma;
end
end

function II = integral(I)
[N,H,W] = size(I);
II = zeros(N,H+1,W+1);
for i = 1:N
    image = squeeze(I(i,:,:));
    II(i,2:H+1,2:W+1) = cumsum(cumsum(double(image), 1), 2);
end
end

function sum = apply_filter(II, rects1, rects2)
sum = 0;
% white rects
for k = 1:size(rects1,1)
    r1 = rects1(k,:);
    w = r1(3);
    h = r1(4);
    sum = sum + sum_rect(II, [0, 0], r1) / (w * h * 255);
end
% black rects
for k = 1:size(rects2,1)
    r2 = rects2(k,:);
    w = r2(3);
    h = r2(4);
    sum = sum - sum_rect(II, [0, 0], r2) / (w * h * 255);
end
end

function result = sum_rect(II, offset, rect)
x_off = offset(1);
y_off = offset(2);

x = rect(1);
y = rect(2);
w = rect(3);
h = rect(4);

a1 = II(:, y_off + y + h, x_off + x + w);
a2 = II(:, y_off + y + h, x_off + x);
a3 = II(:, y_off + y,     x_off + x + w);
a4 = II(:, y_off + y,     x_off + x);

result = a1 - a2 - a3 + a4;
end

function rects = filters_A()
count = 1;
w_min = 4;
h_min = 4;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:2:w_max
    for h = h_min:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w/2;
                r1_h = h;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x + r1_w;
                r2_y = r1_y;
                r2_w = w/2;
                r2_h = h;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                rects{count, 1} = r1; % white
                rects{count, 2} = r2; % black
                count = count + 1;
            end
        end
    end
end
end

function rects = filters_B()
count = 1;
w_min = 4;
h_min = 4;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:w_max
    for h = h_min:2:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w;
                r1_h = h/2;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x;
                r2_y = r1_y + r1_h;
                r2_w = w;
                r2_h = h/2;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                rects{count, 1} = r2; % white
                rects{count, 2} = r1; % black
                count = count + 1;
            end
        end
    end
end
end

function rects = filters_C()
count = 1;
w_min = 6;
h_min = 4;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:3:w_max
    for h = h_min:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w/3;
                r1_h = h;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x + r1_w;
                r2_y = r1_y;
                r2_w = w/3;
                r2_h = h;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                r3_x = r1_x + r1_w + r2_w;
                r3_y = r1_y;
                r3_w = w/3;
                r3_h = h;
                r3 = [r3_x, r3_y, r3_w, r3_h];
                
                rects{count, 1} = [r1; r3]; % white
                rects{count, 2} = r2; % black
                count = count + 1;
            end
        end
    end
end
end

function rects = filters_D()
count = 1;
w_min = 6;
h_min = 6;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:2:w_max
    for h = h_min:2:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w/2;
                r1_h = h/2;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x+r1_w;
                r2_y = r1_y;
                r2_w = w/2;
                r2_h = h/2;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                r3_x = x;
                r3_y = r1_y+r1_h;
                r3_w = w/2;
                r3_h = h/2;
                r3 = [r3_x, r3_y, r3_w, r3_h];
                
                r4_x = r1_x+r1_w;
                r4_y = r1_y+r2_h;
                r4_w = w/2;
                r4_h = h/2;
                r4 = [r4_x, r4_y, r4_w, r4_h];
                
                rects{count, 1} = [r2; r3]; % white
                rects{count, 2} = [r1; r4]; % black
                count = count + 1;
            end
        end
    end
end
end

function test_sum_rect()
% 1
I = zeros(1,16,16);
I(1,2:4,2:4) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [2, 2, 3, 3]) == 9);
assert(sum_rect(II, [0, 0], [10, 10, 2, 2]) == 0);

% 2
I = zeros(1,16,16);
I(1,10:16,10:16) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [10, 10, 2, 2]) == 4);

% 3
I = zeros(1,16,16);
I(1,:,:) = 0;
I(1,3:6,3:6) = 1;
I(1,3:6,11:14) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [11, 3, 6, 6]) == 16);

% 4
I = zeros(1,16,16);
I(1,:,:) = 0;
I(1,3:6,3:6) = 1;
I(1,3:6,11:14) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [3, 4, 4, 4]) == 12);
assert(sum_rect(II, [0, 0], [7, 4, 4, 4]) == 0);
assert(sum_rect(II, [0, 0], [11, 4, 4, 4]) == 12);
assert(sum_rect(II, [0, 0], [3, 3, 4, 4]) == 16);
assert(sum_rect(II, [0, 0], [11, 3, 4, 4]) == 16);

end

function test_filters()

% A
I = zeros(1,16,16);
I(1,:,:) = 255;
I(1,5:8,5:8) = 0;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_A();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r2s(1,3) * r2s(1,4);
    if(and(f_sum > max_sum, f_size == 4*4*2))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [r1s, r2s];
    end
end
assert(max_sum == 1);
assert(max_size == 4*4*2);
assert(isequal(min_f, [1 5 4 4 5 5 4 4]));

% B
I = zeros(1,16,16);
I(1,:,:) = 255;
I(1,2:5,2:5) = 0;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_B();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r2s(1,3) * r2s(1,4);
    if(and(f_sum > max_sum, f_size == 4*4*2))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [r1s, r2s];
    end
end
assert(max_sum == 1);
assert(max_size == 4*4*2);
assert(isequal(min_f, [2 6 4 4 2 2 4 4]));

% C
I = zeros(1,16,16);
I(1,:,:) = 0;
I(1,3:6,3:6) = 255;
I(1,3:6,11:14) = 255;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_C();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r1s(2,3) * r1s(2,4) + r2s(1,3) * r2s(1,4);
    if(and(f_sum > max_sum, f_size == 4*4*3))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [reshape(r1s', [1,8]), r2s];
    end
end
assert(max_sum == 2);
assert(max_size == 4*4*3);
assert(isequal(min_f, [3 3 4 4 11 3 4 4 7 3 4 4]));

% D
I = zeros(1,16,16);
I(1,:,:) = 255;
I(1,2:5,2:5) = 0;
I(1,6:9,6:9) = 0;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_D();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r1s(2,3) * r1s(2,4) + r2s(1,3) * r2s(1,4) + r2s(2,3) * r2s(2,4);
    if(and(f_sum > max_sum, f_size == 4*4*4))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [reshape(r1s', [1,8]), reshape(r2s', [1,8])];
    end
end
assert(max_sum == 2);
assert(max_size == 4*4*4);
assert(isequal(min_f, [6 2 4 4 2 6 4 4 2 2 4 4 6 6 4 4]));

end