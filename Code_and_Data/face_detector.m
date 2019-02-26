function [imageset]=face_detector(image) %image is a m by n scaled image (matrix)
m=size(image,1); %row of pixels
n=size(image,2); %col of pixels
count=1;
imageset=zeros((m-15)*(n-15),16,16);
for j=1:(m-15)
    for i=1:(n-15)
      imageset(count,:,:)=image(j:j+15,i:i+15);
      count=count+1;
    end
end
end