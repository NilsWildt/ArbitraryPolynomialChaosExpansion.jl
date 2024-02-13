function fitting
load('Calibration_data_2005_2010.mat');
figure;
histogram(Post_MC_Vector(1,:),'Normalization','pdf');
hold on;
[f,z]=hist(Post_MC_Vector(1,:),100);
fNorm = f/(sum(f)*(z(2)-z(1)));
%plot(z,fNorm);
pd_1 = fitdist(Post_MC_Vector(1,:)','normal')
hold on;
y_1=pdf(pd_1,z);
line(z,y_1);

clear all;
load('Calibration_data_2005_2010.mat');
figure;
histogram(Post_MC_Vector(2,:),'Normalization','pdf');
hold on;
[f,z]=hist(Post_MC_Vector(2,:),100);
fNorm = f/(sum(f)*(z(2)-z(1)));
%plot(z,fNorm);
pd_2 = fitdist(Post_MC_Vector(2,:)','normal')
hold on;
y_2=pdf(pd_2,z);
line(z,y_2);

clear all;
load('Calibration_data_2005_2010.mat');
figure;
histogram(Post_MC_Vector(3,:),'Normalization','pdf');
hold on;
[f,z]=hist(Post_MC_Vector(3,:),100);
fNorm = f/(sum(f)*(z(2)-z(1)));
%plot(z,fNorm);
pd_3 = fitdist(Post_MC_Vector(3,:)','normal')
hold on;
y_3=pdf(pd_3,z);
line(z,y_3);
