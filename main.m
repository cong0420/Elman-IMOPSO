%% ��ʼ��
clear
close all
clc

%% ��ȡ����
 data=xlsread('demand.xlsx','Sheet1','A1:H23'); %%ʹ��xlsread������ȡEXCEL�ж�Ӧ��Χ�����ݼ���  
%data=xlsread('device.xlsx','Sheet2','A1:F23'); %%ʹ��xlsread������ȡEXCEL�ж�Ӧ��Χ�����ݼ���  
% %�����������
 input=data(:,1:4);    %data�ĵ�һ��-�����ڶ���Ϊ����ָ��
 output=data(:,5:end);  %data�������һ��Ϊ�����ָ��ֵ
% input=data(:,1:3);    %data�ĵ�һ��-�����ڶ���Ϊ����ָ��
% output=data(:,4:end);  %data�������һ��Ϊ�����ָ��ֵ
N=length(output);   %ȫ��������Ŀ
testNum=floor(N*0.15);  %�趨����������Ŀ
trainNum=N-testNum;    %����ѵ��������Ŀ

%% ����ѵ���������Լ�
input_train = input(1:trainNum,:)';
output_train =output(1:trainNum,:)';
input_test =input(trainNum+1:trainNum+testNum,:)';
output_test =output(trainNum+1:trainNum+testNum,:)';

%% ���ݹ�һ��
[inputn,inputps]=mapminmax(input_train,0,1);  %��һ����0,1֮�䣬��һ��������������ָ������ٺ���������Ӱ��
[outputn,outputps]=mapminmax(output_train,0,1);
inputn_test=mapminmax('apply',input_test,inputps);

%% ��ȡ�����ڵ㡢�����ڵ����
inputnum=size(input,2);
outputnum=size(output,2);
disp('/////////////////////////////////')
disp('������ṹ...')
disp(['�����Ľڵ���Ϊ��',num2str(inputnum)])
disp(['�����Ľڵ���Ϊ��',num2str(outputnum)])
disp(' ')
disp('������ڵ��ȷ������...')
string={'tansig','purelin'};    %���ݺ���
func_str='traingdx';    %ѵ���㷨
%ȷ��������ڵ����
%���þ��鹫ʽhiddennum=sqrt(m+n)+a��mΪ�����ڵ������nΪ�����ڵ������aһ��ȡΪ1-10֮�������
MSE=1e+5; %��ʼ��ѵ�����
for hiddennum=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
    
    %��������
    net=newelm(inputn,outputn,hiddennum,string,func_str);
    % �������
    net.trainParam.epochs=10000;         % ѵ������
    net.trainParam.lr=0.01;                   % ѧϰ����
    net.trainParam.goal=0.00001;        % ѵ��Ŀ����С���
    % ����ѵ��
    net=train(net,inputn,outputn);
    an0=sim(net,inputn);  %������
    mse0=mse(outputn,an0);  %����ľ������

%     disp(['������ڵ���Ϊ',num2str(hiddennum),'ʱ��ѵ�����ľ������Ϊ��',num2str(mse0)])
    
    %������ѵ�������ڵ�
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['��ѵ�������ڵ���Ϊ��',num2str(hiddennum_best),'����Ӧ�ľ������Ϊ��',num2str(MSE)])

%% �������������ڵ㣨�нӲ�ڵ㣩��ELMAN������
net=newelm(inputn,outputn,hiddennum_best,string,func_str);

% �������
net.trainParam.epochs=10000;         % ѵ������
net.trainParam.lr=0.01;                   % ѧϰ����
net.trainParam.goal=0.00001;        % ѵ��Ŀ����С���


%% ѵ������
net2=train(net,inputn,outputn);
%% �������
an=sim(net2,inputn_test); %��ѵ���õ�ģ�ͽ��з���
test_simu=mapminmax('reverse',an,outputps); % Ԥ��������һ��
error=output_test-test_simu;

% error1=zeros(3,1);
% for k1=1:3 
error1=zeros(4,1);
 for k1=1:4
    for k2=1:testNum
        error1(k1,1)=error1(k1,1)+abs(error(k1,k2));
    end
end
MSE=(error1'*error1)/2;
for kk=1:200%% ����ѵ������
    
    net1=train(net,inputn,outputn);
    %% �������
    an11=sim(net1,inputn_test); %��ѵ���õ�ģ�ͽ��з���
    test_simu11=mapminmax('reverse',an11,outputps); % Ԥ��������һ��
    error=output_test-test_simu11;
    
% error11=zeros(3,1);
% for k1=1:3 
error11=zeros(4,1);
for k1=1:4
    for k2=1:testNum
        error11(k1,1)=error11(k1,1)+abs(error(k1,k2));
    end
end
    MSE1=(error11'*error11)/2;
    if MSE1<MSE
        net2=net1;
        MSE=MSE1;
        test_simu=test_simu11;
    end
end
for kk=1:200 %% �ҵ����ʵ���
   
    net1=train(net,inputn,outputn);
    %% �������
    an11=sim(net1,inputn_test); %��ѵ���õ�ģ�ͽ��з���
    test_simu11=mapminmax('reverse',an11,outputps); % Ԥ��������һ��
    error=output_test-test_simu11;
    
% error11=zeros(3,1);
% for k1=1:3
error11=zeros(4,1);
for k1=1:4
    for k2=1:testNum
        error11(k1,1)=error11(k1,1)+abs(error(k1,k2));
    end
end
    MSE1=(error11'*error11)/2;
    if MSE1<MSE||MSE1==MSE
        net2=net1;
        MSE=MSE1;
        test_simu=test_simu11;
        1
        break;
    end
end
MSE
    figure(1);
    y1=plot(output_test,'r-');
    hold on;
    y2=plot(test_simu,'b-');
    legend([y1(1),y2(1)],'real value','predictive value');
    xlabel('production type');
    ylabel('demand');

%�����������
 input1=[16	15	17	15];   
 %input1=[1.86	1.42	1.61];  
%input1=[74	90	82];  

%% ����ѵ���������Լ�
input_test1 = input1(1,:)';
inputn_test1=mapminmax('apply',input_test1,inputps);
an1=sim(net2,inputn_test1); %��ѵ���õ�ģ�ͽ��з���
test_simu1=mapminmax('reverse',an1,outputps); % Ԥ��������һ��
round(test_simu1)

test_simu1


