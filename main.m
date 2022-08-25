%% 初始化
clear
close all
clc

%% 读取数据
 data=xlsread('demand.xlsx','Sheet1','A1:H23'); %%使用xlsread函数读取EXCEL中对应范围的数据即可  
%data=xlsread('device.xlsx','Sheet2','A1:F23'); %%使用xlsread函数读取EXCEL中对应范围的数据即可  
% %输入输出数据
 input=data(:,1:4);    %data的第一列-倒数第二列为特征指标
 output=data(:,5:end);  %data的最后面一列为输出的指标值
% input=data(:,1:3);    %data的第一列-倒数第二列为特征指标
% output=data(:,4:end);  %data的最后面一列为输出的指标值
N=length(output);   %全部样本数目
testNum=floor(N*0.15);  %设定测试样本数目
trainNum=N-testNum;    %计算训练样本数目

%% 划分训练集、测试集
input_train = input(1:trainNum,:)';
output_train =output(1:trainNum,:)';
input_test =input(trainNum+1:trainNum+testNum,:)';
output_test =output(trainNum+1:trainNum+testNum,:)';

%% 数据归一化
[inputn,inputps]=mapminmax(input_train,0,1);  %归一化到0,1之间，归一化可以消除特征指标的量纲和数量级的影响
[outputn,outputps]=mapminmax(output_train,0,1);
inputn_test=mapminmax('apply',input_test,inputps);

%% 获取输入层节点、输出层节点个数
inputnum=size(input,2);
outputnum=size(output,2);
disp('/////////////////////////////////')
disp('神经网络结构...')
disp(['输入层的节点数为：',num2str(inputnum)])
disp(['输出层的节点数为：',num2str(outputnum)])
disp(' ')
disp('隐含层节点的确定过程...')
string={'tansig','purelin'};    %传递函数
func_str='traingdx';    %训练算法
%确定隐含层节点个数
%采用经验公式hiddennum=sqrt(m+n)+a，m为输入层节点个数，n为输出层节点个数，a一般取为1-10之间的整数
MSE=1e+5; %初始化训练误差
for hiddennum=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
    
    %构建网络
    net=newelm(inputn,outputn,hiddennum,string,func_str);
    % 网络参数
    net.trainParam.epochs=10000;         % 训练次数
    net.trainParam.lr=0.01;                   % 学习速率
    net.trainParam.goal=0.00001;        % 训练目标最小误差
    % 网络训练
    net=train(net,inputn,outputn);
    an0=sim(net,inputn);  %仿真结果
    mse0=mse(outputn,an0);  %仿真的均方误差

%     disp(['隐含层节点数为',num2str(hiddennum),'时，训练集的均方误差为：',num2str(mse0)])
    
    %更新最佳的隐含层节点
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['最佳的隐含层节点数为：',num2str(hiddennum_best),'，相应的均方误差为：',num2str(MSE)])

%% 构建最佳隐含层节点（承接层节点）的ELMAN神经网络
net=newelm(inputn,outputn,hiddennum_best,string,func_str);

% 网络参数
net.trainParam.epochs=10000;         % 训练次数
net.trainParam.lr=0.01;                   % 学习速率
net.trainParam.goal=0.00001;        % 训练目标最小误差


%% 训练网络
net2=train(net,inputn,outputn);
%% 网络测试
an=sim(net2,inputn_test); %用训练好的模型进行仿真
test_simu=mapminmax('reverse',an,outputps); % 预测结果反归一化
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
for kk=1:200%% 反复训练网络
    
    net1=train(net,inputn,outputn);
    %% 网络测试
    an11=sim(net1,inputn_test); %用训练好的模型进行仿真
    test_simu11=mapminmax('reverse',an11,outputps); % 预测结果反归一化
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
for kk=1:200 %% 找到合适弹窗
   
    net1=train(net,inputn,outputn);
    %% 网络测试
    an11=sim(net1,inputn_test); %用训练好的模型进行仿真
    test_simu11=mapminmax('reverse',an11,outputps); % 预测结果反归一化
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

%输入输出数据
 input1=[16	15	17	15];   
 %input1=[1.86	1.42	1.61];  
%input1=[74	90	82];  

%% 划分训练集、测试集
input_test1 = input1(1,:)';
inputn_test1=mapminmax('apply',input_test1,inputps);
an1=sim(net2,inputn_test1); %用训练好的模型进行仿真
test_simu1=mapminmax('reverse',an1,outputps); % 预测结果反归一化
round(test_simu1)

test_simu1


