%% BP例题
X = [1,0,1]';
W1 = [ 0.2,0.4,-0.5;
      -0.3,0.1, 0.2]';
W2 = [-0.3,-0.2]';
b1= [-0.4,0.2]';
b2= [0.1]';

Z = logsig(W1'*X+b1);
Y = logsig(W2'*Z+b2);
L = 1;
e = L-Y;
E = 0.5*e'*e;

nablaW2E = -Z*(Y.*(1-Y).*e)';
nablaZE = -W2*(Y.*(1-Y).*e);

nablaW1E = -X*(Z.*(1-Z).*-nablaZE)';
nablaXE = -W1*(Z.*(1-Z).*nablaZE);

W1 = W1-0.9*nablaW1E;
W2 = W2-0.9*nablaW2E;



%% 通用神经网络
clear
clc
global Weight
dim = 50;
pos_num = 50;neg_num = 50;
X = [randn(dim, pos_num),0.5*randn(dim, neg_num)+3];
Y = [zeros(pos_num,1);ones(neg_num,1)];
randIndex = randperm(size(X,2));
X_train = X(:,randIndex);
Y_train = Y(randIndex);

hidden_size = [6,10,5];
output_size = 1;
iter = 1000;
mu = 0.99;
Training(X_train,Y_train,hidden_size,output_size,iter,mu)
pred_Y = FeedForward(X);

function Init(hidden_size,input_size,output_size)
    num_hidden = length(hidden_size)+1;
    global Weight
    Weight = cell(num_hidden,2);
    layer_size = [input_size,hidden_size,output_size];
    for i = 1:num_hidden
        Weight{i,1} = randn(layer_size(i),layer_size(i+1));
        Weight{i,2} = randn(layer_size(i+1),1);
    end
end
function Training(X,Y,hidden_size,output_size,iter,mu)

    [input_size, batch_size] = size(X);
    Init(hidden_size,input_size,output_size);
    for i = 1:iter
        pred = FeedForward(X);
        [loss,diffE] = ErrFunc(pred,Y);
        Backward(X,diffE,mu)
        fprintf('The loss of iter %d is %8f\n',i,loss);
    end
end
function pred = FeedForward(X)
    global Weight
    global Hidden
    num_hid = length(Weight);
    Hidden = cell(num_hid,2);    
    [Z,diffZ] = ActivFunc(Weight{1,1}'*X + Weight{1,2});
    Hidden{1,1} = Z; Hidden{1,2} = diffZ; 
    for i = 2:num_hid
        [Z,diffZ] = ActivFunc(Weight{i,1}'*Hidden{i-1,1} + Weight{i,2});
        Hidden{i,1} = Z; Hidden{i,2} = diffZ; 
    end
    pred = Hidden{num_hid,1};
end
function Backward(X,diffE,mu)
    global Weight
    global Hidden
    global Grad
    num_hid = length(Weight);
    Grad = cell(num_hid,3); 
%     分别是对W偏导，b的偏导，z的偏导
    Grad{num_hid,2} = -(Hidden{num_hid,2}.*diffE);
    Grad{num_hid,1} = Hidden{num_hid-1,1}*Grad{num_hid,2}';
    Grad{num_hid,3} = Weight{num_hid,1}*Grad{num_hid,2};
    Weight{num_hid,1} = Weight{num_hid,1}- mu*Grad{num_hid,1};
    Weight{num_hid,2} = Weight{num_hid,2}- mu*Grad{num_hid,2};
    for i = num_hid-1:-1:2
        Grad{i,2} = -(Hidden{i,2}.*(-Grad{i+1,3}));
        Grad{i,1} = Hidden{i-1,1}*Grad{i,2}';
        Grad{i,3} = -Weight{i,1}*Grad{i,2}; 
        Weight{i,1} = Weight{i,1}- mu*Grad{i,1};
        Weight{i,2} = Weight{i,2}- mu*Grad{i,2};
    end
    i=1;
    Grad{i,2} = -(Hidden{i,2}.*(-Grad{i+1,3}));
    Grad{i,1} = X*Grad{i,2}';
    Grad{i,3} = -Weight{i,1}*Grad{i,2};
    Weight{i,1} = Weight{i,1}- mu*Grad{i,1};
    Weight{i,2} = Weight{i,2}- mu*Grad{i,2};    
end
function [Z,diffZ] = ActivFunc(X)
    Z = logsig(X);
    diffZ = logsig(X).*(1-logsig(X));
end
function [E,diffE] = ErrFunc(pred,label)
% % softmax
%     soft_pred = exp(pred)./sum(exp(pred));
%     num = length(label);
%     pred_ = zeros(num,1);
%     for i = 1:num
%         pred_(i) = soft_pred(label(i),i);
%     end
%     e = ones(num,1) - pred_;
%     E = 1/2*e'*e;
%     diffE = ???
% % 1 output
    e = label' - pred;
    E = 1/2*e*e';
    diffE = e;
end