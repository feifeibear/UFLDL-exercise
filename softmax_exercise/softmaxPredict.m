function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta

theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix

% fprintf('In softmaxPredict.m the data size is %d, %d', size(data, 1), size(data, 2) );
% fprintf('In softmaxPredict.m the theta size is %d, %d', size(theta, 1), size(theta, 2) );

pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

[n, m] = size(data);
M = theta * data; % numClasses * M
M = bsxfun(@minus, M, max(M, [], 1)); % devide minus the max of the row
fenzi = exp(M);
P = bsxfun(@rdivide, fenzi, sum(fenzi)); % devide each column by column sum

[dump, pred] = max(P); 

% ---------------------------------------------------------------------q

end

