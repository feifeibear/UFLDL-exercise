function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta

% fprintf('In softmaxCost.m the data size is %d, %d', size(data, 1), size(data, 2) );

theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1)); % numClass*M
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.


[n, m] = size(data);
M = theta * data; % numClasses * M
M = bsxfun(@minus, M, max(M, [], 1)); % devide minus the max of the row
fenzi = exp(M);
P = bsxfun(@rdivide, fenzi, sum(fenzi)); % devide each column by column sum

thetagrad = -1/m*( (groundTruth-P)*data' ) + lambda*theta;

cost = -1/m*sum(sum(log(P) .* groundTruth)) + lambda/2 * sum(sum(theta.^2));

% fprintf('end calculate J_theta\n');
% pause







% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

