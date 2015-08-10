function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_try = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]; % numbers for C and sigma try
pred_error = zeros(numel(C_try)*numel(C_try), 1);

for i = 1:numel(C_try)
    for j = 1:numel(C_try)
        C_ij = C_try(i);
        sigma_ij = C_try(j);
        model= svmTrain(X, y, C_ij, @(x1, x2) gaussianKernel(x1, x2, sigma_ij)); 
        predictions = svmPredict(model, Xval);
        pred_error((i-1)*numel(C_try) + j, 1) = mean(double(predictions ~= yval));
    end
end

[m, ind] = min(pred_error);
C = C_try(floor(ind / numel(C_try)));
sigma = C_try(mod(ind, numel(C_try)));
% =========================================================================

end
