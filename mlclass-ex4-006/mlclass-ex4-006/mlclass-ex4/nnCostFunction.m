function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% recode label y to one of k type
Y = zeros(m, num_labels);
for i = 1:m
    Y(i, y(i)) = 1;
end


a_1 = [ones(m, 1) X];
z_2 = a_1*Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(m, 1) a_2];
z_3 = a_2*Theta2';
a_3 = sigmoid(z_3);

% cost with regularization, loop on num_labels, based on assignment 3
% for j = 1:num_labels
%     y = Y(:, j);
%     pos = find(y == 1);
%     neg = find(y == 0);
%     sum_j = ( sum( -y(pos).*log(a_3(pos, j))) - sum((1-y(neg)).*log(1 - a_3(neg, j))) )/ m;
%     J = J + sum_j;
% end
% 
% J = J / 10;

% cost with regularization, loop on num of trianing examples
for i = 1:m
    J = J + sum( (-1) * Y(i, :) .* log(a_3(i, :)) - (1 - Y(i, :)) .* log(1 - a_3(i, :)));
end
J = J / m;

% get the grad without regularization
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

% implements backpropagation on each example
for i = 1:m
    % step 1, forward pass, which we can get from a_3
    
    % step 2, set delta_3 for each output unit k
    delta_3 = a_3(i, :) - Y(i, :); 
    delta_3 = delta_3'; % 10*1
    
    % step 3, set delta_2 based on delta_3
    temp = Theta2' * delta_3;
    delta_2 = temp(2:end, :) .* sigmoidGradient(z_2(i, :)'); % 25*1
    
    % step 4
    Delta2 = Delta2 + delta_3 * a_2(i, :); % 10*26
    Delta1 = Delta1 + delta_2 * a_1(i, :); % 25*401 
end

Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;

%----------------------------------------part 4----------------------------------

J = J + lambda * ( sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)) ) / 2 / m;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda * Theta1(:, 2:end) / m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda * Theta2(:, 2:end) / m;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
