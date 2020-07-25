function posteriorProbability = eval_nocos(bun, age, neutrophil, spo2, rcdw, sodium)
% Would it be a good idea to separately verify?

%% Standardize the input data and evaluate the trained linear regression model
% mean of each measurement from the training set
mu_ = [25.3426973433329; 63.5685514135602; 136.325537053515; 96.0267968945655; 13.9384770804395; 6.40012139023765];
% standard deviation of each measurement from the training set
sigma_ = [23.9802630804946; 16.3034649370891; 6.19181239135278; 3.92750269491352; 1.85445631179296;  3.9309701012879];
% model coefficients
coefficients = [-0.121049007783428;  -0.091554806460523; -0.0386941891976109; 0.0321739916744435; -0.026911690410463; -0.0241198758542206];
bias = 0.778583937953465;

% create vector of measurements
inputVector = [bun; age; sodium; spo2; rcdw; neutrophil];

% standardize x using a z-score
z = (inputVector - mu_) ./ sigma_;

% perform linear regression using the model
x = sum(z .* coefficients) + bias;

%% Evaluate Bayes rule to compute the posterior probability
% model priors from the training set
priorSurvival = 0.889291968976733;
priorDeath = 0.110708031023267;
% likelihood function parameters from the training set
% Pareto tails and a Levy alpha-stable distribution (approximated as a quartic polynomial) in the center
% shape parameters (lower tail and upper tail for survival and death)
k1s = 0.0317729750140431;
k1d = -0.113138604379259;
k2s = -0.329144258574277;
k2d = -0.123313229001699;
% scale parameters (lower tail and upper tail for survival and death)
sigma1s = 0.138800316294804;
sigma1d = 0.267439621088245;
sigma2s = 0.0790978163103251;
sigma2d = 0.0872248666843174;
% threshold parameters (lower tail and upper tail for survival and death)
theta1s = 0.749462113399424;
theta1d = 0.437053090637476;
theta2s = 0.97412673479784;
theta2d = 0.769367239119786;
% quantiles
p1s = 0.3;  
p1d = 0.3;
p2s = 0.85;
p2d = 0.85;

if x < theta1s
    likelihoodSurvival = p1s*(1/sigma1s)*(1+k1s*(theta1s-x)/sigma1s)^(-1-1/k1s);
elseif x < theta2s
    likelihoodSurvival = 919.12*x^4 - 3207.4*x^3 + 4121.9*x^2 - 2311.8*x + 479.64;
else
    likelihoodSurvival = (1-p2s)*(1/sigma2s)*(1+k2s*(x-theta2s)/sigma2s)^(-1-1/k2s);
end

if x < theta1d
    likelihoodDeath = p1d*(1/sigma1d)*(1+k1d*(theta1d-x)/sigma1d)^(-1-1/k1d);
elseif x < theta2d
    likelihoodDeath = 118.4*x^4 - 310.24*x^3 + 279.57*x^2 - 101.84*x + 13.975;

else
    likelihoodDeath = (1-p2d)*(1/sigma2d)*(1+k2d*(x-theta2d)/sigma2d)^(-1-1/k2d);
end

posteriorProbability = likelihoodSurvival * priorSurvival / ...
    (likelihoodSurvival * priorSurvival + likelihoodDeath * priorDeath);

% The posterior probability can be clipped at 0.1 on the low end 0.95 on the high end
posteriorProbability = min(max(posteriorProbability, 0.1), 0.95);