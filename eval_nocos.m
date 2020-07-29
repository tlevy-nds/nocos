function [posteriorProbability, x] = eval_nocos(bun, age, neutrophil, spo2, rcdw, sodium, survival)
if ~exist('survival', 'var') || isempty(survival)
    % if survival is true, return P, otherwise, return 1-P
    survival = true;
end
%% Standardize the input data and evaluate the trained linear regression model
% mean of each measurement from the training set
mu_ = [25.216167993412; 63.603301312407; 136.390009606148; 96.1058329949926; 13.9274288050143; 6.33864124449337];
% standard deviation of each measurement from the training set
sigma_ = [23.956029968784; 16.4691192040125;  6.18940329487471; 3.78873040592358; 1.8548775509972; 3.85785697600178];
% model coefficients
coefficients = [-0.133620220069468; -0.094421748373521; -0.04012886950106; 0.0412242494490385; -0.033754630786386; -0.0326317691937443];
bias = 0.760179441464902;

% create vector of measurements
inputVector = [bun; age; sodium; spo2; rcdw; neutrophil];

% standardize x using a z-score
z = (inputVector - mu_) ./ sigma_;

z(isnan(z)) = 0;  % mean imputation

% perform linear regression using the model
x = sum(z .* coefficients) + bias;

%% Evaluate Bayes rule to compute the posterior probability
% model priors from the training set
priorSurvival = 0.880259775402517;
priorDeath = 0.119740224597483;
% likelihood function parameters from the training set
% Pareto tails and a Levy alpha-stable distribution (approximated as a quartic polynomial) in the center
% shape parameters (lower tail and upper tail for survival and death)
k1s = 0.0213031296865081;
k1d = -0.123596410738619;
k2s = -0.367357393601326;
k2d = -0.141063702070723;
% scale parameters (lower tail and upper tail for survival and death)
sigma1s = 0.153599247950597;
sigma1d = 0.303794740547108;
sigma2s = 0.0880884786111088;
sigma2d = 0.0979826205902811;
% threshold parameters (lower tail and upper tail for survival and death)
theta1s = 0.734223049257176;
theta1d = 0.380636296498394;
theta2s = 0.976299538590073;
theta2d = 0.749176906772995;
% quantiles
p1s = 0.3;  
p1d = 0.3;
p2s = 0.85;
p2d = 0.85;

if x < theta1s
    likelihoodSurvival = p1s*(1/sigma1s)*(1+k1s*(theta1s-x)/sigma1s)^(-1-1/k1s);
elseif x < theta2s
    z = (x - 0.855261293923624) / 0.070928824473147;
    likelihoodSurvival = 0.0161781027606685*z^4 - 0.0125273411680064*z^3 - 0.301885311044446 * z^2 - 0.0143551112333419*z + 2.53758792307174;
else
    likelihoodSurvival = (1-p2s)*(1/sigma2s)*(1+k2s*(x-theta2s)/sigma2s)^(-1-1/k2s);
end
likelihoodSurvival = real(likelihoodSurvival);

if x < theta1d
    likelihoodDeath = p1d*(1/sigma1d)*(1+k1d*(theta1d-x)/sigma1d)^(-1-1/k1d);
elseif x < theta2d
    z = (x-0.564818148249197) / 0.107931193731143;
    likelihoodDeath = 0.0102540832269135*z^4 - 0.0216265985226132*z^3 - 0.201326417582333*z^2 + 0.0746807824618428*z + 1.67058805959226;
else
    likelihoodDeath = (1-p2d)*(1/sigma2d)*(1+k2d*(x-theta2d)/sigma2d)^(-1-1/k2d);
end
likelihoodDeath = real(likelihoodDeath);

posteriorProbability = likelihoodSurvival * priorSurvival / ...
    (likelihoodSurvival * priorSurvival + likelihoodDeath * priorDeath);

% The posterior probability can be clipped at 0.1 on the low end 0.95 on the high end
posteriorProbability = min(max(posteriorProbability, 0.1), 0.95);

if ~survival
    % convert to a mortality calculator
    posteriorProbability = 1 - posteriorProbability;
end