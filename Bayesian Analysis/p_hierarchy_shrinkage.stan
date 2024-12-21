data {
  int<lower=0> N;             // number of data items
  int<lower=0> J;             // number of predictors
  int<lower=1> G;             // number of groups
  int<lower=1, upper=G> group[N];  // group index for each data item
  vector[J] x[N];             // predictor matrix (NxJ)
  vector[N] y;                // univariate response vector
  
  // Hyperprior parameters
  real<lower=0> scale_icept;  // prior std for the intercept
  real<lower=0> scale_global; // scale for the half-t prior for tau
  
  real<lower=1> nu_global;
  real<lower=1> nu_local;
}

parameters {
  // Residual and group-level intercepts and slopes
  real logsigma;              // residual standard deviation (log-scale)
  vector[G] alpha_raw;        // raw intercepts for each group (non-centered)
  matrix[G, J] beta_raw;      // raw beta coefficients for non-centered parameterization

  // Hyperparameters for hierarchical priors
  real mu_alpha;              // overall mean intercept
  real<lower=0> tau_alpha;    // standard deviation of group-level intercepts
  vector[J] mu_beta;          // overall mean for group-level betas
  vector<lower=0>[J] tau_beta; // standard deviation for group-level betas

  // Local and global shrinkage parameters
  vector[J] z;                // shrinkage factor
  real<lower=0> r1_global;
  real<lower=0> r2_global;
  
  vector<lower=0>[J] r1_local;
  vector<lower=0>[J] r2_local;
}

transformed parameters {
  real<lower=0> sigma;         // residual standard deviation
  matrix[G, J] beta;           // actual group-level beta coefficients
  vector[G] alpha;             // actual intercepts for each group
  
  vector<lower=0>[J] lambda;   // local shrinkage factor
  real<lower=0> tau;           // global shrinkage factor

  sigma = exp(logsigma);
  lambda = r1_local .* sqrt(r2_local);
  tau = r1_global * sqrt(r2_global);

  // Transform raw parameters to obtain actual group-level coefficients
  for (g in 1:G) {
    alpha[g] = mu_alpha + tau_alpha * alpha_raw[g];
    for (j in 1:J) {
      beta[g, j] = (mu_beta[j] + tau_beta[j] * beta_raw[g, j]) * lambda[j] * tau;
    }
  }
}

model {
  // Priors for hyperparameters
  mu_alpha ~ normal(0, scale_icept);          // prior for overall intercept mean
  tau_alpha ~ normal(0, 2);                   // prior for intercept SD across groups
  mu_beta ~ normal(0, 2);                     // prior for mean of beta coefficients across groups
  tau_beta ~ normal(0, 2);                    // prior for SD of beta coefficients across groups
  
  // Priors for shrinkage parameters
  z ~ normal(0, 1);
  r1_local ~ normal(0.0, 0.1);
  r2_local ~ inv_gamma(0.5 * nu_local, 0.5 * nu_local);
  
  r1_global ~ normal(0.0, scale_global * sigma);
  r2_global ~ inv_gamma(0.5 * nu_global, 0.5 * nu_global);

  // Priors for group-level raw parameters (non-centered parameterization)
  alpha_raw ~ normal(0, 1);
  to_vector(beta_raw) ~ normal(0, 1);

  // Likelihood
  vector[N] mu;               // predicted values for response variable

  for (n in 1:N) {
    int g = group[n];         // group index for the observation
    mu[n] = alpha[g] + dot_product(beta[g], x[n]); // group-specific linear predictor
  }

  y ~ normal(mu, sigma);      // likelihood with normal residuals
}

generated quantities {
  vector[N] y_pred;           // predicted values for y

  for (n in 1:N) {
    int g = group[n];         // group index for the observation
    y_pred[n] = normal_rng(alpha[g] + dot_product(beta[g], x[n]), sigma);
  }
}