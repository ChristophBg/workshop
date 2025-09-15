#' Simulate EMA data
#'  - 20 individuals
#'  - 6 prompts/day for 14 days (2 weeks) => 84 prompts/person
#'  - Person-specific missingness (avg response ≈ 80%)
#'  - Variables:
#'     * positive_mood, negative_mood: Normal, truncated to [0,100]
#'       (group means ≈ 60 and ≈ 30), multilevel via person means
#'     * phys_activity: zero-inflated Beta on [0,100], multilevel (person mean & zero prob)
#'     * with_others: yes/no (Bernoulli), multilevel via person rate
#'  - All variables are NA when the prompt is missed
#'
#' @param seed Integer seed for reproducibility
#' @return the simulated EMA dataset
sim_EMA_data = function(seed=1234){

set.seed(seed)

library(dplyr)
library(tidyr)
library(purrr)

# ---------- Design ----------
N_id    <- 20
n_days  <- 14
n_prompts_day <- 6
n_prompts <- n_days * n_prompts_day

# ---------- Helper functions ----------
# Truncate to [0, 100]
cap01 <- function(x, lo = 0, hi = 100) pmin(pmax(x, lo), hi)

# Draw from zero-inflated Beta on [0, 100]
r_zib_0_100 <- function(n, zero_prob, mean_01, phi) {
  # zero_prob: scalar or length-n vector in [0,1]
  # mean_01:   scalar or length-n vector in (0,1)
  # phi:       precision (>0)
  z <- rbinom(n, 1, zero_prob) # 1 => structural zero
  # Beta params
  alpha <- mean_01 * phi
  beta  <- (1 - mean_01) * phi
  # Safe guards
  alpha[alpha <= 0] <- 1e-6
  beta[beta <= 0]   <- 1e-6
  y <- ifelse(z == 1, 0, rbeta(n, alpha, beta) * 100)
  y
}

# ---------- Person-level parameters (random effects) ----------
id <- factor(seq_len(N_id))

# Person-specific response probability ~ Beta with mean ~0.80
# Beta(a,b): mean = a/(a+b). Choose a=16, b=4 to centre at 0.80 with moderate heterogeneity.
resp_prob <- rbeta(N_id, shape1 = 16, shape2 = 4)

# Positive/negative mood person means (Normal around 60 / 40), SD controls between-person spread
mu_pos_person <- rnorm(N_id, mean = 60, sd = 8)
mu_neg_person <- rnorm(N_id, mean = 40, sd = 12)

# Within-person SDs (can be person-specific if desired; here fixed)
sd_pos_within <- 12
sd_neg_within <- 10

# Phys activity (zero-inflated Beta):
#  - person mean on 0..100, then mapped to (0,1)
#  - person zero-inflation probability (probability of exact zero)
mu_pa_person_0_100 <- cap01(rnorm(N_id, mean = 35, sd = 12))
mu_pa_person_01    <- pmin(pmax(mu_pa_person_0_100 / 100, 1e-4), 1 - 1e-4)
pi_zero_person     <- pmin(pmax(rbeta(N_id, 2, 5), 0 ), 1) # mean ~0.29 zeros; adjust as needed
phi_pa <- 10  # precision (higher => less within-person spread)

# Social context (with_others): person-specific probability (mean ~0.4)
p_with_others <- rbeta(N_id, 4, 6)

person_df <- tibble(
  id,
  resp_prob,
  mu_pos_person,
  mu_neg_person,
  mu_pa_person_01,
  pi_zero_person,
  p_with_others
)

# ---------- Prompt grid ----------
grid <- expand_grid(
  id  = id,
  day = seq_len(n_days),
  prompt = seq_len(n_prompts_day)
) %>%
  arrange(id, day, prompt) %>%
  mutate(prompt_index = (day - 1) * n_prompts_day + prompt)

# ---------- Simulate responses (missingness) ----------
# Bernoulli per prompt with person-specific probability
grid <- grid %>%
  left_join(person_df, by = "id") %>%
  rowwise() %>%
  mutate(
    responded = as.integer(rbinom(1, 1, resp_prob))
  ) %>%
  ungroup()

# ---------- Simulate variables only where responded == 1 ----------
# Positive/negative mood (Normal, truncated to [0,100])
# phys_activity: zero-inflated beta on [0,100]
# with_others: Bernoulli
sim_df <- grid %>%
  mutate(
    positive_mood = ifelse(
      responded == 1,
      cap01(rnorm(n(), mean = mu_pos_person, sd = sd_pos_within)),
      NA_real_
    ),
    negative_mood = ifelse(
      responded == 1,
      cap01(rnorm(n(), mean = mu_neg_person, sd = sd_neg_within)),
      NA_real_
    ),
    phys_activity = ifelse(
      responded == 1,
      r_zib_0_100(
        n = n(),
        zero_prob = pi_zero_person,
        mean_01   = mu_pa_person_01,
        phi       = phi_pa
      ),
      NA_real_
    ),
    with_others = ifelse(
      responded == 1,
      rbinom(n(), 1, p_with_others),
      NA_integer_
    )
  ) %>%
  select(id, day, prompt, prompt_index, responded,
         positive_mood, negative_mood, phys_activity, with_others)

# ---------- Quick checks ----------
# Average response rate overall and per person
overall_resp_rate <- mean(sim_df$responded)
resp_rate_by_id <- sim_df %>%
  group_by(id) %>%
  summarise(resp_rate = mean(responded)) %>%
  arrange(resp_rate)

overall_resp_rate
resp_rate_by_id

# Cap means for moods (sanity check)
sim_df %>%
  summarise(
    mean_pos = mean(positive_mood, na.rm = TRUE),
    mean_neg = mean(negative_mood, na.rm = TRUE)
  )

# Zero proportion for phys_activity (should be near mean(pi_zero_person))
mean(sim_df$phys_activity == 0, na.rm = TRUE)

# Distribution of with_others
sim_df %>%
  summarise(
    with_others_rate = mean(with_others, na.rm = TRUE)
  )

# ---------- Result ----------
# sim_df is your EMA dataset:
#  - One row per (id, day, prompt)
#  - 'responded' indicates whether values are observed (1) or missing (0)
#  - Variables are NA when not responded
#  - All variables multilevel via person-level parameters

#head(sim_df, 20)
sim_df
}

#write.csv(sim_df,"simulated_EMA_data.csv")
