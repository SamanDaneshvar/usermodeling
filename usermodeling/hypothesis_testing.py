from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.stats import normaltest
from scipy.stats import ttest_ind

results = DataFrame()


# experiment_names = ('N-grams', 'N-grams, LSA 300')
# results[experiment_names[0]] = [0.8, 0.82222222, 0.80555556, 0.82222222, 0.82222222, 0.79444444, 0.87777778, 0.85, 0.81666667, 0.77777778]
# results[experiment_names[1]] = [0.82777778, 0.84444444, 0.84444444, 0.83888889, 0.81111111, 0.8, 0.88888889, 0.82777778, 0.78333333, 0.81666667]
#
# # %%% TEMP: Testing
# temp_list = []
# for item in results[experiment_names[0]]:
#     temp_list.append(item - 0.02)
# results[experiment_names[0]] = temp_list
#
# experiment_names = ('Word and char n-grams, No LSA (English)', 'Word n-grams, No LSA (English)')
# results[experiment_names[0]] = [0.8, 0.79444444, 0.82222222, 0.87222222, 0.76111111, 0.85, 0.82222222, 0.85555556, 0.81111111, 0.81111111]
# results[experiment_names[1]] = [0.8, 0.78333333, 0.81111111, 0.85, 0.77222222, 0.81666667, 0.80555556, 0.82777778, 0.8, 0.8]


experiment_names = ('Word and char n-grams, No LSA (Arabic)', 'Char n-grams, No LSA (Arabic)')
results[experiment_names[0]] = [0.83333333, 0.77777778, 0.81111111, 0.76666667, 0.81111111, 0.87777778, 0.81111111, 0.77777778, 0.81111111, 0.75555556]
results[experiment_names[1]] = [0.77777778, 0.75555556, 0.83333333, 0.74444444, 0.78888889, 0.82222222, 0.82222222, 0.75555556, 0.77777778, 0.71111111]

# Descriptive stats
print(results.describe())

# Box and whisker plot
results.boxplot()
plt.show()

# Histogram plot
results.hist()
plt.show()

# Normality test
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
alpha = 0.05
print("Confidence level = {}%".format((1-alpha)*100))
for experiment_name in experiment_names:
    statistic, p = normaltest(results[experiment_name])
    # ↳ Null hypothesis: the sample comes from a normal distribution
    print("{}: (skewtest z-score)^2 + (kurtosistest z-score)^2 = {}, p = {}".format(experiment_name, statistic, p))
    if p < alpha:
        # The null hypothesis can be rejected
        print("No: It is unlikely that the sample comes from a Gaussian (normal) distribution")
    else:
        print("Yes: It is likely that the sample comes from a Gaussian (normal) distribution")

# T-test: Compare means for Gaussian samples
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
alpha = 0.05
print("Confidence level = {}%".format((1-alpha)*100))
equal_variances = True
statistic, p = ttest_ind(results[experiment_names[0]], results[experiment_names[1]], equal_var=equal_variances)
# ↳ Null hypothesis: The two samples have identical average (expected) values.
if equal_variances:
    statistic_name = "t-test statistic"
else:
    statistic_name = "Welch’s t-test statistic"

print("{} = {}, p = {}".format(statistic_name, statistic, p))
if p < alpha:
    # The null hypothesis can be rejected
    print("Significant difference between the means: Samples are likely drawn from different distributions")
else:
    print("No significant difference between the means: Samples are likely drawn from the same distribution")
