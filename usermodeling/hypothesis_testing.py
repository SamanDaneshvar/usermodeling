from pandas import DataFrame
from matplotlib import pyplot as plt
from scipy.stats import normaltest
from scipy.stats import ttest_ind

results = DataFrame()
# experimentNames = ('N-grams', 'N-grams, LSA 300')
# results[experimentNames[0]] = [0.8, 0.82222222, 0.80555556, 0.82222222, 0.82222222, 0.79444444, 0.87777778, 0.85, 0.81666667, 0.77777778]
# results[experimentNames[1]] = [0.82777778, 0.84444444, 0.84444444, 0.83888889, 0.81111111, 0.8, 0.88888889, 0.82777778, 0.78333333, 0.81666667]

# # %%% TEMP: Testing
# tempList = []
# for item in results[experimentNames[0]]:
#     tempList.append(item-0.02)
# results[experimentNames[0]] = tempList

# experimentNames = ('Word and char n-grams, No LSA (English)', 'Word n-grams, No LSA (English)')
# results[experimentNames[0]] = [0.8, 0.79444444, 0.82222222, 0.87222222, 0.76111111, 0.85, 0.82222222, 0.85555556, 0.81111111, 0.81111111]
# results[experimentNames[1]] = [0.8, 0.78333333, 0.81111111, 0.85, 0.77222222, 0.81666667, 0.80555556, 0.82777778, 0.8, 0.8]

experimentNames = ('Word and char n-grams, No LSA (Arabic)', 'Char n-grams, No LSA (Arabic)')
results[experimentNames[0]] = [0.83333333, 0.77777778, 0.81111111, 0.76666667, 0.81111111, 0.87777778, 0.81111111, 0.77777778, 0.81111111, 0.75555556]
results[experimentNames[1]] = [0.77777778, 0.75555556, 0.83333333, 0.74444444, 0.78888889, 0.82222222, 0.82222222, 0.75555556, 0.77777778, 0.71111111]

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
for experimentName in experimentNames:
    statistic, p = normaltest(results[experimentName])
    # ↳ Null hypothesis: the sample comes from a normal distribution
    print("{}: (skewtest z-score)^2 + (kurtosistest z-score)^2 = {}, p = {}".format(experimentName, statistic, p))
    if p < alpha:
        # The null hypothesis can be rejected
        print("No: It is unlikely that the sample comes from a Gaussian (normal) distribution")
    else:
        print("Yes: It is likely that the sample comes from a Gaussian (normal) distribution")

# T-test: Compare means for Gaussian samples
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
alpha = 0.05
print("Confidence level = {}%".format((1-alpha)*100))
equalVariances = True
statistic, p = ttest_ind(results[experimentNames[0]], results[experimentNames[1]], equal_var=equalVariances)
# ↳ Null hypothesis: The two samples have identical average (expected) values.
if equalVariances == True:
    statisticName = "t-test statistic"
else:
    statisticName = "Welch’s t-test statistic"

print("{} = {}, p = {}".format(statisticName, statistic, p))
if p < alpha:
    # The null hypothesis can be rejected
    print("Significant difference between the means: Samples are likely drawn from different distributions")
else:
    print("No significant difference between the means: Samples are likely drawn from the same distribution")
