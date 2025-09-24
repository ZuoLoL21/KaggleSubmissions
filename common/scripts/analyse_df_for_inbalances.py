# for each categorical feature, check the counts of each
# send warning if distribution has a value of < xyz
# for each tuple of categorical features, check the counts of each (X, Y) tuple
# send warning if distribution is wack
"""
eg:
for X=x, Y=y, T=0 has 100 data points,
for X=x, Y=y, T=1 has 120 data points,
for X=x, Y=y, T=2 has 2 data points,

WARNING: feature x and y (T=2 under-represented)
"""

# for each continuous feature, use the counts
"""
eg:
for the feature X, 
	T=0 has 100 data points
	T=1 has 120 data points

"""

# maybe analyse distribution?
# find the relationship between 2 continuous features? 3d plots?
# find the relationship between a continuous feature and a categorical feature?
