# ttest.py
# Runs paired t-tests on model performance results (FFN vs CNN vs Ridge).
# Helps evaluate statistical significance of differences between models.

import pandas as pd 
import scipy.stats as st 

ridge=pd.read_csv('results/ridge.tsv', index_col=0, sep='\t')
cnn=pd.read_csv('results/cnn.tsv', index_col=0, sep='\t')
ffn=pd.read_csv('results/ffn.tsv', index_col=0, sep='\t')

# print(ridge)
print()
print('Two sided t-test for paired samples (CNN vs Ridge):')

for var in ['accuracy']:
	print('{} : {}'.format(var, round(st.ttest_rel(ridge[var], cnn[var])[1],4)))

print()
print('Two sided t-test for paired samples (CNN vs FFN):')

for var in ['accuracy']:
	print('{} : {}'.format(var, round(st.ttest_rel(ffn[var], cnn[var])[1],4)))

print()
print('Two sided t-test for paired samples (FFN vs Ridge):')
for var in ['accuracy']:
	print('{} : {}'.format(var, round(st.ttest_rel(ridge[var], ffn[var])[1],4)))