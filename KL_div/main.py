from math import log2

from numpy import asarray
import numpy as np
from scipy.special import rel_entr
from scipy.spatial import distance
import pandas as pd

def kl_div(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

def js_div(p, q):
	m = 0.5 * (p + q)
	return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

def normal(power, mean, std, pos):
    a = 1/(np.sqrt(2*np.pi)*std)
    diff = np.abs(np.power(pos-mean, power))
    b = np.exp(-(diff)/(2*std*std))
    return a*b

data = [['FP', 10, 'JP', 20], ['FP', 15, 'JP', 5], ['FP', 14, 'JP',  14], ['FP', 10, 'JP', 30], ['FP', 15, 'JP', 0],
['FP', 14, 'JP', 88], ['FP', 10, 'JP', 8], ['FP', 15, 'JP', 15], ['FP', 14, 'JP', 18], ['FP', 29, 'JP', 7]]

# Criando um dataframe do pandas
df = pd.DataFrame(data, columns=['Group1', 'Value Responsability 1', 'Group2', 'Value Responsability 2'])

# mostrando o dataframe
print(df)
print('------------------------------------------------------------------------------')

p = df.iloc[:,1]
q = df.iloc[:,3]

# define distributions
#p = asarray([0.10, 0.40, 0.50, 0.20, 0.40, 0.55, 0.20, 0.40])
#q = asarray([0.20, 0.40, 0.55, 0.20, 0.50, 0.85, 0.90, 0.40])

mean_p = np.mean(p)
mean_q = np.mean(q)

std_p = np.std(p)
std_q = np.std(q)

power = 3
val_1 = p
val_2 = q
pdf_1_array = []
pdf_2_array = []
array = np.arange(-2,2,0.1)

for i in array:

    pdf_1 = normal(power, mean_p, std_p, p)
    pdf_1_array.append(pdf_1)

for i in array:

    pdf_2 = normal(power, mean_q, std_q, q)
    pdf_2_array.append(pdf_2)


# calculate JS(P || Q)
js_pq = js_div(pdf_1, pdf_2)
print('JS(P || Q) divergence: %.3f bits' % js_pq)
#print('JS(P || Q) distance: %.3f' % sqrt(js_pq))
print('JS(P || Q) divergence using built-in function: %.3f bits' % distance.jensenshannon(pdf_1, pdf_2))

# calculate JS(Q || P)
js_qp = js_div(pdf_2, pdf_1)
print('JS(Q || P) divergence: %.3f bits' % js_qp)
#print('JS(Q || P) distance: %.3f' % sqrt(js_qp))
print('JS(P || Q) divergence using built-in function: %.3f bits' % distance.jensenshannon(pdf_2, pdf_1))

print('------------------------------------------------------------------------------')

kl_pq = kl_div(pdf_1, pdf_2)
print('KL(P || Q) divergence using my code: %.3f bits' % kl_pq)
#print('KL(P || Q) distance: %.3f' % sqrt(kl_pq))
print('KL(P || Q) divergence using built-in function: %.3f bits' % sum(rel_entr(pdf_1, pdf_2)))
# calculate JS(Q || P)
kl_qp = kl_div(pdf_2, pdf_1)
print('KL(P || Q) divergence using my code: %.3f bits' % kl_qp)
#print('KL(Q || P) distance: %.3f' % sqrt(kl_pq))
print('KL(P || Q) divergence using built-in function: %.3f bits' % sum(rel_entr(pdf_2, pdf_1)))



