import Bayesian_Network as BN
import pandas as pd
import numpy as np

""" rb = BN.initialize_bn()
kb,rb = BN.get_knowledge(rb)
print(kb) 
rb = BN.insert_evidence('MPP_CCF_1',1,'on',rb)
rb = BN.insert_evidence('MPP_CCF_2',1,'on',rb)
rb = BN.insert_evidence('MPP_CCF_3',1,'on',rb)
#rb = BN.insert_evidence('Competence_a_limplication',1,'off',rb)
kb,rb = BN.get_knowledge(rb)
print(kb)  """

data = pd.read_csv('rawData.csv',sep=";")
responses = data.values
questions = list(data.columns.values)
print(questions)
ind = list(range(48))
np.random.shuffle(ind)
print(ind)
responses = responses[:,ind]
kb_time = []
kb=[]
for student in range(len(responses)):
    rb = BN.initialize_bn()
    for question,rep in zip(questions, responses[student]):
        if rep == 1 :
            rb = BN.insert_evidence(question,1,'on',rb)
        else :
            rb = BN.insert_evidence(question,1,'off',rb)
        temp, rb = BN.get_knowledge(rb)
        kb.append(temp)
    kb_time.append(kb)
    kb=[]

df = pd.DataFrame(kb_time)
df.to_csv('bn_data.csv') 

