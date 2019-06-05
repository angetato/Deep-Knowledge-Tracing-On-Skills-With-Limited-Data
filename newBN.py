from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

# create the nodes
a = BbnNode(Variable(0, 'Competence_a_limplication', ['on', 'off']), [0.5, 0.5])
b = BbnNode(Variable(1, 'Causal_Factuel', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
c = BbnNode(Variable(2, 'Abstrait', ['on', 'off']), [0.9, 0.1, 0.1, 0.9])
d = BbnNode(Variable(3, 'CF_Inhiber_PETnonQ', ['on', 'off']), [0.9, 0.1, 0.4, 0.6])
e = BbnNode(Variable(4, 'CF_gestion_3_modeles', ['on', 'off']), [0.9, 0.1, 0.4, 0.6])
f = BbnNode(Variable(5, 'A_Inhiber_PETnonQ', ['on', 'off']), [0.9, 0.1, 0.4, 0.6])
g = BbnNode(Variable(6, 'A_gestion_3_modeles', ['on', 'off']), [0.9, 0.1, 0.4, 0.6])
h = BbnNode(Variable(7, 'MPP_FMD', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
i = BbnNode(Variable(8, 'MTT_FMD', ['on', 'off']), [0.95, 0.05, 0.6, 0.4, 0.7, 0.3, 0.4, 0.6])
j = BbnNode(Variable(9, 'CF_genere_nonPETQ', ['on', 'off']), [0.9, 0.1, 0.2, 0.8])
k = BbnNode(Variable(10, 'A_genere_nonPETQ', ['on', 'off']), [0.9, 0.1, 0.2, 0.8])
l = BbnNode(Variable(11, 'AC_FMA', ['on', 'off']), [0.9, 0.1, 0.4, 0.6])
m = BbnNode(Variable(12, 'DA_FFA', ['on', 'off']), [0.9, 0.1, 0.4, 0.6])
n = BbnNode(Variable(13, 'MTT_A', ['on', 'off']), [0.95, 0.05, 0.6, 0.4, 0.3, 0.7, 0.4, 0.6])
u = BbnNode(Variable(20, 'MPP_A', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
o = BbnNode(Variable(14, 'AC_A', ['on', 'off']), [0.9, 0.1, 0.4, 0.6])
p = BbnNode(Variable(15, 'DA_A', ['on', 'off']), [0.9, 0.1, 0.4, 0.6])
q = BbnNode(Variable(16, 'MPP_FFD', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
r = BbnNode(Variable(17, 'MTT_FFD', ['on', 'off']), [0.95, 0.05, 0.6, 0.4, 0.7, 0.3, 0.3, 0.7])
s = BbnNode(Variable(18, 'AC_FFA', ['on', 'off']), [0.9, 0.1, 0.4, 0.6])
t = BbnNode(Variable(19, 'DA_FMA', ['on', 'off']), [0.9, 0.1, 0.4, 0.6])

a1 = BbnNode(Variable(73, 'Causal_Conter_Factuel', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
a2 = BbnNode(Variable(74, 'CCF_Inhiber_PETnonQ', ['on', 'off']), [0.9, 0.1, 0.4, 0.6])
a3 = BbnNode(Variable(75, 'CCF_genere_nonPETQ', ['on', 'off']), [0.9, 0.1, 0.2, 0.8])
a4 = BbnNode(Variable(76, 'CCF_gestion_3_modeles', ['on', 'off']), [0.9, 0.1, 0.4, 0.6])
a5 = BbnNode(Variable(21, 'MPP_CCF', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
a6 = BbnNode(Variable(22, 'MTT_CCF', ['on', 'off']), [0.95, 0.05, 0.6, 0.4, 0.7, 0.3, 0.3, 0.7])
a7 = BbnNode(Variable(23, 'AC_CCF', ['on', 'off']), [0.9, 0.1, 0.4, 0.6])
a8 = BbnNode(Variable(24, 'DA_CCF', ['on', 'off']), [0.9, 0.1, 0.4, 0.6])

#Beaucoup d'antécédents alternatifs
q1 = BbnNode(Variable(25, 'MPP_FMD_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q6 = BbnNode(Variable(26, 'MPP_FMD_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q10 = BbnNode(Variable(27, 'MPP_FMD_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])

q3 = BbnNode(Variable(28, 'MTT_FMD_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q5 = BbnNode(Variable(29, 'MTT_FMD_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q11 = BbnNode(Variable(30, 'MTT_FMD_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])

q2 = BbnNode(Variable(31, 'AC_FMA_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q8 = BbnNode(Variable(32, 'AC_FMA_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q9 = BbnNode(Variable(33, 'AC_FMA_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])

q4 = BbnNode(Variable(34, 'DA_FMA_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q7 = BbnNode(Variable(35, 'DA_FMA_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q12 = BbnNode(Variable(36, 'DA_FMA_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])

#Peu d'antécédents alternatifs
q15 = BbnNode(Variable(37, 'MPP_FFD_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q17 = BbnNode(Variable(38, 'MPP_FFD_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q23 = BbnNode(Variable(39, 'MPP_FFD_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])

q14 = BbnNode(Variable(40, 'MTT_FFD_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q18 = BbnNode(Variable(41, 'MTT_FFD_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q24 = BbnNode(Variable(42, 'MTT_FFD_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])

q16 = BbnNode(Variable(43, 'AC_FFA_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q19 = BbnNode(Variable(44, 'AC_FFA_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q21 = BbnNode(Variable(45, 'AC_FFA_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])

q13 = BbnNode(Variable(46, 'DA_FFA_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q20 = BbnNode(Variable(47, 'DA_FFA_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q22 = BbnNode(Variable(48, 'DA_FFA_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])


#Conséquent abstrait
q27 = BbnNode(Variable(49, 'MPP_A_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q32 = BbnNode(Variable(50, 'MPP_A_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q36 = BbnNode(Variable(51, 'MPP_A_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])

q28 = BbnNode(Variable(52, 'MTT_A_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q29 = BbnNode(Variable(53, 'MTT_A_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q35 = BbnNode(Variable(54, 'MTT_A_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])

q26 = BbnNode(Variable(55, 'AC_A_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q31 = BbnNode(Variable(56, 'AC_A_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q34 = BbnNode(Variable(57, 'AC_A_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])

q25 = BbnNode(Variable(58, 'DA_A_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q30 = BbnNode(Variable(59, 'DA_A_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
q33 = BbnNode(Variable(60, 'DA_A_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])


#Prémisse complètement abstrait
qCCF1 = BbnNode(Variable(61, 'MPP_CCF_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
qCCF2 = BbnNode(Variable(62, 'MPP_CCF_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
qCCF3 = BbnNode(Variable(63, 'MPP_CCF_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])

qCCF4 = BbnNode(Variable(64, 'MTT_CCF_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
qCCF5 = BbnNode(Variable(65, 'MTT_CCF_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
qCCF6 = BbnNode(Variable(66, 'MTT_CCF_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])

qCCF7 = BbnNode(Variable(67, 'AC_CCF_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
qCCF8 = BbnNode(Variable(68, 'AC_CCF_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
qCCF9 = BbnNode(Variable(69, 'AC_CCF_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])

qCCF10 = BbnNode(Variable(70, 'DA_CCF_1', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
qCCF11 = BbnNode(Variable(71, 'DA_CCF_2', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])
qCCF12 = BbnNode(Variable(72, 'DA_CCF_3', ['on', 'off']), [0.9, 0.1, 0.3, 0.7])



# create the network structure
bbn = Bbn() \
    .add_node(a) \
    .add_node(b) \
    .add_node(c) \
    .add_node(d) \
    .add_node(e) \
    .add_node(f) \
    .add_node(g) \
    .add_node(h) \
    .add_node(i) \
    .add_node(j) \
    .add_node(k) \
    .add_node(l) \
    .add_node(m) \
    .add_node(n) \
    .add_node(o) \
    .add_node(p) \
    .add_node(q) \
    .add_node(r) \
    .add_node(s) \
    .add_node(t) \
    .add_node(u) \
    .add_node(a1) \
    .add_node(a2) \
    .add_node(a3) \
    .add_node(a4) \
    .add_node(a5) \
    .add_node(a6) \
    .add_node(a7) \
    .add_node(a8) \
    .add_node(q1) \
    .add_node(q2) \
    .add_node(q3) \
    .add_node(q4) \
    .add_node(q5) \
    .add_node(q6) \
    .add_node(q6) \
    .add_node(q8) \
    .add_node(q9) \
    .add_node(q10) \
    .add_node(q11) \
    .add_node(q12) \
    .add_node(q13) \
    .add_node(q14) \
    .add_node(q15) \
    .add_node(q16) \
    .add_node(q17) \
    .add_node(q18) \
    .add_node(q19) \
    .add_node(q20) \
    .add_node(q21) \
    .add_node(q22) \
    .add_node(q23) \
    .add_node(q24) \
    .add_node(q25) \
    .add_node(q26) \
    .add_node(q27) \
    .add_node(q28) \
    .add_node(q29) \
    .add_node(q30) \
    .add_node(q31) \
    .add_node(q32) \
    .add_node(q33) \
    .add_node(q34) \
    .add_node(q35) \
    .add_node(q36) \
    .add_node(qCCF1) \
    .add_node(qCCF2) \
    .add_node(qCCF3) \
    .add_node(qCCF4) \
    .add_node(qCCF5) \
    .add_node(qCCF6) \
    .add_node(qCCF7) \
    .add_node(qCCF8) \
    .add_node(qCCF9) \
    .add_node(qCCF10) \
    .add_node(qCCF11) \
    .add_node(qCCF12) \
    .add_edge(Edge(a, b, EdgeType.DIRECTED)) \
    .add_edge(Edge(a, a1, EdgeType.DIRECTED)) \
    .add_edge(Edge(a1, a2, EdgeType.DIRECTED)) \
    .add_edge(Edge(a1, a4, EdgeType.DIRECTED)) \
    .add_edge(Edge(a4, a3, EdgeType.DIRECTED)) \
    .add_edge(Edge(a4, a6, EdgeType.DIRECTED)) \
    .add_edge(Edge(a3, a7, EdgeType.DIRECTED)) \
    .add_edge(Edge(a3, a8, EdgeType.DIRECTED)) \
    .add_edge(Edge(a, c, EdgeType.DIRECTED)) \
    .add_edge(Edge(b, d, EdgeType.DIRECTED)) \
    .add_edge(Edge(b, e, EdgeType.DIRECTED)) \
    .add_edge(Edge(c, f, EdgeType.DIRECTED)) \
    .add_edge(Edge(c, g, EdgeType.DIRECTED)) \
    .add_edge(Edge(d, h, EdgeType.DIRECTED)) \
    .add_edge(Edge(d, i, EdgeType.DIRECTED)) \
    .add_edge(Edge(e, i, EdgeType.DIRECTED)) \
    .add_edge(Edge(e, j, EdgeType.DIRECTED)) \
    .add_edge(Edge(g, k, EdgeType.DIRECTED)) \
    .add_edge(Edge(j, l, EdgeType.DIRECTED)) \
    .add_edge(Edge(j, m, EdgeType.DIRECTED)) \
    .add_edge(Edge(f, u, EdgeType.DIRECTED)) \
    .add_edge(Edge(f, n, EdgeType.DIRECTED)) \
    .add_edge(Edge(g, n, EdgeType.DIRECTED)) \
    .add_edge(Edge(k, o, EdgeType.DIRECTED)) \
    .add_edge(Edge(k, p, EdgeType.DIRECTED)) \
    .add_edge(Edge(d, q, EdgeType.DIRECTED)) \
    .add_edge(Edge(d, r, EdgeType.DIRECTED)) \
    .add_edge(Edge(e, r, EdgeType.DIRECTED)) \
    .add_edge(Edge(j, s, EdgeType.DIRECTED)) \
    .add_edge(Edge(j, t, EdgeType.DIRECTED)) \
    .add_edge(Edge(h, q1, EdgeType.DIRECTED)) \
    .add_edge(Edge(h, q6, EdgeType.DIRECTED)) \
    .add_edge(Edge(h, q10, EdgeType.DIRECTED)) \
    .add_edge(Edge(i, q3, EdgeType.DIRECTED)) \
    .add_edge(Edge(i, q5, EdgeType.DIRECTED)) \
    .add_edge(Edge(i, q11, EdgeType.DIRECTED)) \
    .add_edge(Edge(l, q2, EdgeType.DIRECTED)) \
    .add_edge(Edge(l, q8, EdgeType.DIRECTED)) \
    .add_edge(Edge(l, q9, EdgeType.DIRECTED)) \
    .add_edge(Edge(t, q4, EdgeType.DIRECTED)) \
    .add_edge(Edge(t, q7, EdgeType.DIRECTED)) \
    .add_edge(Edge(t, q12, EdgeType.DIRECTED)) \
    .add_edge(Edge(q, q15, EdgeType.DIRECTED)) \
    .add_edge(Edge(q, q17, EdgeType.DIRECTED)) \
    .add_edge(Edge(q, q23, EdgeType.DIRECTED)) \
    .add_edge(Edge(r, q14, EdgeType.DIRECTED)) \
    .add_edge(Edge(r, q18, EdgeType.DIRECTED)) \
    .add_edge(Edge(r, q24, EdgeType.DIRECTED)) \
    .add_edge(Edge(s, q16, EdgeType.DIRECTED)) \
    .add_edge(Edge(s, q19, EdgeType.DIRECTED)) \
    .add_edge(Edge(s, q21, EdgeType.DIRECTED)) \
    .add_edge(Edge(m, q13, EdgeType.DIRECTED)) \
    .add_edge(Edge(m, q20, EdgeType.DIRECTED)) \
    .add_edge(Edge(m, q22, EdgeType.DIRECTED)) \
    .add_edge(Edge(u, q27, EdgeType.DIRECTED)) \
    .add_edge(Edge(u, q32, EdgeType.DIRECTED)) \
    .add_edge(Edge(u, q36, EdgeType.DIRECTED)) \
    .add_edge(Edge(n, q28, EdgeType.DIRECTED)) \
    .add_edge(Edge(n, q29, EdgeType.DIRECTED)) \
    .add_edge(Edge(n, q35, EdgeType.DIRECTED)) \
    .add_edge(Edge(o, q26, EdgeType.DIRECTED)) \
    .add_edge(Edge(o, q31, EdgeType.DIRECTED)) \
    .add_edge(Edge(o, q34, EdgeType.DIRECTED)) \
    .add_edge(Edge(p, q25, EdgeType.DIRECTED)) \
    .add_edge(Edge(p, q30, EdgeType.DIRECTED)) \
    .add_edge(Edge(p, q33, EdgeType.DIRECTED)) \
    .add_edge(Edge(a2, a6, EdgeType.DIRECTED)) \
    .add_edge(Edge(a2, a5, EdgeType.DIRECTED)) \
    .add_edge(Edge(a5, qCCF1, EdgeType.DIRECTED)) \
    .add_edge(Edge(a5, qCCF2, EdgeType.DIRECTED)) \
    .add_edge(Edge(a5, qCCF3, EdgeType.DIRECTED)) \
    .add_edge(Edge(a6, qCCF4, EdgeType.DIRECTED)) \
    .add_edge(Edge(a6, qCCF5, EdgeType.DIRECTED)) \
    .add_edge(Edge(a6, qCCF6, EdgeType.DIRECTED)) \
    .add_edge(Edge(a7, qCCF7, EdgeType.DIRECTED)) \
    .add_edge(Edge(a7, qCCF8, EdgeType.DIRECTED)) \
    .add_edge(Edge(a7, qCCF9, EdgeType.DIRECTED)) \
    .add_edge(Edge(a8, qCCF10, EdgeType.DIRECTED)) \
    .add_edge(Edge(a8, qCCF11, EdgeType.DIRECTED)) \
    .add_edge(Edge(a8, qCCF12, EdgeType.DIRECTED)) 




# convert the BBN to a join tree
join_tree = InferenceController.apply(bbn)

# insert an observation evidence
ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name('Competence_a_limplication')) \
    .with_evidence('on', 1.0) \
    .build()
join_tree.set_observation(ev)

names = ['MPP_FFD','MPP_FMD','MTT_FFD','MTT_FMD','AC_FMA','DA_FMA','AC_FFA','DA_FFA','MPP_CCF','MTT_CCF','AC_CCF','DA_CCF','MPP_A','MTT_A','AC_A','DA_A']


knowledgeVcetor = []
for name in names:
    node = join_tree.get_bbn_node_by_name(name)
    potential = join_tree.get_bbn_potential(node)
    knowledgeVcetor.append(potential.entries[0].value)
print(knowledgeVcetor)

