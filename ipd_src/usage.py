from game import *
from ipd import *
from strategies import *

create a classic prisoner's dilemma
ipd_scores =[(3,3),(0,5),(5,0),(1,1)]   # Prisoner's dilemma
g = Game(ipd_scores,['C','D'])

g.prettyPrint()

print('------------- Nash & Pareto --------------')
print("Nash: ", g.getNash())
print("Pareto: ", g.getPareto())
print()

create iterated prisoner's dilemma with different strategies
tft=Tft()
collaborator=Periodic("C", "allC")
defector=Periodic("D", "allD")

m1 = Meeting(g,tft,defector,10)
m1.run()
m1.prettyPrint()

m2 = Meeting(g, tft, collaborator, 10)
m2.run()
m2.prettyPrint()

m3 = Meeting(g, defector, defector, 10)
m3.run()
m3.prettyPrint()

m4 = Meeting(g, collaborator, collaborator, 10)
m4.run()
m4.prettyPrint()

m5 = Meeting(g, collaborator, defector, 10)
m5.run()
m5.prettyPrint()

print()
print("Number of cooperations : " )
print (m.s1.name+"\t" + str(m.nb_cooperation_s1))
print (m.s2.name+"\t" + str(m.nb_cooperation_s2))
