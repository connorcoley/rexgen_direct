#python scripts/coverage.py
cand = open('core-wln-global/model-300-3-direct/test_withReagents.cbond')
gold = open('data/test.txt.proc')

ks = [6, 8, 10, 12, 14, 16, 18, 20]
topks = [0 for k in ks]
tot = 0
for line in cand:
    tot += 1
    cand_bonds = []
    for v in line.split():
        x,y,t = v.split('-')
        cand_bonds.append((int(x),int(y),float(t)))
    
    line = gold.readline()
    tmp = line.split()[1]
    gold_bonds = []
    for v in tmp.split(';'):
        x,y,t = v.split('-')
        x,y = int(x),int(y)
        x,y = min(x,y), max(x,y)
        gold_bonds.append((x, y ,float(t)))
    for i in range(len(ks)):
        if set(gold_bonds) <= set(cand_bonds[:ks[i]]):
            topks[i] += 1.0
print(ks)
print [topk/tot for topk in topks]
