from transformers import pipeline, AutoModelForPreTraining, AutoTokenizer
from random import randint
from collections import Counter

unmasker = pipeline('fill-mask', model='./models/215000')
#unmasker = pipeline('fill-mask', model='./models/92500')

#model = AutoModelForMaskedLM.from_pretrained('./models/92500')

#model = AutoModelForPreTraining.from_pretrained('./models/92500')
#tokenizer = AutoTokenizer.from_pretrained("./models/92500")
#unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)

masked = []
sentences = []
top = []
"""
with open('data/processed/trips_pred_h3_10.txt') as f:
    for j, line in enumerate(f):
        row = ""
        tr = line.strip().split(',')[-1].split(' ')
        print(j, '*************************')
        row += ' '.join(tr)
        i = randint(0, len(tr)-1)
        masked.append(tr[i])
        tr[i] = '[MASK]'
        #print(j, len(tr), i)
        #print(' '.join(tr))
        #print(masked[-1])
        res = unmasker(' '.join(tr))
        found = 0
        rec = []
        for k, r in enumerate(res):
            rec.append(r['token_str'])
            if masked[-1] == r['token_str']:
                found = k + 1
        #print('Rec:\t', ' '.join(rec))
        row += ',' + ' '.join(rec)
        row += ',' + masked[-1]
        print('x= "%s"' % row)
        top.append(found)
        if j == 20:
            break
print(top)
c = Counter(top)
print(c)
print(c.most_common(10))
"""

gt = '8a536bc8a637fff'
traj = ("8a536bc8b697fff 8a536bc8b697fff 8a536b524aeffff 8a536b524337fff " +\
"8a536b5243a7fff 8a536bc8a637fff 8a536bc8a717fff 8a536bc8a0cffff " +\
"8a536bc8942ffff 8a536bc8972ffff 8a537bc89367fff 8a536bcd4197fff "+\
"8a536bcd4a57fff 8a536bcd5d97fff").split(' ')
traj = ("8a536bc8b697fff 8a536b524aeffff 8a536b524337fff " +\
"8a536b5243a7fff " +\
"8a536bc8942ffff 8a536bc8972ffff 8a536bc89367fff 8a536bcd4197fff "+\
"8a536bcd4a57fff 8a536bcd5d97fff").split(' ')

pos = 4
traj.insert(pos, '[MASK]')
print('INPUT:', ' '.join(traj))
for i in range(15):
    res = unmasker(' '.join(traj[pos-4:]))
    pred = res[0]['token_str']
    if pred == traj[pos-1]:
        pred = res[1]['token_str']
    print(' '.join(traj))
    print (i, pred)
    print('**********')
    traj.insert(pos,pred)
    pos += 1
"""
replace = 4
for i in range(8):
    gt = traj[replace]
    traj[replace] = '[MASK]'
    res = unmasker(' '.join(traj))
    pred = res[0]['token_str']
    print(' '.join(traj))
    print (i, pred)
    print('**********')
    traj.insert(replace,pred)
    replace += 1



for i, r in enumerate(res):
    if i == 0:
    
    
    if gt == r['token_str']:
        print("[*] -> %s\t%s" % (r['token_str'], r['score']))
    else:
        print("%s\t%s" % (r['token_str'], r['score']))

print("**************")
"""
