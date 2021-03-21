from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer
from random import randint
from collections import Counter
import torch


# Sofiane: there are two ways to create an unmasker:
# Method 1: can be used in huggingFace starting from a given version I don't
# remember.
unmasker = pipeline('fill-mask', model='./models/215000')

# Method 2: you can create it manually in case you need specific tokenizer.
# model = AutoModelForMaskedLM.from_pretrained('./models/215000')
# tokenizer = AutoTokenizer.from_pretrained("./models/215000")
# unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)



# take a random trajectory traj:
traj = ("8a536bc8b697fff 8a536b524aeffff 8a536b524337fff " +\
"8a536b5243a7fff " +\
"8a536bc8942ffff 8a536bc8972ffff 8a536bc89367fff 8a536bcd4197fff "+\
"8a536bcd4a57fff 8a536bcd5d97fff").split(' ')

# Insert MASK at a preferred position, in my example it is position 5 (0-4)
pos = 4
traj.insert(pos, '[MASK]')
print('INPUT:', ' '.join(traj))

# Sofiane: Here I'll perform 15 successive predictions using beam search. the
# idea is that I'll start requesting a prediction for the cell in position 4,
# then iteratively append the the predicted value to the query and push MASK to
# i+1. If predicted hex == previous hex (the one before MASK), take second
# prediction. 

for i in range(15):
    res = unmasker(' '.join(traj[pos-4:]))
    pred = res[0]['token_str']
    if pred == traj[pos-1]:
        pred = res[1]['token_str']
    print('Query Traj:', ' '.join(traj))
    print ('Predicted hex at position:', i, '-->', pred)
    print('**********')
    traj.insert(pos,pred)
    pos += 1

"""
masked = []
sentences = []
top = []
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
"""

#pos = 4
#traj.insert(pos, '[MASK]')

#tt = ' '.join(traj)
#sentences = tt.strip()
#print('Sent:', sentences)
#inputs = tokenizer.encode_plus(sentences, add_special_tokens=True, return_tensors="pt")
#print("Sofiane Inputs", inputs)
#device = torch.device('cpu')
#prediction = model(inputs['input_ids'].to(device), 
#        token_type_ids=inputs['token_type_ids'].to(device))

        #token_type_ids=inputs['token_type_ids'].to(self.device))[0].argmax().item()
#print("Preds:", prediction[0].shape)
#print("Label:", tokenizer.convert_ids_to_tokens(prediction))
#sys.exit()


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
