# Calculate F1 scores for each LLM's predictions against human labels
# Requires text files:
# task_3_human_labels.txt
# task_3_gpt_labels.txt
# task_3_deep_labels.txt
# task_3_gemma3_labels.txt
# task_3_gemma4_labels.txt

from sklearn.metrics import f1_score

b1 = open('task_3_human_labels.txt', 'r')  
b2 = open('task_3_gpt_labels.txt', 'r')
b3 = open('task_3_deep_labels.txt', 'r')
b4 = open('task_3_gemma3_labels.txt', 'r')
b5 = open('task_3_gemma4_labels.txt', 'r')

tids = []
human_labels = []
gpt_labels = []
deep_labels = []
gemma3_labels = []
gemma4_labels = []

lines1 = b1.readlines() 
lines2 = b2.readlines()
lines3 = b3.readlines()
lines4 = b4.readlines()
lines5 = b5.readlines()

# extract human_label from each line in lines1
for line in lines1:
    parts = line.split('\t')
    tids.append(parts[0])
    if len(parts) < 2:
        human_labels.append('')
    else:
        human_labels.append(parts[1].strip())

# extract gpt_label from each line in lines2
for line in lines2:
    parts = line.split('\t')
    if len(parts) < 2:
        gpt_labels.append('')
    else:
        gpt_labels.append(parts[1].strip())

# extract deep_label from each line in lines3
for line in lines3:
    parts = line.split('\t')
    if len(parts) < 2:
        deep_labels.append('')
    else:
        deep_labels.append(parts[1].strip())

# extract gemma3_label from each line in lines4
for line in lines4:
    parts = line.split('\t')
    if len(parts) < 2:
        gemma3_labels.append('')
    else:
        gemma3_labels.append(parts[1].strip())

# extract gemma4_label from each line in lines5
for line in lines5:
    parts = line.split('\t')
    if len(parts) < 2:
        gemma4_labels.append('')
    else:
        gemma4_labels.append(parts[1].strip())

# convert human_labels, gpt_labels, deep_labels, gemma_labels to integers: 
# STRONG is converted to 2, WEAK is converted to 1, '' is converted to 0, other values are converted to -1

human_labels = [2 if x == 'STRONG' else (1 if x == 'WEAK' else (0 if x == '' else -1)) for x in human_labels]
gpt_labels = [2 if x == 'STRONG' else (1 if x == 'WEAK' else (0 if x == '' else -1)) for x in gpt_labels]
deep_labels = [2 if x == 'STRONG' else (1 if x == 'WEAK' else (0 if x == '' else -1)) for x in deep_labels]
gemma3_labels = [2 if x == 'STRONG' else (1 if x == 'WEAK' else (0 if x == '' else -1)) for x in gemma3_labels]
gemma4_labels = [2 if x == 'STRONG' else (1 if x == 'WEAK' else (0 if x == '' else -1)) for x in gemma4_labels]

# calculate micro-F1 score between gpt_labels as predicted and human_labels as true labels, and print the average F1 score
f1_gpt3 = f1_score(human_labels, gpt_labels, average='micro')
print('Average F1 score (GPT-OSS):\t', f1_gpt3)

f1_deep = f1_score(human_labels, deep_labels, average='micro')
print('Average F1 score (Deep):\t', f1_deep)

f1_gemma3 = f1_score(human_labels, gemma3_labels, average='micro')
print('Average F1 score (Gemma3):\t', f1_gemma3)

f1_gemma4 = f1_score(human_labels, gemma4_labels, average='micro')
print('Average F1 score (Gemma4):\t', f1_gemma4)
