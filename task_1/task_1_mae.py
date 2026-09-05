# Requires task_1_semeval_gold.txt, 
# task_1_gpt_zeroshot.txt, task_1_gpt_fewshot.txt, task_1_gpt_cot123.txt, 
# task_1_olmo_cot123.txt, task_1_gemma_cot123.txt, task_1_deepseek_cot123.txt

import imblearn

# tids = []
human_scores = []
gold_scores_f = open('task_1_semeval_gold.txt', 'r')
gold_scores_lines = gold_scores_f.readlines()
for n in range(len(gold_scores_lines)):
    parts = gold_scores_lines[n].strip().split('\t')
    human_scores.append(int(parts[1]))

zeroshot_scores = []
zershot_scores_f = open('task_1_gpt_zeroshot.txt', 'r')
zeroshot_scores_lines = zershot_scores_f.readlines()
# extract the second column (zeroshot_score) from each line in zeroshot_scores_lines and append to zeroshot_scores list
for n in range(len(zeroshot_scores_lines)):
    parts = zeroshot_scores_lines[n].strip().split('\t')
    zeroshot_scores.append(int(parts[1]))
# calculate macro-averaged MAE score between gpt_zeroshot_scores as predicted and human_scores as true labels, and print the average F1 score
f1_zeroshot = imblearn.metrics.macro_averaged_mean_absolute_error(human_scores, zeroshot_scores)
print('MAE score gpt-oss Zeroshot:\t', f1_zeroshot)

fewshot_scores = []
fewshot_scores_f = open('task_1_gpt_fewshot.txt', 'r')
fewshot_scores_lines = fewshot_scores_f.readlines()
# extract the second column (fewshot_score) from each line in fewshot_scores_lines and append to fewshot_scores list
for n in range(len(fewshot_scores_lines)):
    parts = fewshot_scores_lines[n].strip().split('\t')
    fewshot_scores.append(int(parts[1]))
# calculate macro-averaged MAE score between gpt_fewshot_scores as predicted and human_scores as true labels, and print the average F1 score
f1_fewshot = imblearn.metrics.macro_averaged_mean_absolute_error(human_scores, fewshot_scores)
print('MAE score gpt-oss Fewshot:\t', f1_fewshot)

gpt_scores_f = open('task_1_gpt_cot123.txt', 'r')
gpt_scores_lines = gpt_scores_f.readlines()
# extract the columns 3, 4, 5 from each line in gpt_scores_lines and append to gpt_cot1, gpt_cot2, gpt_cot3 lists
gpt_cot1 = []
gpt_cot2 = []
gpt_cot3 = []
for n in range(len(gpt_scores_lines)):
    parts = gpt_scores_lines[n].strip().split('\t')
    gpt_cot1.append(int(parts[2]))
    gpt_cot2.append(int(parts[3]))
    gpt_cot3.append(int(parts[4]))
# calculate macro-averaged MAE score between deep_p1_scores as predicted and human_scores as true labels, and print the average F1 score
f1_gpt_cot1 = imblearn.metrics.macro_averaged_mean_absolute_error(human_scores, gpt_cot1)
print('MAE score gpt-oss GPT COT1:\t', f1_gpt_cot1)
f1_gpt_cot2 = imblearn.metrics.macro_averaged_mean_absolute_error(human_scores, gpt_cot2)
print('MAE score gpt-oss GPT COT2:\t', f1_gpt_cot2)
f1_gpt_cot3 = imblearn.metrics.macro_averaged_mean_absolute_error(human_scores, gpt_cot3)
print('MAE score gpt-oss GPT COT3:\t', f1_gpt_cot3)

olmo_scores_f = open('task_1_olmo_cot123.txt', 'r')
olmo_scores_lines = olmo_scores_f.readlines()
# extract the columns 3, 4, 5 from each line in olmo_scores_lines and append to olmo_cot1, olmo_cot2, olmo_cot3 lists
olmo_cot1 = []
olmo_cot2 = []
olmo_cot3 = []
for n in range(len(olmo_scores_lines)):
    parts = olmo_scores_lines[n].strip().split('\t')
    olmo_cot1.append(int(parts[2]))
    olmo_cot2.append(int(parts[3]))
    olmo_cot3.append(int(parts[4]))
# calculate macro-averaged MAE score between deep_p1_scores as predicted and human_scores as true labels, and print the average F1 score
f1_olmo_cot1 = imblearn.metrics.macro_averaged_mean_absolute_error(human_scores, olmo_cot1)
print('MAE score olmo COT1:\t', f1_olmo_cot1)
f1_olmo_cot2 = imblearn.metrics.macro_averaged_mean_absolute_error(human_scores, olmo_cot2)
print('MAE score olmo COT2:\t', f1_olmo_cot2)
f1_olmo_cot3 = imblearn.metrics.macro_averaged_mean_absolute_error(human_scores, olmo_cot3)
print('MAE score olmo COT3:\t', f1_olmo_cot3)

gemma_scores_f = open('task_1_gemma_cot123.txt', 'r')
gemma_scores_lines = gemma_scores_f.readlines()
# extract the columns 3, 4, 5 from each line in gemma_scores_lines and append to gemma_cot1, gemma_cot2, gemma_cot3 lists
gemma_cot1 = []
gemma_cot2 = []
gemma_cot3 = []
for n in range(len(gemma_scores_lines)):
    parts = gemma_scores_lines[n].strip().split('\t')
    gemma_cot1.append(int(parts[2]))
    gemma_cot2.append(int(parts[3]))
    gemma_cot3.append(int(parts[4]))
# calculate macro-averaged MAE score between gemma_p1_scores as predicted and human_scores as true labels, and print the average F1 score
f1_gemma_cot1 = imblearn.metrics.macro_averaged_mean_absolute_error(human_scores, gemma_cot1)
print('MAE score gemma COT1:\t', f1_gemma_cot1)
f1_gemma_cot2 = imblearn.metrics.macro_averaged_mean_absolute_error(human_scores, gemma_cot2)
print('MAE score gemma COT2:\t', f1_gemma_cot2)
f1_gemma_cot3 = imblearn.metrics.macro_averaged_mean_absolute_error(human_scores, gemma_cot3)
print('MAE score gemma COT3:\t', f1_gemma_cot3)

deep_scores_f = open('task_1_deepseek_cot123.txt', 'r')
deep_scores_lines = deep_scores_f.readlines()
# extract the columns 3, 4, 5 from each line in deep_scores_lines and append to deep_cot1, deep_cot2, deep_cot3 lists
deep_cot1 = []
deep_cot2 = []
deep_cot3 = []
for n in range(len(deep_scores_lines)):
    parts = deep_scores_lines[n].strip().split('\t')
    deep_cot1.append(int(parts[2]))
    deep_cot2.append(int(parts[3]))
    deep_cot3.append(int(parts[4]))
# calculate macro-averaged MAE score between deep_p1_scores as predicted and human_scores as true labels, and print the average F1 score
f1_deep_cot1 = imblearn.metrics.macro_averaged_mean_absolute_error(human_scores, deep_cot1)
print('MAE score deepseek P1 COT1:\t', f1_deep_cot1)
f1_deep_cot2 = imblearn.metrics.macro_averaged_mean_absolute_error(human_scores, deep_cot2)
print('MAE score deepseek P1 COT2:\t', f1_deep_cot2)
f1_deep_cot3 = imblearn.metrics.macro_averaged_mean_absolute_error(human_scores, deep_cot3)
print('MAE score deepseek P1 COT3:\t', f1_deep_cot3)

