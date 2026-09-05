# calculate BLEU scores of LLM rewrites against human rewrites, and print the average BLEU score for each LLM
# Requires task_2_human_headers.txt, task_2_gpt_headers.txt, task_2_deepseek_headers.txt, task_2_gemma3_headers.txt

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

human_f = open('task_2_human_headers.txt', 'r', encoding='utf-8', errors='ignore')  
gpt_f = open('task_2_gpt_headers.txt', 'r', encoding='utf-8', errors='ignore')  
deepseek_f = open('task_2_deepseek_headers.txt', 'r', encoding='utf-8', errors='ignore')  
gemma3_f = open('task_2_gemma3_headers.txt', 'r', encoding='utf-8', errors='ignore')  

human_lines = human_f.readlines()
# extract second column (human_rewrite) from each line in human_lines and append to humans list
humans = []
for n in range(len(human_lines)):
    # take the second column if present; otherwise, append the null string
    if len(human_lines[n].strip().split('\t')) == 2:
        parts = human_lines[n].strip().split('\t')
        humans.append(parts[1])
    else:
        humans.append("")

gpt_lines = gpt_f.readlines() 
# extract second column (gpt_rewrite) from each line in gpt_lines and append to gpts list
gpts = []
for n in range(len(gpt_lines)):
    # take the second column if present; otherwise, append the null string
    if len(gpt_lines[n].strip().split('\t')) == 2:
        parts = gpt_lines[n].strip().split('\t')
        gpts.append(parts[1])
    else:
        gpts.append("")

deepseek_lines = deepseek_f.readlines() 
# extract second column (deepseek_rewrite) from each line in deepseek_lines and append to deepseeks list
deepseeks = []
for n in range(len(deepseek_lines)):
    # take the second column if present; otherwise, append the null string
    if len(deepseek_lines[n].strip().split('\t')) == 2:
        parts = deepseek_lines[n].strip().split('\t')
        deepseeks.append(parts[1])
    else:
        deepseeks.append("")

gemma3_lines = gemma3_f.readlines() 
# extract second column (gemma3_rewrite) from each line in gemma3_lines and append to gemma3s list
gemma3s = []
for n in range(len(gemma3_lines)):
    # take the second column if present; otherwise, append the null string
    if len(gemma3_lines[n].strip().split('\t')) == 2:
        parts = gemma3_lines[n].strip().split('\t')
        gemma3s.append(parts[1])
    else:
        gemma3s.append("")  

# calculate BLEU scores
smoothie = SmoothingFunction().method4
bleu_scores_col1 = []
bleu_scores_col2 = []
bleu_scores_col3 = []
bleu_scores_col4 = []
col1 = humans
col2 = gpts
col3 = deepseeks
col4 = gemma3s

for i in range(len(col1)):
    reference = [col1[i].split()]
    candidate2 = col2[i].split()
    candidate3 = col3[i].split()
    candidate4 = col4[i].split()
    bleu_scores_col2.append(sentence_bleu(reference, candidate2, smoothing_function=smoothie))
    bleu_scores_col3.append(sentence_bleu(reference, candidate3, smoothing_function=smoothie))
    bleu_scores_col4.append(sentence_bleu(reference, candidate4, smoothing_function=smoothie))
print("Average BLEU score for GPT: ", sum(bleu_scores_col2) / len(bleu_scores_col2))
print("Average BLEU score for deep: ", sum(bleu_scores_col3) / len(bleu_scores_col3))
print("Average BLEU score for gemma3: ", sum(bleu_scores_col4) / len(bleu_scores_col4))



