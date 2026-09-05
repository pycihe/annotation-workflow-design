# Extract SemEval 2016/2017 gold data for tweet and reply processing
# Requires task_1_semeval_gold_80_tweets_id.txt (tweet IDs of 80 tweets)
# Requires SemEval-2016 Task 4: Sentiment Analysis in Twitter datasets from:
# http://alt.qcri.org/semeval2017/task4/data/uploads/download.zip
# twitter-2016train-CE.txt, twitter-2016test-CE.txt, twitter-2016devtest-CE.txt, twitter-2016dev-CE.txt

# Output task_1_semeval_gold.txt with 80 tweets and their gold sentiment scores in the format: tid, sentiment_score

# open text file 80_tweets_tid.txt and read first column of all lines to list
with open("task_1_semeval_gold_tweets_id.txt", "r", encoding="utf-8", errors="ignore") as f:
    lines = [line.split()[0] for line in f.readlines()]

# open text file twitter-2016train-CE.txt and read all lines to list
with open("twitter-2016train-CE.txt", "r", encoding="utf-8", errors="ignore") as f:
    lines2 = f.readlines()
# open text file twitter-2016test-CE.txt and append all lines to lines2 list
with open("twitter-2016test-CE.txt", "r", encoding="utf-8", errors="ignore") as f:
    lines2 += f.readlines()
# open text file twitter-2016devtest-CE.txt and append all lines to lines2 list
with open("twitter-2016devtest-CE.txt", "r", encoding="utf-8", errors="ignore") as f:
    lines2 += f.readlines()
# open text file twitter-2016dev-CE.txt and append all lines to lines2 list
with open("twitter-2016dev-CE.txt", "r", encoding="utf-8", errors="ignore") as f:
    lines2 += f.readlines()

# for each line in lines, find the matching line where the first column matches, 
# and add the matching line to a new list called filtered_lines. If no matching line is found, skip that line.
filtered_lines = []
for line in lines:
    for line2 in lines2:
        if line == line2.split()[0]:
            filtered_lines.append(line2)
            break

# separate each line in filtered_lines into columns by tabs
# process filtered_lines to remove column 2 and keep only columns 1, 3
filtered_lines = [line.split('\t')[0] + '\t' + line.split('\t')[2] for line in filtered_lines]

# write filtered_lines to new text file SemEval2017-task4-subtask-CE_filtered.txt
with open("task_1_semeval_gold.txt", "w", encoding="utf-8", errors="ignore") as f:
    f.writelines(filtered_lines)
