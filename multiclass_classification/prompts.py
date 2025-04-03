def generate_zero_shot_prompt(row):
    return f"""
    You are a Fake News detection AI. Based on the provided details, classify the statement into one of these categories: 
    true, mostly-true, half-true, false, barely-true, pants-on-fire.

    Statement: "{row['statement']}"  
    Speaker: {row['speaker']} ({row['speaker_job_title']})  
    Context: "{row['context']}"  
    Past Truthfulness: {row['barely_true_counts']} barely true, {row['false_counts']} false, {row['half_true_counts']} half-true, {row['mostly_true_counts']} mostly true, {row['pants_on_fire_counts']} pants-on-fire.  

    Choose only one category from: true, mostly-true, half-true, false, barely-true, pants-on-fire.  
    Answer with only the category name and nothing else.
    """
def generate_few_shot_prompt(row, formatted_examples):
    return f"""
    You are a Fake News detection AI. Based on the provided details, classify the statement into one of these categories: 
    true, mostly-true, half-true, false, barely-true, pants-on-fire.

    Learn from these training examples:
    {formatted_examples}  

    Now classify this new statement:
    Statement: "{row['statement']}"  
    Speaker: {row['speaker']} ({row['speaker_job_title']})  
    Context: "{row['context']}"  
    Past Truthfulness: {row['barely_true_counts']} barely true, {row['false_counts']} false, {row['half_true_counts']} half-true, {row['mostly_true_counts']} mostly true, {row['pants_on_fire_counts']} pants-on-fire.  

    Choose only one category from: true, mostly-true, half-true, false, barely-true, pants-on-fire.  
    Answer with only the category name and nothing else.
    """
