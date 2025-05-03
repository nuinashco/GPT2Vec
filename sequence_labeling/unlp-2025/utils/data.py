def convert_to_seq_labeling(text, tokenizer, max_length=None, trigger_spans=None):
    tokenized_output = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        
        max_length=max_length,
        truncation=(max_length is not None),
        padding=False
    )
    tokens = tokenized_output["input_ids"]
    offsets = tokenized_output["offset_mapping"]

    # Get subword tokenized versions of the text
    token_strings = tokenizer.convert_ids_to_tokens(tokens)

    
    # Initialize labels as 'O'
    labels = [0] * len(tokens)

    if trigger_spans is not None:
        # Assign 'TRIGGER' to overlapping tokens
        for start, end in trigger_spans:
            for i, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == 0 and tok_end == 0:
                    continue
                if tok_start < end and tok_end > start:  # If token overlaps with the trigger span
                    labels[i] = 1

    tokenized_output['labels'] = labels
    return tokenized_output


def preprocess_df(df, tokenizer, max_length):
    """Modified processing incorporating trigger span handling"""
    
    df['seq_labels'] = df.progress_apply(
        lambda row: convert_to_seq_labeling(
            text=row['content'],
            tokenizer=tokenizer,
            trigger_spans=row.get('trigger_words', None),  # Handle both validation and test cases
            max_length=max_length
        ),
        axis=1
    )
    
    # Extract all tokenizer outputs
    for column in df.seq_labels.iloc[0].keys():
        df[column] = df.seq_labels.apply(lambda x: x.get(column))
    
    return df