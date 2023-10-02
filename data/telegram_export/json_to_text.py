import json
import pandas as pd
import re

def clean_chat_json(json_path):
    # Read the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        chat_data = json.load(f)

    # Extract each text from message and save it as a list
    messages = chat_data.get('messages', [])
    message_texts_cleaned = []

    for message in messages:
        if message.get('type') == 'message':
            text = message.get('text', '')
            
            if isinstance(text, str):
                if text.strip():  # Remove empty strings (which would turn into NaN in DataFrame)
                    if not re.match(r'http[s]?://', text):  # Remove links
                        message_texts_cleaned.append(text)
            elif isinstance(text, list):
                full_text = ''.join([item.get('text', '') if isinstance(item, dict) else item for item in text])
                if full_text.strip():  # Remove empty strings
                    if not re.match(r'http[s]?://', full_text):  # Remove links
                        message_texts_cleaned.append(full_text)

    # Create a DataFrame with the revised text extraction
    df_cleaned = pd.DataFrame(columns=['user', 'bot'])
    user_messages_cleaned = message_texts_cleaned[::2]
    bot_messages_cleaned = message_texts_cleaned[1::2]

    # Filling in the DataFrame
    df_cleaned['user'] = user_messages_cleaned[:len(bot_messages_cleaned)]
    df_cleaned['bot'] = bot_messages_cleaned

    return df_cleaned

# Example usage:
json_path = "result.json"
df = clean_chat_json(json_path)
# df.to_csv("cleaned_chat_dataset.csv", index=False)

def save_to_text_file(df, text_file_path):
    with open(text_file_path, 'w', encoding='utf-8') as f:
        for index, row in df.iterrows():
            # f.write(f"<user> {row['user']}\n")
            # f.write(f"<bot> {row['bot']}\n")
            f.write(f"{row['user']}\n")
            f.write(f"{row['bot']}\n")

# Example usage to save the DataFrame to a text file:
text_file_path = "input.txt"
save_to_text_file(df, text_file_path)
