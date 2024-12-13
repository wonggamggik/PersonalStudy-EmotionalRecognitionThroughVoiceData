import json
import os

def restructure_multiple_conversations(input_directory, output_directory):
    """
    Restructures multiple JSON files in a directory and saves them with updated filenames.

    Args:
        input_directory (str): Path to the directory containing input JSON files.
        output_directory (str): Path to the directory to save output JSON files.
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Loop through files in the input directory
    for i in range(1, 11):  # Assuming files are named 1.json to 10.json
        input_file = os.path.join(input_directory, f"{i}.json")
        output_file = os.path.join(output_directory, f"output{i}.json")

        # Check if the input file exists
        if not os.path.exists(input_file):
            print(f"File {input_file} does not exist. Skipping.")
            continue

        # Load the JSON file
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Restructure the "Conversation" field
        restructured_conversations = []
        for conversation in data.get("Conversation", []):
            restructured_conversation = {
                "Text": conversation["Text"],
                "SpeakerNo": conversation["SpeakerNo"],
                "StartTime": conversation["StartTime"],
                "EndTime": conversation["EndTime"],
                "SpeakerEmotionCategory": conversation["SpeakerEmotionCategory"],
                "SpeakerEmotionTarget": conversation["SpeakerEmotionTarget"],
                "SpeakerEmotionLevel": conversation["SpeakerEmotionLevel"]
            }
            restructured_conversations.append(restructured_conversation)

        # Update the data with restructured conversations
        data["Conversation"] = restructured_conversations

        # Save the updated data to a new JSON file
        with open(output_file, 'w', encoding='utf-8') as output_file_obj:
            json.dump(data, output_file_obj, ensure_ascii=False, indent=4)

        print(f"Restructured data from {input_file} has been saved to {output_file}")

# Example usage
# Replace 'input_directory' and 'output_directory' with the actual paths
input_directory = "./script_preprocessing_input_files"
output_directory = "./script_preprocessing_output_files"
restructure_multiple_conversations(input_directory, output_directory)
