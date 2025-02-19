import os
from openai import OpenAI

# Set OpenAI API key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def get_files_from_downloads():
    downloads_folder = os.path.expanduser('~/Downloads')
    files = [f for f in os.listdir(downloads_folder) if os.path.isfile(os.path.join(downloads_folder, f))]
    return files

def analyze_files_with_gpt(files, gpt_version="gpt-3.5-turbo"):
    prompt = f"Group the following files into categories. Be very concise; return only the categories and sorted filenames:\n{files}"
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=gpt_version,
        max_tokens=250,
        stream=True,
    )
    result = ""
    for chunk in response:
        result += chunk.choices[0].delta.content or ""
    return result


if __name__ == "__main__":
    files = get_files_from_downloads()
    analysis = analyze_files_with_gpt(files, gpt_version="gpt-4o-mini")
    print("Analysis and grouping of files:")
    print(analysis)
