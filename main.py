"""
This program is build with Qwen/Qwen2.5-3B-Instruct and gpt-4o-mini to be able to parse the resumes and generate rankin respectively.

> It accepts two parameters provided as a command line input.
> The first parameter for directory where resumes and JD exist, second for open ai api_key
> Job Description file must be in docx format and starts with key-word JD
> The output will be an HTML report displaying Rank, Employee Name, Score, Justification

Syntax: python main.py <string> <string> <string>

The following example is given for your reference:

Terminal Input: python main.py "/content/sample_data" "apikey"



"""

!pip install pypdf
!pip install python-docx

# Import **Libraries**"""

import json
import re
import sys
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from pypdf import PdfReader
from docx import Document
from openai import OpenAI
from google.colab import userdata
import pandas as pd
from IPython.display import HTML, display

##### You may comment this section to see verbose -- but you must un-comment this before final submission. ######
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
#################################################################################################################

"""# ***Function To Scan Through all the resume present in directory***"""

def _get_files(directory_path):
    """
    Scans the given directory for resume files based on common extensions.
    """
    target_dir = Path(directory_path)

    # Check if directory exists
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: The directory '{directory_path}' does not exist.")
        return []

    # Define common resume file extensions
    valid_extensions = {'.pdf', '.docx'}

    resume_files = []
    jd_file = []
    # Iterate through files in the directory
    for file_path in target_dir.iterdir():
        #print(file_path)
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
          if file_path.name[:2].lower() == 'jd':
            jd_file.append(file_path)
          else:
            resume_files.append(file_path)
            #print(file_path.suffix)

    return resume_files, jd_file

"""## ***Extract Raw Text for pdf resumes***"""

def _get_pdf_rawText(resume_path):
  reader = PdfReader(resume_path)
  number_of_pages = len(reader.pages)
  raw_text  = ""
  combined_payload = ""
  #print('Total No of Pages: ',number_of_pages)
  for pg in range(number_of_pages):
    #print(pg)
    page = reader.pages[pg]
    raw_text = raw_text+page.extract_text()
  #combined_payload += f"\n<resume path='{resume_path}'>\n{raw_text}\n</resume>\n"
  #print(raw_text)
  return raw_text

"""## ***Extract Raw Text for Docx Resumes***"""

def _get_doc_rawText(resume_path):
  doc = Document(resume_path)
  raw_text = ""
  combined_payload = ""
  for para in doc.paragraphs:
    raw_text = raw_text + para.text
  #print(raw_text)
  #combined_payload += f"\n<resume path='{resume_path}'>\n{raw_text}\n</resume>\n"
  return raw_text

"""# ***Clean Text to minimize tokenization***

***Remove White Space***
"""

def _remove_white_space(resume_raw_text):
  # Remove excessive whitespace
  cleaned_text = re.sub(r'[ \t]+', ' ', resume_raw_text)
  # Remove excessive newline characters
  cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
  lines = cleaned_text.split('\n')
  cleaned_lines = []
  #print(len(resume_raw_text)-len(cleaned_text))
  return cleaned_text

"""# ***Feed Resume Cleaned Text into Qwen LLM to get JSON***"""

def _get_json_from_llm(cleaned_text,category,model,tokenizer):


    if category == 'Resume':

      schema_template = """{

        "name": "string",
        "email": "string",
        "phone": "string",
        "skills": ["string"],
        "education": [{"degree":"string",
                      "Marks": "string"}],
        "experience": [{"company":"string",
                      "designation": "string",
                      "start_date": "string",
                      "end_date": "string"
                      }]

      }"""

      messages = [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a precise data extraction assistant. Output ONLY a valid"
                                        " JSON object matching the requested schema. Do not include"
                                        " markdown code blocks, conversational filler, or explanations."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        f"Extract the resume into a JSON object adhering strictly to this"
                                        f" schema:\n{schema_template}\n\nResume text:\n{cleaned_text}"
                                    ),
                                },
                            ]
      text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

      """
      #Old Prompt
      prompt = fParse cleaned_text and generate valid JSON object stricly following schema: {schema_template}\n\n
                  use resume as only context strictly no other source or history\n\n
                  generate only json object no explanation or other text\n\n

      resume: {cleaned_text}
      """
    else:
      #print("Prompt for JD")
      schema_template = """{
                            "Summary": "string",
                            "Skills": "[string]",
                            "Experience": "[string]"
                          }"""

      messages = [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a precise data extraction assistant. Output ONLY a valid"
                                        " JSON object matching the requested schema. Do not include"
                                        " markdown code blocks, conversational filler, or explanations."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        f"Extract the jd into a JSON object adhering strictly to this"
                                        f" schema:\n{schema_template}\n\njd text:\n{cleaned_text}"
                                    ),
                                },
                            ]
      text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



      """
      #Old Prompt
      text_prompt = fParse cleaned_text and generate valid JSON object strictly following schema: {schema_template}\n\n
                   use jd as only context strictly no other source or history\n\n
                   generate only json object no explanation or other text\n\n

                jd: {cleaned_text}
                """
    tokenized_prompt = tokenizer(text_prompt, return_tensors="pt").input_ids.to(model.device)
    input_len = tokenized_prompt[0].shape
    #input_len = tokenized_prompt.input_ids.shape[1]
    #print(input_len[0])
    outputs = model.generate(tokenized_prompt, do_sample=False,max_new_tokens=1000,repetition_penalty=1.2)
    #print(outputs
    generated_tokens = outputs[:, input_len[0]:]
    #print(outputs)
    if category == 'Resume':
      resume = tokenizer.decode(generated_tokens, skip_special_tokens=True)[0]
       #print(resume)
      return resume
    else:
      jd = tokenizer.decode(generated_tokens, skip_special_tokens=True)[0]
      #print(jd)
      return jd

"""# ***Feed Resume Json object and Job Description into OpenAI model to get the Ranking***"""

def rank_resumes_with_openai(resume_json_array, job_criteria,api_key):
  # Initialize the OpenAI client
  api_key = api_key #userdata.get('OPENAI')
  client = OpenAI(api_key=api_key)
  #print(client)
  resumes_payload = json.dumps(resume_json_array, indent=2)

  rank_schema = """[{"name": "string",
                  "rank": "string",
                  "score": "string",
                  "justification": "string"
                  }]"""
  rule = f"""(
              1.justification must be in 50 charachters maximum
              2.Rank starts from 1, highest scorer must have rank 1

              )"""

  system_prompt = f"""(
          You are an expert technical recruiter\n\n
          Scan through JSON array containing multiple candidate resumes\n\n
          Your task is to evaluate each candidate and rank them out of scale of 10 for provided job description\n\n
          follow the rules strictly: {rule}\n\n
          return valid json object only strictly following schema: {rank_schema}\n\n
          No text or explanation needed

      )"""

  user_prompt = (
      f"Job Evaluation Criteria:\n{job_criteria}\n\n"
      f"Candidate Resumes Data:\n{resumes_payload}\n\n"
      "Please analyze, score, and rank these candidates strictly as a JSON object."
  )

  try:
      # 5. Call gpt-4o-mini using JSON mode
      response = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt}
          ],
          response_format={"type": "json_object"}
      )

      # 6. Parse and return the resulting dictionary
      ranking_content = response.choices[0].message.content
      return json.loads(ranking_content)

  except Exception as e:
      print(f"Error during OpenAI ranking call: {e}")
      return None

"""# ***Generate Report Output File From JSON***"""

def generate_ranking_report(
    ranking_json_data,
    job_details_data=None,
    output_filename="candidate_ranking_report.html",
):
  """Takes the JSON ranking object and optional job details, sorts candidates

  by rank lowest to highest, and outputs a professional HTML report.
  """
  # 1. Handle input format for rankings
  if isinstance(ranking_json_data, str):
    data = json.loads(ranking_json_data)
  else:
    data = ranking_json_data

  candidates_list = (
      data.get("rankings", data.get("candidates", data))
      if isinstance(data, dict)
      else data
  )

  # 2. Handle job details input format
  #job_info = {}
  if job_details_data:
    if isinstance(job_details_data, str):
      job_info = json.loads(job_details_data)
    else:
      job_info = job_details_data
  print(job_info)
  # 3. Load candidates into Pandas DataFrame and sort
  df = pd.DataFrame(candidates_list)
  if "rank" in df.columns:
    df["rank"] = pd.to_numeric(df["rank"])
    df = df.sort_values(by="rank", ascending=True)

  # 4. Design professional CSS styling and HTML layout
  html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Candidate Ranking Report</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: #f8f9fa;
                color: #333333;
                margin: 0;
                padding: 40px;
            }}
            .container {{
                max-width: 1000px;
                margin: auto;
                background: #ffffff;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            }}
            h2 {{
                color: #1a73e8;
                margin-top: 0;
                font-size: 24px;
                border-bottom: 2px solid #f1f3f4;
                padding-bottom: 15px;
            }}
            .job-card {{
                background-color: #f1f3f4;
                border-left: 4px solid #1a73e8;
                padding: 15px 20px;
                border-radius: 6px;
                margin-bottom: 25px;
            }}
            .job-card h3 {{
                margin: 0 0 10px 0;
                color: #202124;
                font-size: 18px;
            }}
            .job-card p {{
                margin: 5px 0;
                font-size: 14px;
                color: #444444;
            }}
            .meta-info {{
                font-size: 14px;
                color: #6c757d;
                margin-bottom: 25px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
                text-align: left;
            }}
            th {{
                background-color: #f1f3f4;
                color: #202124;
                font-weight: 600;
                padding: 12px 16px;
                border-bottom: 2px solid #dee2e6;
            }}
            td {{
                padding: 14px 16px;
                border-bottom: 1px solid #e9ecef;
                font-size: 14px;
                vertical-align: top;
            }}
            tr:hover {{
                background-color: #f8f9fa;
            }}
            .rank-badge {{
                display: inline-block;
                background-color: #e8f0fe;
                color: #1a73e8;
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 20px;
                text-align: center;
            }}
            .score-pill {{
                font-weight: bold;
                color: #137333;
                background-color: #ceead6;
                padding: 4px 8px;
                border-radius: 6px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>AI Recruiter: Candidate Evaluation & Ranking Report</h2>
    """

  # Conditionally render the Job Details card if provided
  if job_info:
    job_title = job_info.get("Summary", "N/A")
    experience = job_info.get("Experience", [])
    skills = job_info.get("Skills", [])

    if isinstance(experience, list):
      exp_html = "".join([f"<li>{exp}</li>" for exp in experience])
    else:
      exp_html = f"<li>{experience}</li>"

    if isinstance(skills, list):
      skills_html = "".join([f"<li>{skill}</li>" for skill in skills])
    else:
      skills_html = f"<li>{skills}</li>"

    html_content += f"""
            <div class="job-card">
                <div class="job-header">📋 <strong>Job Summary & Requirements</strong></div>
                <div class="job-section">
                    <p class="summary-text">{job_title}</p>
                </div>
                <div class="job-grid">
                    <div class="job-column">
                        <strong>Required Skills:</strong>
                        <ul class="job-list">{skills_html}</ul>
                    </div>
                    <div class="job-column">
                        <strong>Experience & Qualifications:</strong>
                        <ul class="job-list">{exp_html}</ul>
                    </div>
                </div>
            </div>

    """

  html_content += f"""
            <div class="meta-info">
                <strong>Sorting Order:</strong> Rank (Lowest to Highest) | <strong>Total Candidates Evaluated:</strong> {len(df)}
            </div>

            <table>
                <thead>
                    <tr>
                        <th style="width: 10%;">Rank</th>
                        <th style="width: 25%;">Candidate Name</th>
                        <th style="width: 15%;">Match Score</th>
                        <th style="width: 50%;">Justification</th>
                    </tr>
                </thead>
                <tbody>
    """

  # Populate table rows dynamically from DataFrame sorting
  #for _, row in df.iterrows():
  candidates_list = next((value for value in data.values() if isinstance(value, list)), [])
  for candidate in candidates_list:
        #rank = candidate['rank']
        #name = candidate['name']
        #score = candidate['score']
        #justification = candidate['justification']
        rank = candidate.get("rank")
        name = candidate.get("name")
        score = candidate.get("score")
        justification = candidate.get("justification")

        html_content += f"""
                          <tr>
                              <td><span class="rank-badge">#{rank}</span></td>
                              <td><strong>{name}</strong></td>
                              <td><span class="score-pill">{score} / 10</span></td>
                              <td>{justification}</td>
                          </tr>
    """

  html_content += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

  # 5. Save the output file
  with open(output_filename, "w", encoding="utf-8") as f:
    f.write(html_content)

  print(f"Report successfully generated and saved to: {output_filename}")

  # Render directly inside Google Colab output cell
  display(HTML(html_content))


"""# ***Main Call***"""

if __name__ == "__main__":
    
    #sys.argv = ['main.py', '', '']
    #sys.argv[1] = "/content/sample_data"
    #sys.argv[2] = "key"

    directory_to_scan = sys.argv[1].strip()
    key = sys.argv[2].strip()

    print("Current Directory to scan for files: ",directory_to_scan)

    resume_raw_text = ""
    clear_text =""
    found_resumes,found_jd = _get_files(directory_to_scan)
    resume_json_pdf = ""
    resume_json_doc = ""
    final_json_array = []
    print(f"Found {len(found_jd)} Job Detail:")
    print(f"Found {len(found_resumes)} resume(s):")

    if len(found_jd) > 0 and len(found_resumes) > 0:
      print("Start ranking process...")
      
      ####Initialize Model#########
      
      model_name = "Qwen/Qwen2.5-3B-Instruct"
      model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map="auto")
      tokenizer = AutoTokenizer.from_pretrained(model_name)

      ##############################

      for resume in found_resumes:
        resume_raw_text = ""
        clear_text = ""
        resume_json_pdf = ""
        resume_json_doc = ""
        print(f"- {resume.name} (Full path: {resume.absolute()})")
        #print("File Extension: ",resume.suffix)
        if resume.suffix == '.pdf':
          resume_raw_text = _get_pdf_rawText(resume)
          clear_text = _remove_white_space(resume_raw_text)
          #print('Extracting pdf raw text..')
          resume_json_pdf= _get_json_from_llm(clear_text,'Resume',model,tokenizer)
          final_json_array.append(json.loads(resume_json_pdf.replace("`","").replace("json","").strip()))
          #print(final_json)
        elif resume.suffix.lower() == '.docx':
          resume_raw_text = _get_doc_rawText(resume)
          clear_text = _remove_white_space(resume_raw_text)
          resume_json_doc= _get_json_from_llm(clear_text,'Resume',model,tokenizer)
          final_json_array.append(json.loads(resume_json_doc.replace("`","").replace("json","").strip()))
          #print('Extracting doc or docx raw text..')

    #Parse JD to get JSON
    print(f"- {found_jd[0].name} (Full path: {found_jd[0].absolute()})")
    jd_raw_text = _get_doc_rawText(found_jd[0])
    clear_text = _remove_white_space(jd_raw_text)
    #print(clear_text)
    jd_json_doc= _get_json_from_llm(clear_text,'JD',model,tokenizer)
    #print("jd_json_doc: ",jd_json_doc)
    jd_json = json.loads(jd_json_doc.replace("`","").replace("json","").strip())

    #print(jd_json)

    ranking_results = rank_resumes_with_openai(final_json_array, jd_json,key)

    if ranking_results:
        #print("\n--- Final Candidate Rankings ---")
        #print(json.dumps(ranking_results, indent=4))
        generate_ranking_report(json.dumps(ranking_results, indent=4),jd_json)
        