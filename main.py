"""
Main script for resume parsing, evaluation, and ranking.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Tuple
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import docx2txt
import PyPDF2
from pathlib import Path
from initialize import initialize_llm_from_env, get_model
import logging
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Change the global variable name
default_llm = initialize_llm_from_env()

def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of the given text."""
    return hashlib.sha256(text.encode()).hexdigest()

def parse_file_to_text(file_path: str) -> str:
    """Parse a file into text."""
    logging.info(f"Parsing file: {file_path}")
    if file_path.endswith('.docx'):
        text = docx2txt.process(file_path)
    elif file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ' '.join([page.extract_text() for page in reader.pages])
    else:  # .txt
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    logging.info(f"Successfully parsed file: {file_path}")
    return text

def save_json(data: Dict, file_path: str):
    """Save data as JSON to a file."""
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logging.info(f"Saved JSON to: {file_path}")

def load_json(file_path: str) -> Dict:
    """Load JSON data from a file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Loaded JSON from: {file_path}")
        return data
    logging.info(f"No JSON file found at: {file_path}")
    return None

def parse_resume_folder_to_paths(root_folder: str) -> List[str]:
    """Parse resume folder into list of resume file paths."""
    logging.info(f"Parsing resume folder: {root_folder}")
    resume_paths = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(('.docx', '.pdf', '.txt')):
                resume_paths.append(os.path.join(root, file))
    logging.info(f"Found {len(resume_paths)} resume files")
    return resume_paths

def parse_resume_path_to_text(resume_path: str) -> Tuple[str, str]:
    """Parse resume into text and compute its hash."""
    logging.info(f"Parsing resume: {resume_path}")
    text = parse_file_to_text(resume_path)
    file_hash = compute_hash(text)
    logging.info(f"Parsed resume {resume_path} with hash: {file_hash}")
    return file_hash, text

def parse_resume_text_to_dict(resume_hash: str, resume_text: str, resume_path: str, llm=default_llm) -> Dict:
    """Parse resume into dict using LLM."""
    logging.info(f"Parsing resume text to dict for hash: {resume_hash}")
    prompt = PromptTemplate(
        input_variables=["resume_text"],
        template="""
        Extract the following information from the resume text and format it as a JSON object with the exact keys specified:

        {{
            "first_name": "string",
            "last_name": "string",
            "email": "string",
            "phone": "string",
            "school": "string",
            "degree": "string",
            "degree_level": "string",
            "gpa": "number",
            "graduation_date": "string",
            "skills": ["string"],
            "years_experience": "number",
            "writing_quality": "number",
            "overall_quality": "number",
            "summary": "string"
        }}

        Resume Text:
        {resume_text}

        Provide the output as a valid JSON object without any additional formatting or code block markers.
        """
    )
    
    chain = (
        {"resume_text": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    result = chain.invoke(resume_text)
    
    try:
        resume_dict = json.loads(result)
        logging.info(f"Successfully parsed resume text to dict for hash: {resume_hash}")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing LLM output for hash {resume_hash}: {e}")
        logging.error(f"Raw LLM output: {result}")
        resume_dict = {}
    
    resume_dict['hash'] = resume_hash
    resume_dict['resume_path'] = resume_path
    
    return resume_dict

def parse_resume_paths_to_dicts(resume_paths: List[str], cache_folder: str = '_cache', llm=default_llm) -> List[Dict]:
    """Parse resume paths into resume dicts, using cache when available."""
    logging.info(f"Parsing {len(resume_paths)} resume paths to dicts")
    resume_dicts = []
    for path in resume_paths:
        file_hash, text = parse_resume_path_to_text(path)
        cache_path = os.path.join(cache_folder, f"{file_hash}.json")
        cached_dict = load_json(cache_path)
        if cached_dict:
            resume_dicts.append(cached_dict)
        else:
            resume_dict = parse_resume_text_to_dict(file_hash, text, path, llm)
            save_json(resume_dict, cache_path)
            resume_dicts.append(resume_dict)
    logging.info(f"Parsed {len(resume_dicts)} resume dicts")
    return resume_dicts

def parse_job_description_to_dict(job_description_text: str, job_description_path: str, llm=default_llm) -> Dict:
    """Parse job description into dict using LLM."""
    logging.info(f"Parsing job description: {job_description_path}")
    job_hash = compute_hash(job_description_text)
    prompt = PromptTemplate(
        input_variables=["job_description_text"],
        template="""
        Extract the following information from the job description and format it as a JSON object with the exact keys specified:

        {{
            "job_title": "string",
            "company": "string",
            "location": "string",
            "job_type": "string",
            "summary": "string",
            "skills": ["string"],
            "min_years_experience": "number",
            "min_degree": "string",
            "min_gpa": "number"
        }}

        Job Description:
        {job_description_text}

        Provide the output as a valid JSON object without any additional formatting or code block markers.
        """
    )
    
    chain = (
        {"job_description_text": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    result = chain.invoke(job_description_text)
    
    try:
        job_dict = json.loads(result)
        logging.info(f"Successfully parsed job description to dict: {job_description_path}")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing LLM output for job description {job_description_path}: {e}")
        logging.error(f"Raw LLM output: {result}")
        job_dict = {}
    
    job_dict['hash'] = job_hash
    job_dict['job_description_path'] = job_description_path
    
    return job_dict

def parse_job_description_path_to_dict(job_description_path: str, cache_folder: str = '_cache', llm=default_llm) -> Dict:
    """Parse job description into dict, using cache when available."""
    logging.info(f"Parsing job description path to dict: {job_description_path}")
    job_description_text = parse_file_to_text(job_description_path)
    job_hash = compute_hash(job_description_text)
    cache_path = os.path.join(cache_folder, f"{job_hash}.json")
    cached_dict = load_json(cache_path)
    if cached_dict:
        return cached_dict
    else:
        job_dict = parse_job_description_to_dict(job_description_text, job_description_path, llm)
        save_json(job_dict, cache_path)
        return job_dict

def evaluation_hash_path(resume_hash: str, job_description_hash: str, cache_folder: str = '_cache') -> str:
    """Compute the path for the cached evaluation."""
    evaluation_hash = compute_hash(f"{resume_hash}_{job_description_hash}")
    return os.path.join(cache_folder, f"eval_{evaluation_hash}.json")

def evaluate_resume_against_job_description(resume_dict: Dict, job_description_dict: Dict, cache_folder: str = '_cache', llm=default_llm) -> Dict:
    """Evaluate resume against job description using LLM, with caching."""
    logging.info(f"Evaluating resume {resume_dict['resume_path']} against job description {job_description_dict['job_description_path']}")
    
    # Check for cached evaluation
    cache_path = evaluation_hash_path(resume_dict['hash'], job_description_dict['hash'], cache_folder)
    cached_evaluation = load_json(cache_path)
    if cached_evaluation:
        logging.info("Using cached evaluation result")
        return cached_evaluation
    
    prompt = PromptTemplate(
        input_variables=["resume", "job_description"],
        template="""
        Evaluate the resume against the job description and provide the following information as a JSON object with the exact keys specified:

        {{
            "meets_minimum_requirements": "boolean",
            "num_skills_matched": "number",
            "match_score": "number",
            "feedback": "string"
        }}
        
        Resume:
        {resume}
        
        Job Description:
        {job_description}
        
        Provide the output as a valid JSON object without any additional formatting or code block markers.
        """
    )
    
    chain = (
        {"resume": lambda x: x["resume"], "job_description": lambda x: x["job_description"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = chain.invoke({"resume": json.dumps(resume_dict), "job_description": json.dumps(job_description_dict)})
    
    try:
        evaluation_dict = json.loads(result)
        logging.info(f"Successfully evaluated resume {resume_dict['resume_path']} against job description {job_description_dict['job_description_path']}")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing LLM output for resume {resume_dict['resume_path']} and job description {job_description_dict['job_description_path']}: {e}")
        logging.error(f"Raw LLM output: {result}")
        evaluation_dict = {}
    
    evaluation_dict['resume_hash'] = resume_dict['hash']
    evaluation_dict['job_description_hash'] = job_description_dict['hash']
    evaluation_dict['evaluation_hash'] = compute_hash(f"{resume_dict['hash']}_{job_description_dict['hash']}_{result}")
    
    # Save the evaluation to cache
    save_json(evaluation_dict, cache_path)
    
    return evaluation_dict

def evaluate_resumes_against_job_description(resume_dicts: List[Dict], job_description_dict: Dict, cache_folder: str = '_cache', llm=default_llm) -> List[Dict]:
    """Evaluate resumes against job description."""
    logging.info(f"Evaluating {len(resume_dicts)} resumes against job description {job_description_dict['job_description_path']}")
    evaluation_results = []
    for resume_dict in resume_dicts:
        evaluation = evaluate_resume_against_job_description(resume_dict, job_description_dict, cache_folder, llm)
        evaluation_results.append(evaluation)
    logging.info(f"Completed evaluation of {len(evaluation_results)} resumes")
    return evaluation_results

def rank_resume_evaluation_results(df: pd.DataFrame) -> pd.DataFrame:
    """Rank resumes by job description."""
    logging.info(f"Ranking {len(df)} resume evaluation results")
    
    # Create degree_level_rank column
    df['degree_level_rank'] = df['degree_level'].map({'B': 1, 'M': 2, 'PhD': 3})
    
    # Sort the DataFrame
    df_sorted = df.sort_values(
        by=['match_score', 'years_experience', 'degree_level_rank', 'gpa'],
        ascending=[False, False, False, False]
    )
    
    # Drop the temporary ranking column and reset index
    df_sorted = df_sorted.drop('degree_level_rank', axis=1)
    df_sorted = df_sorted.reset_index(drop=True)
    
    logging.info("Completed ranking of resume evaluation results")
    return df_sorted

def run_resume_evaluation_and_ranking(resume_paths: List[str], job_description_path: str, cache_folder: str = '_cache', llm=default_llm) -> pd.DataFrame:
    """Run the resume evaluation and ranking process."""
    logging.info("Starting resume evaluation and ranking process")
    
    # Parse resumes and job description
    resume_dicts = parse_resume_paths_to_dicts(resume_paths, cache_folder, llm)
    job_description_dict = parse_job_description_path_to_dict(job_description_path, cache_folder, llm)
    
    # Create a DataFrame from resume_dicts
    resume_df = pd.DataFrame(resume_dicts)
    
    # Evaluate resumes
    evaluation_results = evaluate_resumes_against_job_description(resume_dicts, job_description_dict, cache_folder, llm)
    
    # Create a DataFrame from evaluation results
    evaluation_df = pd.DataFrame(evaluation_results)
    
    # Merge resume information with evaluation results
    merged_df = pd.merge(evaluation_df, resume_df, left_on='resume_hash', right_on='hash')
    
    # Rank the merged results
    ranked_resumes = rank_resume_evaluation_results(merged_df)
    
    logging.info("Completed resume evaluation and ranking process")
    return ranked_resumes

def main(resume_folder: str = "resumes", job_description_path: str = "job_description.txt", output_path: str = "ranked_resumes.csv", cache_folder: str = '_cache', llm=default_llm) -> pd.DataFrame:
    """
    Main function to run the resume evaluation and ranking process.

    Args:
        resume_folder (str): Root folder path to resume files.
        job_description_path (str): Job description file path.
        output_path (str): Output file path for ranked resumes.
        cache_folder (str): Folder to store cached parsed information.
        llm: Language model to use for processing (default is the global default_llm).

    Returns:
        pd.DataFrame: Pandas DataFrame of ranked resumes.
    """
    logging.info(f"Starting main process with resume folder: {resume_folder}, job description: {job_description_path}")
    
    # Ensure cache folder exists
    os.makedirs(cache_folder, exist_ok=True)
    
    resume_paths = parse_resume_folder_to_paths(resume_folder)
    ranked_resumes = run_resume_evaluation_and_ranking(resume_paths, job_description_path, cache_folder, llm)
    
    # Save to CSV
    ranked_resumes.to_csv(output_path, index=False)
    logging.info(f"Saved ranked resumes to {output_path}")
    
    return ranked_resumes

if __name__ == "__main__":
    logging.info("Starting script execution")
    main(
        resume_folder="sample_resumes",
        job_description_path="job_descriptions/job_description_softeng.txt",
        output_path="sample_output.csv",
        llm=get_model("OLLAMA")
    )
    logging.info("Script execution completed")
