import os
from pprint import pformat
from initialize import initialize_llm_from_env, get_model
from main import (
    parse_file_to_text,
    parse_resume_text_to_dict,
    parse_job_description_to_dict,
    evaluate_resume_against_job_description
)

def sample_resume_evaluation(
    resume_path: str,
    job_description_path: str,
    output_file: str,
    llm=None
):
    # Use the default LLM if none is provided
    if llm is None:
        llm = initialize_llm_from_env()

    with open(output_file, 'w') as f:
        def write_output(message):
            f.write(message + '\n')
            f.flush()  # Ensure the message is written immediately

        write_output(f"Using LLM: {type(llm).__name__}")
        write_output("\n" + "="*50 + "\n")

        # Step 1: Parse resume to text
        write_output("Step 1: Parsing resume to text")
        resume_text = parse_file_to_text(resume_path)
        write_output(f"Resume text (first 500 characters):\n{resume_text[:500]}...\n")
        write_output("="*50 + "\n")

        # Step 2: Parse job description to text
        write_output("Step 2: Parsing job description to text")
        job_description_text = parse_file_to_text(job_description_path)
        write_output(f"Job description text (first 500 characters):\n{job_description_text[:500]}...\n")
        write_output("="*50 + "\n")

        # Step 3: LLM parsing of resume
        write_output("Step 3: LLM parsing of resume")
        resume_uuid = os.path.basename(resume_path)
        resume_dict = parse_resume_text_to_dict(resume_uuid, resume_text, resume_path, llm)
        write_output("Parsed resume dict:")
        write_output(pformat(resume_dict))
        write_output("\n" + "="*50 + "\n")

        # Step 4: LLM parsing of job description
        write_output("Step 4: LLM parsing of job description")
        job_dict = parse_job_description_to_dict(job_description_text, job_description_path, llm)
        write_output("Parsed job description dict:")
        write_output(pformat(job_dict))
        write_output("\n" + "="*50 + "\n")

        # Step 5: Evaluate resume against job description
        write_output("Step 5: Evaluating resume against job description")
        evaluation = evaluate_resume_against_job_description(resume_dict, job_dict, llm)
        write_output("Evaluation result:")
        write_output(pformat(evaluation))
        write_output("\n" + "="*50 + "\n")

        write_output("Sample resume evaluation complete!")

    print(f"Evaluation results have been written to {output_file}")

if __name__ == "__main__":
    # You can modify these paths as needed
    resume_path = "resumes/computer-scientist-resume-example.pdf"
    job_description_path = "job_descriptions/job_description_softeng.txt"
    output_file = "sample_evaluation_output.txt"
    llm = get_model("OLLAMA")
    sample_resume_evaluation(resume_path, job_description_path, output_file, llm)
