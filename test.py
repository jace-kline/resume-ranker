"""
Test file for resume parsing, evaluation, and ranking functions.
"""

import unittest
import pandas as pd
import os
import json
from main import (
    parse_resume_folder_to_paths,
    parse_file_to_text,
    parse_resume_text_to_dict,
    parse_resume_paths_to_dicts,
    parse_job_description_to_dict,
    parse_job_description_path_to_dict,
    evaluate_resume_against_job_description,
    evaluate_resumes_against_job_description,
    rank_resume_evaluation_results,
    run_resume_evaluation_and_ranking,
    save_parsed_dict,
    load_cached_dict
)

class TestResumeFunctions(unittest.TestCase):

    def setUp(self):
        self.cache_folder = '_test_cache'
        os.makedirs(self.cache_folder, exist_ok=True)

    def tearDown(self):
        # Clean up the test cache folder
        for file in os.listdir(self.cache_folder):
            os.remove(os.path.join(self.cache_folder, file))
        os.rmdir(self.cache_folder)

    def test_parse_resume_folder_to_paths(self):
        paths = parse_resume_folder_to_paths("test_resumes")
        self.assertIsInstance(paths, list)
        self.assertTrue(all(isinstance(path, str) for path in paths))

    def test_parse_file_to_text(self):
        text = parse_file_to_text("test_resumes/sample_resume.txt")
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_parse_resume_text_to_dict(self):
        text = parse_file_to_text("test_resumes/sample_resume.txt")
        resume_dict = parse_resume_text_to_dict("test_uuid", text, "test_resumes/sample_resume.txt")
        self.assertIsInstance(resume_dict, dict)
        self.assertIn('uuid', resume_dict)
        self.assertIn('first_name', resume_dict)
        self.assertIn('last_name', resume_dict)
        self.assertIn('email', resume_dict)
        self.assertIn('phone', resume_dict)
        self.assertIn('school', resume_dict)
        self.assertIn('degree', resume_dict)
        self.assertIn('degree_level', resume_dict)
        self.assertIn('gpa', resume_dict)
        self.assertIn('graduation_date', resume_dict)
        self.assertIn('skills', resume_dict)
        self.assertIn('years_experience', resume_dict)
        self.assertIn('writing_quality', resume_dict)
        self.assertIn('overall_quality', resume_dict)
        self.assertIn('summary', resume_dict)

    def test_parse_resume_paths_to_dicts(self):
        paths = parse_resume_folder_to_paths("test_resumes")
        resume_dicts = parse_resume_paths_to_dicts(paths, self.cache_folder)
        self.assertIsInstance(resume_dicts, list)
        self.assertTrue(all(isinstance(resume, dict) for resume in resume_dicts))

    def test_parse_job_description_to_dict(self):
        text = parse_file_to_text("test_job_description.txt")
        job_dict = parse_job_description_to_dict(text, "test_job_description.txt")
        self.assertIsInstance(job_dict, dict)
        self.assertIn('uuid', job_dict)
        self.assertIn('job_title', job_dict)
        self.assertIn('company', job_dict)
        self.assertIn('location', job_dict)
        self.assertIn('job_type', job_dict)
        self.assertIn('summary', job_dict)
        self.assertIn('skills', job_dict)
        self.assertIn('min_years_experience', job_dict)
        self.assertIn('min_degree', job_dict)
        self.assertIn('min_gpa', job_dict)

    def test_parse_job_description_path_to_dict(self):
        job_dict = parse_job_description_path_to_dict("test_job_description.txt", self.cache_folder)
        self.assertIsInstance(job_dict, dict)
        self.assertIn('uuid', job_dict)
        self.assertIn('job_title', job_dict)

    def test_evaluate_resume_against_job_description(self):
        resume_text = parse_file_to_text("test_resumes/sample_resume.txt")
        resume_dict = parse_resume_text_to_dict("test_uuid", resume_text, "test_resumes/sample_resume.txt")
        job_text = parse_file_to_text("test_job_description.txt")
        job_dict = parse_job_description_to_dict(job_text, "test_job_description.txt")
        evaluation = evaluate_resume_against_job_description(resume_dict, job_dict)
        self.assertIsInstance(evaluation, dict)
        self.assertIn('resume_uuid', evaluation)
        self.assertIn('job_description_uuid', evaluation)
        self.assertIn('meets_minimum_requirements', evaluation)
        self.assertIn('num_skills_matched', evaluation)
        self.assertIn('match_score', evaluation)
        self.assertIn('feedback', evaluation)

    def test_evaluate_resumes_against_job_description(self):
        resume_dicts = [
            parse_resume_text_to_dict("uuid1", parse_file_to_text("test_resumes/sample_resume.txt"), "test_resumes/sample_resume.txt"),
            parse_resume_text_to_dict("uuid2", parse_file_to_text("test_resumes/sample_resume2.txt"), "test_resumes/sample_resume2.txt")
        ]
        job_dict = parse_job_description_path_to_dict("test_job_description.txt", self.cache_folder)
        evaluations = evaluate_resumes_against_job_description(resume_dicts, job_dict)
        self.assertIsInstance(evaluations, list)
        self.assertTrue(all(isinstance(eval_dict, dict) for eval_dict in evaluations))

    def test_rank_resume_evaluation_results(self):
        evaluations = [
            {'match_score': 80, 'years_experience': 5, 'degree_level': 'M', 'gpa': 3.8},
            {'match_score': 90, 'years_experience': 3, 'degree_level': 'B', 'gpa': 3.5},
        ]
        ranked = rank_resume_evaluation_results(evaluations)
        self.assertIsInstance(ranked, pd.DataFrame)
        self.assertEqual(ranked.iloc[0]['match_score'], 90)

    def test_run_resume_evaluation_and_ranking(self):
        paths = parse_resume_folder_to_paths("test_resumes")
        ranked_resumes = run_resume_evaluation_and_ranking(paths, "test_job_description.txt", self.cache_folder)
        self.assertIsInstance(ranked_resumes, pd.DataFrame)
        self.assertGreater(len(ranked_resumes), 0)

    def test_save_and_load_cached_dict(self):
        test_dict = {"test_key": "test_value"}
        save_parsed_dict(test_dict, "test_file.txt", self.cache_folder)
        loaded_dict = load_cached_dict("test_file.txt", self.cache_folder)
        self.assertEqual(test_dict, loaded_dict)

if __name__ == '__main__':
    unittest.main()
